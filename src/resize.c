/*
 * resize.c -- PART 2: image resizing by separable resampling.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * RESIZING IS RESAMPLING
 * ---------------------------------------------------------------------------
 * Resizing is not "picking pixels".  Each output pixel is a weighted average
 * of nearby input pixels, the weights coming from a reconstruction kernel k:
 *
 *   y[p] = sum_x k((x + 0.5 - c_p) / sigma) * x[x],   c_p = (p + 0.5) * L_in/L_out
 *
 * The +0.5 terms implement the half-pixel convention: pixels are unit squares
 * and their CENTRES are aligned.  This is what Pillow, OpenCV and modern
 * PyTorch do.
 *
 *   nearest   support --    no interpolation; the only correct choice for
 *                           label masks, since it never invents a class value
 *   bilinear  support 1     triangle kernel; smooth and cheap
 *   bicubic   support 2     Keys cubic a = -0.5; sharper, slight ringing
 *
 * ---------------------------------------------------------------------------
 * ANTI-ALIASING -- the part that is usually wrong
 * ---------------------------------------------------------------------------
 * Shrinking an image is DECIMATION.  Sampling a 1000-pixel row at 100
 * positions with a one-pixel-wide kernel reads about 10% of the input; the
 * rest does not vanish, it folds back as moire.  The fix is to widen the
 * kernel by the downscaling factor so every input pixel contributes:
 *
 *   sigma = max(1, L_in / L_out)
 *
 * Pillow has always done this; torch.nn.functional.interpolate does NOT
 * unless asked.  `antialias` selects the behaviour so the report can show
 * both.
 *
 * ---------------------------------------------------------------------------
 * SEPARABILITY AND THE TWO-PHASE PARALLEL STRUCTURE
 * ---------------------------------------------------------------------------
 * All three kernels are separable, so a 2-D resize is two 1-D passes:
 *
 *     (H, W)  --horizontal-->  (H, outW)  --vertical-->  (outH, outW)
 *
 * For a P x Q output with T taps per axis this costs O(PQT) instead of
 * O(PQT^2).  At T = 4 (bicubic) that is a 4x reduction in arithmetic before
 * a single thread is spawned -- a reminder that algorithmic choice dominates
 * parallelisation.  A 4x algorithmic win beats a 4-core machine, and the two
 * multiply.
 *
 * Both passes are parallelised over OUTPUT ROWS, which are disjoint, so no
 * synchronisation is needed.  The two passes are separated by an implicit
 * barrier (the end of the first parallel region): the vertical pass reads
 * rows the horizontal pass wrote, so it genuinely cannot start early.  This
 * is the one real data dependency in the whole project, and it is exactly
 * what a barrier is for.
 *
 * The weight tables are computed ONCE, sequentially, before either pass.
 * They are O(outH*T + outW*T) -- a few kilobytes -- so parallelising their
 * construction would cost more in thread startup than it could ever save.
 * Knowing what NOT to parallelise is part of the exercise.
 */

#include "imgproc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Reconstruction kernels                                             */
/* ------------------------------------------------------------------ */
static inline float kernel_triangle(float t)
{
    t = fabsf(t);
    return (t < 1.0f) ? (1.0f - t) : 0.0f;
}

/* Keys' cubic with a = -0.5 (Catmull-Rom): the variant used by Pillow,
 * OpenCV and PyTorch. */
static inline float kernel_cubic(float t)
{
    const float a = -0.5f;
    float x = fabsf(t), x2, x3;

    x2 = x * x;
    x3 = x2 * x;
    if (x < 1.0f) {
        return (a + 2.0f) * x3 - (a + 3.0f) * x2 + 1.0f;
    }
    if (x < 2.0f) {
        return a * x3 - 5.0f * a * x2 + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

static inline float kernel_eval(float t, resize_method method)
{
    switch (method) {
    case RESIZE_BILINEAR: return kernel_triangle(t);
    case RESIZE_BICUBIC:  return kernel_cubic(t);
    default:              return 0.0f;   /* nearest handled separately */
    }
}

static inline float kernel_support(resize_method method)
{
    switch (method) {
    case RESIZE_BILINEAR: return 1.0f;
    case RESIZE_BICUBIC:  return 2.0f;
    default:              return 0.5f;
    }
}

/* ------------------------------------------------------------------ */
/* Weight table for one axis                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    int taps;       /* number of source samples per output sample */
    int *index;     /* out_size * taps source indices, clamped in range */
    float *weight;  /* out_size * taps weights, each row summing to 1 */
} Contributions;

static void contributions_free(Contributions *c)
{
    if (!c) {
        return;
    }
    free(c->index);
    free(c->weight);
    c->index = NULL;
    c->weight = NULL;
}

/*
 * Samples falling outside the image contribute nothing and the surviving
 * weights are renormalised -- equivalent to Pillow's border handling.  The
 * renormalisation guarantees the PARTITION OF UNITY property: every output's
 * weights sum to exactly 1, so a constant image resizes to the same constant
 * and mean brightness is preserved.  Without it, resizing would subtly
 * darken every border.
 */
static int contributions_build(Contributions *c, int in_size, int out_size,
                               resize_method method, int antialias)
{
    /*
     * The weight table is built once per resize -- O(out_size * taps), a few
     * kilobytes -- so it costs nothing to compute the sample geometry in
     * `double` even though the pixel data itself is `float`.
     *
     * This is not pedantry.  With float32, `floorf((p + 0.5f) * scale)` can
     * land on the wrong side of an integer boundary when the product falls
     * within a few ULP of a whole number, which happens often for awkward
     * ratios such as 256/97.  For nearest-neighbour that picks a different
     * source pixel outright, and on a high-contrast edge the result differs
     * by the full 255 levels.  Cross-checking against the float64 reference
     * implementation is what exposed it.
     */
    const double scale = (double)in_size / (double)out_size;
    double filter_scale = 1.0, support, scaled_support;
    int p, t;

    memset(c, 0, sizeof(*c));

    if (method == RESIZE_NEAREST) {
        c->taps = 1;
        c->index = (int *)malloc((size_t)out_size * sizeof(int));
        c->weight = (float *)malloc((size_t)out_size * sizeof(float));
        if (!c->index || !c->weight) {
            contributions_free(c);
            return 0;
        }
        for (p = 0; p < out_size; ++p) {
            int s = (int)floor(((double)p + 0.5) * scale);
            if (s < 0) s = 0;
            if (s > in_size - 1) s = in_size - 1;
            c->index[p] = s;
            c->weight[p] = 1.0f;
        }
        return 1;
    }

    if (antialias && scale > 1.0) {
        filter_scale = scale;            /* widen when downscaling */
    }
    support = (double)kernel_support(method);
    scaled_support = support * filter_scale;

    c->taps = (int)ceil(scaled_support * 2.0) + 1;
    if (c->taps < 1) {
        c->taps = 1;
    }
    c->index = (int *)malloc((size_t)out_size * c->taps * sizeof(int));
    c->weight = (float *)malloc((size_t)out_size * c->taps * sizeof(float));
    if (!c->index || !c->weight) {
        contributions_free(c);
        return 0;
    }

    for (p = 0; p < out_size; ++p) {
        const double centre = ((double)p + 0.5) * scale;
        const int left = (int)floor(centre - scaled_support + 0.5);
        int *idx = c->index + (size_t)p * c->taps;
        float *wts = c->weight + (size_t)p * c->taps;
        float total = 0.0f;

        for (t = 0; t < c->taps; ++t) {
            const int pos = left + t;
            float w;

            if (pos < 0 || pos > in_size - 1) {
                w = 0.0f;                /* outside: contributes nothing */
            } else {
                w = kernel_eval((float)(((double)pos + 0.5 - centre)
                                        / filter_scale), method);
            }
            idx[t] = (pos < 0) ? 0 : (pos > in_size - 1 ? in_size - 1 : pos);
            wts[t] = w;
            total += w;
        }

        if (fabsf(total) < 1e-12f) {
            /* Degenerate row (only possible for pathological sizes): fall
             * back to nearest so the output stays defined rather than NaN. */
            int s = (int)floor(centre);
            if (s < 0) s = 0;
            if (s > in_size - 1) s = in_size - 1;
            for (t = 0; t < c->taps; ++t) {
                idx[t] = s;
                wts[t] = (t == 0) ? 1.0f : 0.0f;
            }
        } else {
            const float inv = 1.0f / total;
            for (t = 0; t < c->taps; ++t) {
                wts[t] *= inv;
            }
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* The two passes                                                     */
/* ------------------------------------------------------------------ */
/* Horizontal: (H, W, C) -> (H, outW, C), one source row at a time. */
static inline void resize_row_h(float *restrict dst, const float *restrict src,
                                int out_w, int channels, const Contributions *c)
{
    int ox, t, ch;

    for (ox = 0; ox < out_w; ++ox) {
        const int *idx = c->index + (size_t)ox * c->taps;
        const float *wts = c->weight + (size_t)ox * c->taps;
        float *o = dst + (size_t)ox * channels;

        for (ch = 0; ch < channels; ++ch) {
            o[ch] = 0.0f;
        }
        for (t = 0; t < c->taps; ++t) {
            const float w = wts[t];
            const float *s = src + (size_t)idx[t] * channels;
            if (w == 0.0f) {
                continue;
            }
            for (ch = 0; ch < channels; ++ch) {
                o[ch] += w * s[ch];
            }
        }
    }
}

/* Vertical: (H, outW, C) -> (outH, outW, C), one destination row at a time. */
static inline void resize_row_v(float *restrict dst, const float *restrict mid,
                                int out_w, int channels, int oy,
                                const Contributions *c)
{
    const int *idx = c->index + (size_t)oy * c->taps;
    const float *wts = c->weight + (size_t)oy * c->taps;
    const size_t row_floats = (size_t)out_w * channels;
    size_t i;
    int t;

    for (i = 0; i < row_floats; ++i) {
        dst[i] = 0.0f;
    }
    for (t = 0; t < c->taps; ++t) {
        const float w = wts[t];
        const float *s = mid + (size_t)idx[t] * row_floats;
        if (w == 0.0f) {
            continue;
        }
        /* Unit-stride accumulation over a whole row: vectorises cleanly. */
#ifdef _OPENMP
#pragma omp simd
#endif
        for (i = 0; i < row_floats; ++i) {
            dst[i] += w * s[i];
        }
    }
}


/* ------------------------------------------------------------------ */
/* Pass drivers                                                       */
/* ------------------------------------------------------------------ */
/*
 * The serial and OpenMP drivers are separate functions rather than one
 * function with `#pragma omp parallel for if(parallel)`.  With an `if`
 * clause GCC still emits the libgomp loop protocol even when the condition
 * is false, and that overhead would be charged to the sequential baseline,
 * quietly inflating every speedup figure in the report.  See the same note
 * in normalize.c.
 */
static void pass_h_serial(const float *src, float *mid, int IH, int IW, int C,
                          int out_w, const Contributions *cx)
{
    int y;
    for (y = 0; y < IH; ++y) {
        resize_row_h(mid + (size_t)y * out_w * C,
                     src + (size_t)y * IW * C, out_w, C, cx);
    }
}

static void pass_h_omp(const float *src, float *mid, int IH, int IW, int C,
                       int out_w, const Contributions *cx)
{
    int y;
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(src, mid, cx) firstprivate(IH, IW, C, out_w)
#endif
    for (y = 0; y < IH; ++y) {
        resize_row_h(mid + (size_t)y * out_w * C,
                     src + (size_t)y * IW * C, out_w, C, cx);
    }
}

static void pass_v_serial(const float *mid, float *dst, int out_h, int out_w,
                          int C, const Contributions *cy)
{
    int oy;
    for (oy = 0; oy < out_h; ++oy) {
        resize_row_v(dst + (size_t)oy * out_w * C, mid, out_w, C, oy, cy);
    }
}

static void pass_v_omp(const float *mid, float *dst, int out_h, int out_w,
                       int C, const Contributions *cy)
{
    int oy;
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(mid, dst, cy) firstprivate(C, out_h, out_w)
#endif
    for (oy = 0; oy < out_h; ++oy) {
        resize_row_v(dst + (size_t)oy * out_w * C, mid, out_w, C, oy, cy);
    }
}

/* ------------------------------------------------------------------ */
/* Driver                                                             */
/* ------------------------------------------------------------------ */
static Image *resize_impl(const Image *in, int out_h, int out_w,
                          resize_method method, int antialias, int parallel)
{
    Image *out = NULL;
    float *mid = NULL;
    Contributions ch_x, ch_y;
    int ok;

    if (!in || out_h <= 0 || out_w <= 0 ||
        method < 0 || method >= RESIZE_METHOD_COUNT) {
        return NULL;
    }
#ifndef _OPENMP
    (void)parallel;                      /* no parallel regions to gate */
#endif

    memset(&ch_x, 0, sizeof(ch_x));
    memset(&ch_y, 0, sizeof(ch_y));

    /* Phase 0: weight tables, computed once, sequentially. */
    ok = contributions_build(&ch_x, in->width, out_w, method, antialias) &&
         contributions_build(&ch_y, in->height, out_h, method, antialias);
    if (!ok) {
        contributions_free(&ch_x);
        contributions_free(&ch_y);
        return NULL;
    }

    out = image_alloc(out_h, out_w, in->channels);
    mid = (float *)malloc((size_t)in->height * out_w * in->channels
                          * sizeof(float));
    if (!out || !mid) {
        image_free(out);
        free(mid);
        contributions_free(&ch_x);
        contributions_free(&ch_y);
        return NULL;
    }

    if (parallel) {
        pass_h_omp(in->data, mid, in->height, in->width, in->channels,
                   out_w, &ch_x);
        /* Implicit barrier at the end of that parallel region: the vertical
         * pass reads rows the horizontal pass produced, so it genuinely
         * cannot start early.  This is the only real data dependency in the
         * project, and a barrier is exactly what it calls for. */
        pass_v_omp(mid, out->data, out_h, out_w, in->channels, &ch_y);
    } else {
        pass_h_serial(in->data, mid, in->height, in->width, in->channels,
                      out_w, &ch_x);
        pass_v_serial(mid, out->data, out_h, out_w, in->channels, &ch_y);
    }

    free(mid);
    contributions_free(&ch_x);
    contributions_free(&ch_y);
    return out;
}

Image *resize_serial(const Image *in, int out_h, int out_w,
                     resize_method method, int antialias)
{
    return resize_impl(in, out_h, out_w, method, antialias, 0);
}

Image *resize_omp(const Image *in, int out_h, int out_w,
                  resize_method method, int antialias)
{
    return resize_impl(in, out_h, out_w, method, antialias, 1);
}
