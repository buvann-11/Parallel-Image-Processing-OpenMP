/*
 * padding.c -- PART 1: padding for a CNN setting.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHY PADDING EXISTS
 * ---------------------------------------------------------------------------
 * A k x k convolution without padding shrinks an H x W feature map to
 * (H-k+1) x (W-k+1).  Stack a few layers and the spatial extent collapses.
 * Worse, a corner pixel of the input contributes to exactly one output while
 * a central pixel contributes to k*k of them, so the network sees the
 * interior of the image far more often than its border.  Padding fixes both.
 *
 * ---------------------------------------------------------------------------
 * PARALLELISATION
 * ---------------------------------------------------------------------------
 * Padding is a pure gather: every output element reads exactly one input
 * element (or a constant).  There are no dependencies whatsoever between
 * output elements, which makes this the simplest kernel in the project and a
 * useful control in the scaling study -- it is almost entirely
 * memory-bandwidth bound, so it is expected to scale WORSE than convolution,
 * not better.  A kernel that only moves bytes saturates the memory bus long
 * before it saturates the cores.  See report section 5.3.
 *
 * The parallel loop is `collapse(3)` over (batch, channel, output row).
 * Collapsing three loops matters: with a single feature map (n = c = 1), a
 * loop over `n` alone would have one iteration and could use only one
 * thread.  Collapsing flattens the whole iteration space so the work divides
 * evenly however the tensor is shaped.
 *
 * Threads write to distinct output rows.  Rows are at least 64-byte aligned
 * (see core.c), so two threads never share a cache line -- no false sharing.
 */

#include "imgproc.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Border index mapping                                               */
/* ------------------------------------------------------------------ */
/*
 * Maps a padded coordinate i (which may be negative, or >= n) back to the
 * source index it copies from.  Modular folding is used rather than repeated
 * reflection so that a padding width LARGER than the axis itself still works
 * -- numpy.pad refuses that case for 'reflect'.
 *
 * Returns -1 for PAD_ZEROS when the coordinate lies outside the image, which
 * the caller turns into a literal zero.
 */
static inline int map_index(int i, int n, pad_mode mode)
{
    if (i >= 0 && i < n) {
        return i;                       /* the common case: interior */
    }

    switch (mode) {
    case PAD_ZEROS:
        return -1;

    case PAD_REPLICATE:
        return (i < 0) ? 0 : n - 1;

    case PAD_CIRCULAR: {
        int m = i % n;
        return (m < 0) ? m + n : m;
    }

    case PAD_REFLECT: {
        /* Mirror about the border pixel without repeating it:
         *   ... c b | a b c d | c b ...
         * Period is 2n-2; positions past n-1 fold back. */
        int period, m;
        if (n == 1) {
            return 0;
        }
        period = 2 * n - 2;
        m = i % period;
        if (m < 0) {
            m += period;
        }
        return (m >= n) ? period - m : m;
    }

    default:
        return (i < 0) ? 0 : n - 1;
    }
}

/* ------------------------------------------------------------------ */
/* Shared row kernel                                                  */
/* ------------------------------------------------------------------ */
/*
 * Fills one output row.  Factored out so the serial and OpenMP versions run
 * byte-identical arithmetic -- if they differed, any speedup measurement
 * would be comparing two different computations.
 *
 * `src_y` is precomputed by the caller because it is constant along the row.
 */
static inline void pad_row(float *restrict dst, const float *restrict plane,
                           int src_y, int in_w, int out_w, int pad_w,
                           pad_mode mode)
{
    int ox;

    if (src_y < 0) {                    /* whole row lies in zero padding */
        memset(dst, 0, (size_t)out_w * sizeof(float));
        return;
    }

    plane += (size_t)src_y * in_w;

    /* Interior: a straight contiguous copy, which the compiler turns into a
     * vectorised memcpy.  Splitting it out keeps the per-pixel index
     * mapping off the fast path. */
    memcpy(dst + pad_w, plane, (size_t)in_w * sizeof(float));

    for (ox = 0; ox < pad_w; ++ox) {
        int sx = map_index(ox - pad_w, in_w, mode);
        dst[ox] = (sx < 0) ? 0.0f : plane[sx];
    }
    for (ox = pad_w + in_w; ox < out_w; ++ox) {
        int sx = map_index(ox - pad_w, in_w, mode);
        dst[ox] = (sx < 0) ? 0.0f : plane[sx];
    }
}

/* ------------------------------------------------------------------ */
/* Serial baseline                                                    */
/* ------------------------------------------------------------------ */
Tensor *pad2d_serial(const Tensor *in, int pad_h, int pad_w, pad_mode mode)
{
    Tensor *out;
    int out_h, out_w, n, c, oy;

    if (!in || pad_h < 0 || pad_w < 0 || mode < 0 || mode >= PAD_MODE_COUNT) {
        return NULL;
    }
    out_h = in->h + 2 * pad_h;
    out_w = in->w + 2 * pad_w;

    out = tensor_alloc(in->n, in->c, out_h, out_w);
    if (!out) {
        return NULL;
    }

    for (n = 0; n < in->n; ++n) {
        for (c = 0; c < in->c; ++c) {
            const float *plane = in->data
                + ((size_t)n * in->c + c) * in->h * in->w;
            float *dst_plane = out->data
                + ((size_t)n * out->c + c) * out_h * out_w;
            for (oy = 0; oy < out_h; ++oy) {
                int sy = map_index(oy - pad_h, in->h, mode);
                pad_row(dst_plane + (size_t)oy * out_w, plane, sy,
                        in->w, out_w, pad_w, mode);
            }
        }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* OpenMP version                                                     */
/* ------------------------------------------------------------------ */
Tensor *pad2d_omp(const Tensor *in, int pad_h, int pad_w, pad_mode mode)
{
    Tensor *out;
    int out_h, out_w;

    if (!in || pad_h < 0 || pad_w < 0 || mode < 0 || mode >= PAD_MODE_COUNT) {
        return NULL;
    }
    out_h = in->h + 2 * pad_h;
    out_w = in->w + 2 * pad_w;

    out = tensor_alloc(in->n, in->c, out_h, out_w);
    if (!out) {
        return NULL;
    }

    {
        const int N = in->n, C = in->c, IH = in->h, IW = in->w;
        const float *const src = in->data;
        float *const dst = out->data;
        int n, c, oy;

        /* collapse(3): flatten (n, c, oy) into a single iteration space so
         * the work divides evenly even for a single 1-channel image. */
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(runtime) default(none) \
        shared(src, dst) firstprivate(N, C, IH, IW, out_h, out_w, pad_h, pad_w, mode)
#endif
        for (n = 0; n < N; ++n) {
            for (c = 0; c < C; ++c) {
                for (oy = 0; oy < out_h; ++oy) {
                    const float *plane = src + ((size_t)n * C + c) * IH * IW;
                    float *row = dst + ((size_t)n * C + c) * out_h * out_w
                                     + (size_t)oy * out_w;
                    int sy = map_index(oy - pad_h, IH, mode);
                    pad_row(row, plane, sy, IW, out_w, pad_w, mode);
                }
            }
        }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Output-size arithmetic                                             */
/* ------------------------------------------------------------------ */
/*
 *   L_out = floor((L_in + pad_before + pad_after - dilation*(k-1) - 1) / stride) + 1
 *
 * dilation*(k-1)+1 is the "effective kernel size": dilation spreads the k
 * taps over a wider footprint without adding parameters.  Defined here once
 * and called by convolution and pooling alike, so the two can never disagree.
 */
int conv_output_size(int in_size, int kernel, int stride,
                     int pad_before, int pad_after, int dilation)
{
    int effective, numerator;

    if (in_size <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 ||
        pad_before < 0 || pad_after < 0) {
        return -1;
    }
    effective = dilation * (kernel - 1) + 1;
    numerator = in_size + pad_before + pad_after - effective;
    if (numerator < 0) {
        return -1;                      /* kernel does not fit */
    }
    return numerator / stride + 1;
}
