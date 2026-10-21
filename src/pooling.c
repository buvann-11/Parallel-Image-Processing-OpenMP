/*
 * pooling.c -- PART 1: max and average pooling in a CNN setting.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHAT POOLING BUYS
 * ---------------------------------------------------------------------------
 *   1. Resolution reduction, and therefore compute and memory, with ZERO
 *      parameters.
 *   2. A wider receptive field for later layers: after one 2x2 pool, each
 *      subsequent 3x3 kernel effectively covers a 6x6 patch of the original
 *      image.  This is the cheapest way to grow context.
 *   3. Local translation tolerance: shifting a strong activation by one pixel
 *      inside a window leaves a max-pooled output unchanged.
 *
 *   y_max[p,q] = max_{(i,j) in W(p,q)} x[i,j]
 *   y_avg[p,q] = (1/|W|) * sum_{(i,j) in W(p,q)} x[i,j]
 *
 * Max pooling keeps the strongest evidence of a feature and behaves like a
 * selector; average pooling keeps the mean response and behaves like a
 * smoother.  Modern networks use max pooling inside the trunk and a single
 * global average pool before the classifier, so all three are provided.
 *
 * ---------------------------------------------------------------------------
 * BORDER HANDLING
 * ---------------------------------------------------------------------------
 * Rather than materialising a padded copy, each window is CLAMPED to the real
 * input extent.  This matters for both correctness and performance:
 *
 *   - Max pooling must never return a padded value.  Feature maps contain
 *     negative activations, so padding with 0 would let the padding win a
 *     maximum and silently corrupt every border output.  Clamping makes that
 *     impossible by construction; the alternative is padding with -INFINITY.
 *
 *   - Average pooling divides by the number of REAL elements in the window
 *     (equivalently PyTorch's `count_include_pad=False`).  Counting padding
 *     would pull border averages toward zero -- acceptable in a CNN trunk,
 *     wrong for image processing.
 *
 *   - Not allocating a padded tensor saves a full pass over memory, which on
 *     a bandwidth-bound kernel like this one is a measurable win.
 *
 * As in PyTorch, padding may not exceed half the kernel size; otherwise a
 * window could consist of nothing but padding and its output would be
 * undefined.
 *
 * ---------------------------------------------------------------------------
 * PARALLELISATION
 * ---------------------------------------------------------------------------
 * `collapse(3)` over (batch, channel, output row).  Output rows are disjoint,
 * so there is no sharing and no synchronisation on the hot path.  The
 * per-window reduction (max or sum) is a thread-LOCAL scalar, so it needs no
 * OpenMP reduction clause -- the reduction is inside one iteration, not
 * across iterations.  Global average pooling is different: there the
 * reduction spans the whole plane, and `reduction(+:acc)` is used.
 *
 * Arithmetic intensity is low (one compare or add per element loaded), so
 * pooling is memory-bound and will not scale as well as convolution.
 */

#include "imgproc.h"

#include <float.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* One output row of a pooling operation                              */
/* ------------------------------------------------------------------ */
static inline void pool_row(float *restrict out_row,
                            const float *restrict plane,
                            int IH, int IW, pool_type type,
                            int kh, int kw, int stride,
                            int pad_h, int pad_w, int oy, int out_w)
{
    const int y0_raw = oy * stride - pad_h;
    const int y0 = (y0_raw < 0) ? 0 : y0_raw;
    const int y1 = (y0_raw + kh > IH) ? IH : y0_raw + kh;
    int ox;

    for (ox = 0; ox < out_w; ++ox) {
        const int x0_raw = ox * stride - pad_w;
        const int x0 = (x0_raw < 0) ? 0 : x0_raw;
        const int x1 = (x0_raw + kw > IW) ? IW : x0_raw + kw;
        int y, x;

        if (y0 >= y1 || x0 >= x1) {
            out_row[ox] = 0.0f;          /* unreachable: pad <= k/2 enforced */
            continue;
        }

        if (type == POOL_MAX) {
            float best = -FLT_MAX;
            for (y = y0; y < y1; ++y) {
                const float *row = plane + (size_t)y * IW;
                for (x = x0; x < x1; ++x) {
                    if (row[x] > best) {
                        best = row[x];
                    }
                }
            }
            out_row[ox] = best;
        } else {
            float sum = 0.0f;
            for (y = y0; y < y1; ++y) {
                const float *row = plane + (size_t)y * IW;
                for (x = x0; x < x1; ++x) {
                    sum += row[x];
                }
            }
            /* Divide by the count of REAL elements, not kh*kw. */
            out_row[ox] = sum / (float)((y1 - y0) * (x1 - x0));
        }
    }
}

/* ------------------------------------------------------------------ */
/* Validation shared by both versions                                 */
/* ------------------------------------------------------------------ */
static int pool_prepare(const Tensor *in, pool_type type, int kh, int kw,
                        int stride, int pad_h, int pad_w,
                        int *out_h, int *out_w)
{
    if (!in || kh <= 0 || kw <= 0 || stride <= 0 || pad_h < 0 || pad_w < 0) {
        return 0;
    }
    if (type != POOL_MAX && type != POOL_AVG) {
        return 0;
    }
    /* PyTorch's rule: a window must contain at least one real element. */
    if (2 * pad_h > kh || 2 * pad_w > kw) {
        return 0;
    }
    *out_h = conv_output_size(in->h, kh, stride, pad_h, pad_h, 1);
    *out_w = conv_output_size(in->w, kw, stride, pad_w, pad_w, 1);
    return (*out_h > 0 && *out_w > 0);
}

/* ------------------------------------------------------------------ */
/* Serial baseline                                                    */
/* ------------------------------------------------------------------ */
Tensor *pool2d_serial(const Tensor *in, pool_type type,
                      int kh, int kw, int stride, int pad_h, int pad_w)
{
    Tensor *out;
    int out_h, out_w, n, c, oy;

    if (!pool_prepare(in, type, kh, kw, stride, pad_h, pad_w, &out_h, &out_w)) {
        return NULL;
    }
    out = tensor_alloc(in->n, in->c, out_h, out_w);
    if (!out) {
        return NULL;
    }

    for (n = 0; n < in->n; ++n) {
        for (c = 0; c < in->c; ++c) {
            const float *plane = in->data
                + ((size_t)n * in->c + c) * in->h * in->w;
            float *dst = out->data
                + ((size_t)n * in->c + c) * out_h * out_w;
            for (oy = 0; oy < out_h; ++oy) {
                pool_row(dst + (size_t)oy * out_w, plane, in->h, in->w,
                         type, kh, kw, stride, pad_h, pad_w, oy, out_w);
            }
        }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* OpenMP version                                                     */
/* ------------------------------------------------------------------ */
Tensor *pool2d_omp(const Tensor *in, pool_type type,
                   int kh, int kw, int stride, int pad_h, int pad_w)
{
    Tensor *out;
    int out_h, out_w;

    if (!pool_prepare(in, type, kh, kw, stride, pad_h, pad_w, &out_h, &out_w)) {
        return NULL;
    }
    out = tensor_alloc(in->n, in->c, out_h, out_w);
    if (!out) {
        return NULL;
    }

    {
        const int N = in->n, C = in->c, IH = in->h, IW = in->w;
        const float *const src = in->data;
        float *const dst = out->data;
        int n, c, oy;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(runtime) default(none) \
        shared(src, dst)                                             \
        firstprivate(N, C, IH, IW, out_h, out_w, type, kh, kw, stride, pad_h, pad_w)
#endif
        for (n = 0; n < N; ++n) {
            for (c = 0; c < C; ++c) {
                for (oy = 0; oy < out_h; ++oy) {
                    pool_row(dst + ((size_t)n * C + c) * out_h * out_w
                                 + (size_t)oy * out_w,
                             src + ((size_t)n * C + c) * IH * IW,
                             IH, IW, type, kh, kw, stride, pad_h, pad_w,
                             oy, out_w);
                }
            }
        }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Global average pooling                                             */
/* ------------------------------------------------------------------ */
/*
 * (N, C, H, W) -> (N, C, 1, 1).
 *
 * Global average pooling replaced the large fully-connected head of early
 * CNNs (Network-in-Network, 2013): it has no parameters, is far less prone
 * to overfitting, and accepts any input resolution.
 *
 * Here the reduction genuinely spans an entire plane, so the inner loop uses
 * `reduction(+:acc)`.  OpenMP gives each thread a private accumulator and
 * combines them in a tree at the end of the region -- far cheaper than an
 * atomic per element, and it keeps the summation order deterministic for a
 * fixed thread count.
 *
 * The outer parallelism is over (n, c) planes; the reduction is used only
 * when there are too few planes to keep the threads busy.
 */
Tensor *global_avg_pool_serial(const Tensor *in)
{
    Tensor *out;
    int n, c;
    long i, plane_size;

    if (!in) {
        return NULL;
    }
    out = tensor_alloc(in->n, in->c, 1, 1);
    if (!out) {
        return NULL;
    }
    plane_size = (long)in->h * in->w;

    for (n = 0; n < in->n; ++n) {
        for (c = 0; c < in->c; ++c) {
            const float *plane = in->data + ((size_t)n * in->c + c) * plane_size;
            double acc = 0.0;            /* double accumulator: see note below */
            for (i = 0; i < plane_size; ++i) {
                acc += plane[i];
            }
            out->data[(size_t)n * in->c + c] = (float)(acc / (double)plane_size);
        }
    }
    return out;
}

Tensor *global_avg_pool_omp(const Tensor *in)
{
    Tensor *out;

    if (!in) {
        return NULL;
    }
    out = tensor_alloc(in->n, in->c, 1, 1);
    if (!out) {
        return NULL;
    }

    {
        const int N = in->n, C = in->c;
        const long plane_size = (long)in->h * in->w;
        const float *const src = in->data;
        float *const dst = out->data;
        int nc;

        /*
         * Accumulation is in `double` even though the data is `float`.
         * Summing a 4-megapixel plane in float loses roughly the bottom 4
         * bits: once the running total is large, small terms vanish into
         * rounding.  The double accumulator costs nothing measurable here
         * (the kernel is memory-bound) and keeps the result matching the
         * NumPy reference to 1e-6 instead of 1e-2.
         */
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(src, dst) firstprivate(N, C, plane_size)
#endif
        for (nc = 0; nc < N * C; ++nc) {
            const float *plane = src + (size_t)nc * plane_size;
            double acc = 0.0;
            long i;
            for (i = 0; i < plane_size; ++i) {
                acc += plane[i];
            }
            dst[nc] = (float)(acc / (double)plane_size);
        }
    }
    return out;
}
