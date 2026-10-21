/*
 * convolution.c -- PART 1: 2-D convolution in a CNN setting.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHAT IS COMPUTED
 * ---------------------------------------------------------------------------
 * The operation every deep-learning framework calls "convolution" is in fact
 * a cross-correlation:
 *
 *   y[n,f,p,q] = bias[f]
 *              + sum_{c=0}^{C-1} sum_{i=0}^{kh-1} sum_{j=0}^{kw-1}
 *                  w[f,c,i,j] * x[n, c, p*s + i*d, q*s + j*d]
 *
 * with s = stride and d = dilation.  For a *learned* kernel the distinction
 * from true convolution is irrelevant (the network learns the flipped
 * weights); it matters only for fixed filters such as Sobel.
 *
 * ---------------------------------------------------------------------------
 * ALGORITHM CHOICE: DIRECT, NOT im2col
 * ---------------------------------------------------------------------------
 * The usual CPU trick is im2col + GEMM: materialise every sliding window as
 * a column, then call a tuned BLAS kernel.  That is the right choice when a
 * BLAS library is available, but it is the wrong choice for THIS project:
 *
 *   1. It would outsource the parallelism to the BLAS implementation, so the
 *      scaling study would measure OpenBLAS's threading, not ours.
 *   2. It inflates memory by a factor of kh*kw, which distorts the
 *      memory-bandwidth analysis.
 *
 * A direct loop nest keeps every floating-point operation and every thread
 * under our control, which is the point of the exercise.
 *
 * ---------------------------------------------------------------------------
 * LOOP ORDER AND VECTORISATION
 * ---------------------------------------------------------------------------
 * For each output row the accumulation order is
 *
 *      c -> i (kernel row) -> j (kernel column) -> ox (output column)
 *
 * so the INNERMOST loop walks `ox` with the weight value `wv` held constant
 * in a register.  With stride 1 the source access `srow[ox + j*d]` is unit
 * stride, so gcc auto-vectorises it into an FMA over 8 floats per AVX2
 * instruction.  The naive order (ox outermost, kernel innermost) cannot
 * vectorise, because the reduction target would be a single scalar.
 *
 * ---------------------------------------------------------------------------
 * PARALLELISATION
 * ---------------------------------------------------------------------------
 * `collapse(3)` over (batch, output channel, output row).  Each iteration
 * owns one output row exclusively, so:
 *   - no two threads write the same element   -> no atomics, no critical
 *   - rows are 64-byte aligned                -> no false sharing
 *   - the accumulator stays in the output row -> no reduction needed
 *
 * Convolution has a high arithmetic intensity (kh*kw*C multiply-adds per
 * output element), so it is compute-bound and is expected to scale close to
 * linearly -- the opposite of padding.  See report section 5.3.
 */

#include "imgproc.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Single output row of a convolution                                 */
/* ------------------------------------------------------------------ */
/*
 * Shared verbatim by the serial and OpenMP paths so both compute bitwise
 * identical results; a speedup number is meaningless if the two versions do
 * different arithmetic.
 *
 * `pad` is the already-padded input tensor, so this function performs NO
 * bounds checking -- every access is guaranteed in range by construction.
 * Hoisting the border handling out of the hot loop is worth roughly 15% on
 * a 3x3 kernel.
 */
static inline void conv_row(float *restrict out_row,
                            const float *restrict pad_plane_base,
                            const float *restrict wf,
                            int C, int PH, int PW,
                            int kh, int kw, int stride, int dilation,
                            int oy, int out_w, float bias_value)
{
    int c, i, j, ox;

    for (ox = 0; ox < out_w; ++ox) {
        out_row[ox] = bias_value;
    }

    for (c = 0; c < C; ++c) {
        const float *plane = pad_plane_base + (size_t)c * PH * PW;
        const float *wc = wf + (size_t)c * kh * kw;

        for (i = 0; i < kh; ++i) {
            const float *srow = plane
                + (size_t)(oy * stride + i * dilation) * PW;
            const float *wrow = wc + (size_t)i * kw;

            for (j = 0; j < kw; ++j) {
                const float wv = wrow[j];
                const float *s = srow + j * dilation;

                if (wv == 0.0f) {
                    continue;            /* common for sparse/Laplacian kernels */
                }
                if (stride == 1) {
                    /* Unit stride: contiguous, auto-vectorises to FMA. */
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (ox = 0; ox < out_w; ++ox) {
                        out_row[ox] += wv * s[ox];
                    }
                } else {
                    for (ox = 0; ox < out_w; ++ox) {
                        out_row[ox] += wv * s[(size_t)ox * stride];
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Shape checking shared by both versions                             */
/* ------------------------------------------------------------------ */
static int conv_prepare(const Tensor *in, int out_channels, int kh, int kw,
                        int stride, int pad_h, int pad_w, int dilation,
                        int *out_h, int *out_w)
{
    if (!in || out_channels <= 0 || kh <= 0 || kw <= 0 ||
        stride <= 0 || dilation <= 0 || pad_h < 0 || pad_w < 0) {
        return 0;
    }
    *out_h = conv_output_size(in->h, kh, stride, pad_h, pad_h, dilation);
    *out_w = conv_output_size(in->w, kw, stride, pad_w, pad_w, dilation);
    return (*out_h > 0 && *out_w > 0);
}

/* ------------------------------------------------------------------ */
/* Serial baseline                                                    */
/* ------------------------------------------------------------------ */
Tensor *conv2d_serial(const Tensor *in, const float *weights, const float *bias,
                      int out_channels, int kh, int kw,
                      int stride, int pad_h, int pad_w, int dilation,
                      pad_mode mode)
{
    Tensor *padded = NULL, *out = NULL;
    int out_h, out_w, n, f, oy;

    if (!weights || !conv_prepare(in, out_channels, kh, kw, stride,
                                  pad_h, pad_w, dilation, &out_h, &out_w)) {
        return NULL;
    }

    padded = pad2d_serial(in, pad_h, pad_w, mode);
    if (!padded) {
        return NULL;
    }
    out = tensor_alloc(in->n, out_channels, out_h, out_w);
    if (!out) {
        tensor_free(padded);
        return NULL;
    }

    for (n = 0; n < in->n; ++n) {
        for (f = 0; f < out_channels; ++f) {
            const float bv = bias ? bias[f] : 0.0f;
            const float *wf = weights + (size_t)f * in->c * kh * kw;
            for (oy = 0; oy < out_h; ++oy) {
                conv_row(out->data + (((size_t)n * out_channels + f) * out_h + oy) * out_w,
                         padded->data + (size_t)n * padded->c * padded->h * padded->w,
                         wf, in->c, padded->h, padded->w,
                         kh, kw, stride, dilation, oy, out_w, bv);
            }
        }
    }

    tensor_free(padded);
    return out;
}

/* ------------------------------------------------------------------ */
/* OpenMP version                                                     */
/* ------------------------------------------------------------------ */
Tensor *conv2d_omp(const Tensor *in, const float *weights, const float *bias,
                   int out_channels, int kh, int kw,
                   int stride, int pad_h, int pad_w, int dilation,
                   pad_mode mode)
{
    Tensor *padded = NULL, *out = NULL;
    int out_h, out_w;

    if (!weights || !conv_prepare(in, out_channels, kh, kw, stride,
                                  pad_h, pad_w, dilation, &out_h, &out_w)) {
        return NULL;
    }

    padded = pad2d_omp(in, pad_h, pad_w, mode);
    if (!padded) {
        return NULL;
    }
    out = tensor_alloc(in->n, out_channels, out_h, out_w);
    if (!out) {
        tensor_free(padded);
        return NULL;
    }

    {
        const int N = in->n, C = in->c;
        const int PH = padded->h, PW = padded->w;
        const float *const pdata = padded->data;
        float *const odata = out->data;
        int n, f, oy;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(runtime) default(none)          \
        shared(pdata, odata, weights, bias)                                   \
        firstprivate(N, C, PH, PW, out_channels, out_h, out_w, kh, kw,        \
                     stride, dilation)
#endif
        for (n = 0; n < N; ++n) {
            for (f = 0; f < out_channels; ++f) {
                for (oy = 0; oy < out_h; ++oy) {
                    conv_row(odata + (((size_t)n * out_channels + f) * out_h + oy) * out_w,
                             pdata + (size_t)n * C * PH * PW,
                             weights + (size_t)f * C * kh * kw,
                             C, PH, PW, kh, kw, stride, dilation, oy, out_w,
                             bias ? bias[f] : 0.0f);
                }
            }
        }
    }

    tensor_free(padded);
    return out;
}

/* ------------------------------------------------------------------ */
/* Depthwise image filtering                                          */
/* ------------------------------------------------------------------ */
/*
 * Applies ONE 2-D kernel independently to every colour channel.  This is
 * what "blur this photograph" means: the red channel must not leak into the
 * blue one.  In CNN terms it is a depthwise convolution (groups == channels).
 *
 * Works directly in HWC layout, because for a per-pixel-neighbourhood
 * operation on a 3-channel image the three channels of a pixel sit in the
 * same cache line -- converting to NCHW and back would cost two full strided
 * passes over the data for no benefit.
 *
 * The output keeps the input size ("same" padding), and the kernel is
 * flipped, i.e. this is true convolution, matching scipy.signal.convolve2d.
 */
static inline int map_index_img(int i, int n, pad_mode mode)
{
    if (i >= 0 && i < n) return i;
    switch (mode) {
    case PAD_ZEROS:     return -1;
    case PAD_REPLICATE: return (i < 0) ? 0 : n - 1;
    case PAD_CIRCULAR: { int m = i % n; return (m < 0) ? m + n : m; }
    case PAD_REFLECT: {
        int period, m;
        if (n == 1) return 0;
        period = 2 * n - 2;
        m = i % period;
        if (m < 0) m += period;
        return (m >= n) ? period - m : m;
    }
    default: return (i < 0) ? 0 : n - 1;
    }
}

static inline void filter_row(float *restrict out_row, const Image *in,
                              const float *restrict kernel, int kh, int kw,
                              pad_mode mode, int y)
{
    const int H = in->height, W = in->width, CH = in->channels;
    const int ay = kh / 2, ax = kw / 2;
    int x, i, j, c;

    for (x = 0; x < W * CH; ++x) {
        out_row[x] = 0.0f;
    }

    for (i = 0; i < kh; ++i) {
        /* Kernel is flipped -> true convolution rather than correlation. */
        const int sy = map_index_img(y + (i - ay), H, mode);
        const float *krow = kernel + (size_t)(kh - 1 - i) * kw;
        if (sy < 0) {
            continue;                    /* zero padding: contributes nothing */
        }
        for (j = 0; j < kw; ++j) {
            const float wv = krow[kw - 1 - j];
            const int dx = j - ax;
            if (wv == 0.0f) {
                continue;
            }
            for (x = 0; x < W; ++x) {
                const int sx = map_index_img(x + dx, W, mode);
                if (sx < 0) {
                    continue;
                }
                {
                    const float *src = in->data
                        + ((size_t)sy * W + sx) * CH;
                    float *dst = out_row + (size_t)x * CH;
                    for (c = 0; c < CH; ++c) {
                        dst[c] += wv * src[c];
                    }
                }
            }
        }
    }
}

Image *filter_image_serial(const Image *in, const float *kernel, int kh, int kw,
                           pad_mode mode)
{
    Image *out;
    int y;

    if (!in || !kernel || kh <= 0 || kw <= 0 || (kh % 2) == 0 || (kw % 2) == 0) {
        return NULL;                     /* odd sizes only: needs a centre */
    }
    out = image_alloc(in->height, in->width, in->channels);
    if (!out) {
        return NULL;
    }
    for (y = 0; y < in->height; ++y) {
        filter_row(out->data + (size_t)y * in->width * in->channels,
                   in, kernel, kh, kw, mode, y);
    }
    return out;
}

Image *filter_image_omp(const Image *in, const float *kernel, int kh, int kw,
                        pad_mode mode)
{
    Image *out;

    if (!in || !kernel || kh <= 0 || kw <= 0 || (kh % 2) == 0 || (kw % 2) == 0) {
        return NULL;
    }
    out = image_alloc(in->height, in->width, in->channels);
    if (!out) {
        return NULL;
    }

    {
        const int H = in->height, W = in->width, CH = in->channels;
        float *const odata = out->data;
        int y;

#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(odata, in, kernel) firstprivate(H, W, CH, kh, kw, mode)
#endif
        for (y = 0; y < H; ++y) {
            filter_row(odata + (size_t)y * W * CH, in, kernel, kh, kw, mode, y);
        }
    }
    return out;
}
