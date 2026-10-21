/*
 * grayscale.c -- PART 2: RGB to greyscale conversion.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHY A WEIGHTED SUM, NOT AN AVERAGE
 * ---------------------------------------------------------------------------
 * Averaging the channels assumes the eye is equally sensitive to red, green
 * and blue.  It is not: the retina has roughly twice as many green-sensitive
 * cones as red-sensitive ones and far fewer blue-sensitive ones.  Broadcast
 * standards encode this as a weighted sum whose weights sum to 1:
 *
 *   Standard        w_R      w_G      w_B     Used by
 *   ------------    ------   ------   ------  --------------------------
 *   ITU-R BT.601    0.2990   0.5870   0.1140  SDTV; OpenCV, Pillow, MATLAB
 *   ITU-R BT.709    0.2126   0.7152   0.0722  HDTV / sRGB; scikit-image
 *
 * Full-intensity red maps to 76 under BT.601 but 54 under BT.709 and 85
 * under a naive average -- a 40% spread.  The choice is not cosmetic.
 *
 * ---------------------------------------------------------------------------
 * LUMA vs LUMINANCE (the gamma subtlety)
 * ---------------------------------------------------------------------------
 * Stored sRGB values are gamma-encoded: 128 is not "half the light" of 255,
 * it is about 21% of it.  Applying the weights to the encoded values gives
 * *luma* Y' -- fast, conventional, and what OpenCV and Pillow produce.
 * Linearising first, weighting, then re-encoding gives true relative
 * *luminance* Y -- photometrically correct and noticeably brighter in
 * saturated regions.  GRAY_LUMINANCE does the latter.
 *
 * ---------------------------------------------------------------------------
 * PARALLELISATION
 * ---------------------------------------------------------------------------
 * This is a pure element-wise map: output pixel i depends only on input
 * pixel i.  It is the most trivially parallel kernel in the project and,
 * for exactly that reason, the most memory-bound: 3 floats in, 1 float out,
 * and 3 multiply-adds of arithmetic.  It is included in the scaling study as
 * the low-arithmetic-intensity end of the spectrum -- expect it to stop
 * scaling once the memory bus saturates, typically well before the core
 * count is reached.  See report section 5.3.
 *
 * GRAY_LUMINANCE is the interesting exception: powf() is expensive enough
 * that the kernel becomes compute-bound and scales nearly linearly.  Having
 * both in one file makes the contrast easy to demonstrate.
 */

#include "imgproc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* sRGB transfer function                                             */
/* ------------------------------------------------------------------ */
/*
 * The linear segment near black avoids the infinite slope a pure power law
 * would have at zero.  IEC 61966-2-1.
 */
static inline float srgb_to_linear(float c)
{
    if (c <= 0.04045f) {
        return c / 12.92f;
    }
    return powf((c + 0.055f) / 1.055f, 2.4f);
}

static inline float linear_to_srgb(float c)
{
    if (c <= 0.0031308f) {
        return c * 12.92f;
    }
    return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

/* ------------------------------------------------------------------ */
/* Per-pixel conversion                                               */
/* ------------------------------------------------------------------ */
static inline float gray_pixel(float r, float g, float b, gray_method method)
{
    switch (method) {
    case GRAY_BT601:
        return 0.299f * r + 0.587f * g + 0.114f * b;

    case GRAY_BT709:
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;

    case GRAY_AVERAGE:
        return (r + g + b) * (1.0f / 3.0f);

    case GRAY_LUMINANCE: {
        /* Work in [0,1], linearise, weight with BT.709, re-encode. */
        const float rl = srgb_to_linear(r * (1.0f / 255.0f));
        const float gl = srgb_to_linear(g * (1.0f / 255.0f));
        const float bl = srgb_to_linear(b * (1.0f / 255.0f));
        const float y = 0.2126f * rl + 0.7152f * gl + 0.0722f * bl;
        return linear_to_srgb(y) * 255.0f;
    }

    default:
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

/* ------------------------------------------------------------------ */
/* Serial baseline                                                    */
/* ------------------------------------------------------------------ */
Image *grayscale_serial(const Image *in, gray_method method)
{
    Image *out;
    long i, pixels;

    if (!in || method < 0 || method >= GRAY_METHOD_COUNT) {
        return NULL;
    }
    if (in->channels == 1) {
        return image_clone(in);          /* already greyscale */
    }
    if (in->channels != 3) {
        return NULL;
    }

    out = image_alloc(in->height, in->width, 1);
    if (!out) {
        return NULL;
    }
    pixels = (long)in->height * in->width;

    for (i = 0; i < pixels; ++i) {
        const float *px = in->data + (size_t)i * 3;
        out->data[i] = gray_pixel(px[0], px[1], px[2], method);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* OpenMP version                                                     */
/* ------------------------------------------------------------------ */
Image *grayscale_omp(const Image *in, gray_method method)
{
    Image *out;

    if (!in || method < 0 || method >= GRAY_METHOD_COUNT) {
        return NULL;
    }
    if (in->channels == 1) {
        return image_clone(in);
    }
    if (in->channels != 3) {
        return NULL;
    }

    out = image_alloc(in->height, in->width, 1);
    if (!out) {
        return NULL;
    }

    {
        const long pixels = (long)in->height * in->width;
        const float *const src = in->data;
        float *const dst = out->data;
        long i;

        /*
         * One flat loop over pixels rather than a (row, column) nest: the
         * data is contiguous in HWC, so a flat loop lets the scheduler hand
         * each thread a contiguous byte range.  That maximises hardware
         * prefetch efficiency and gives each thread its own set of pages,
         * which keeps first-touch NUMA placement sensible on multi-socket
         * machines.
         */
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(src, dst) firstprivate(pixels, method)
#endif
        for (i = 0; i < pixels; ++i) {
            const float *px = src + (size_t)i * 3;
            dst[i] = gray_pixel(px[0], px[1], px[2], method);
        }
    }
    return out;
}
