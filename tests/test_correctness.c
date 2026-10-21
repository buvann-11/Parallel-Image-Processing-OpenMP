/*
 * test_correctness.c -- correctness test suite.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHAT A PARALLEL PROJECT HAS TO PROVE
 * ---------------------------------------------------------------------------
 * A sequential program is correct or it is not.  A parallel program has two
 * further obligations, and both are tested here:
 *
 *   1. DETERMINISM ACROSS THREAD COUNTS.  Running with 1, 2, 4 ... threads
 *      must give bit-identical output.  If it does not, there is a race, or
 *      a reduction whose combination order leaks into the result.  This is
 *      the single most valuable test in the file: a data race usually does
 *      NOT crash, it quietly produces slightly wrong pixels, and no amount
 *      of staring at the code reliably finds it.
 *
 *   2. EQUIVALENCE TO THE SEQUENTIAL VERSION.  The parallel code must
 *      compute what the serial code computes, exactly -- not "close enough".
 *      Every kernel here is written so that each output element is produced
 *      by one thread in one arithmetic order, so equality is exact, not
 *      approximate.  (Global average pooling is the one exception where a
 *      tolerance is used, and the reason is documented at that test.)
 *
 * On top of that, the usual algorithmic checks: known-value cases, and
 * mathematical invariants that must hold for any input.
 *
 * Build and run:  make test
 */

#include "imgproc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Tiny test harness                                                  */
/* ------------------------------------------------------------------ */
static int g_checks = 0;
static int g_failures = 0;
static const char *g_current = "";

static void begin(const char *name)
{
    g_current = name;
    printf("  %-52s", name);
    fflush(stdout);
}

static void end(void)
{
    printf("ok\n");
}

#define CHECK(cond, ...)                                                \
    do {                                                                \
        ++g_checks;                                                     \
        if (!(cond)) {                                                  \
            ++g_failures;                                               \
            printf("FAILED\n    [%s:%d] in '%s': ",                     \
                   __FILE__, __LINE__, g_current);                      \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
            return;                                                     \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, tol, ...)                                      \
    do {                                                                \
        double va = (double)(a), vb = (double)(b);                      \
        ++g_checks;                                                     \
        if (!(fabs(va - vb) <= (tol))) {                                \
            ++g_failures;                                               \
            printf("FAILED\n    [%s:%d] in '%s': %g != %g (tol %g): ",   \
                   __FILE__, __LINE__, g_current, va, vb, (double)(tol)); \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
            return;                                                     \
        }                                                               \
    } while (0)

/* ------------------------------------------------------------------ */
/* Comparison helpers                                                 */
/* ------------------------------------------------------------------ */
static int tensors_identical(const Tensor *a, const Tensor *b)
{
    size_t i, n;

    if (!a || !b) return 0;
    if (a->n != b->n || a->c != b->c || a->h != b->h || a->w != b->w) return 0;
    n = tensor_elements(a);
    for (i = 0; i < n; ++i) {
        if (a->data[i] != b->data[i]) {   /* exact: see file header */
            return 0;
        }
    }
    return 1;
}

static int images_identical(const Image *a, const Image *b)
{
    size_t i, n;

    if (!a || !b) return 0;
    if (a->height != b->height || a->width != b->width ||
        a->channels != b->channels) return 0;
    n = image_elements(a);
    for (i = 0; i < n; ++i) {
        if (a->data[i] != b->data[i]) {
            /* Say WHERE and by how much: "differs" alone is undebuggable
             * when the failure only shows up on someone else's machine. */
            size_t count = 0, j;
            for (j = i; j < n; ++j) {
                count += (a->data[j] != b->data[j]);
            }
            fprintf(stderr, "\n    first difference at element %lu "
                    "(y=%lu x=%lu c=%lu): %.9g vs %.9g; %lu of %lu differ\n",
                    (unsigned long)i,
                    (unsigned long)(i / a->channels / a->width),
                    (unsigned long)(i / a->channels % a->width),
                    (unsigned long)(i % a->channels),
                    a->data[i], b->data[i],
                    (unsigned long)count, (unsigned long)n);
            return 0;
        }
    }
    return 1;
}

static double image_max_abs_diff(const Image *a, const Image *b)
{
    size_t i, n = image_elements(a);
    double worst = 0.0;

    for (i = 0; i < n; ++i) {
        double d = fabs((double)a->data[i] - (double)b->data[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/* ------------------------------------------------------------------ */
/* PART 1 -- padding                                                  */
/* ------------------------------------------------------------------ */
static void test_padding_shapes(void)
{
    Tensor *in, *out;

    begin("padding: output shape");
    in = tensor_alloc(2, 3, 7, 9);
    CHECK(in != NULL, "allocation failed");
    out = pad2d_serial(in, 2, 4, PAD_ZEROS);
    CHECK(out != NULL, "pad returned NULL");
    CHECK(out->n == 2 && out->c == 3 && out->h == 11 && out->w == 17,
          "got %dx%dx%dx%d", out->n, out->c, out->h, out->w);
    tensor_free(in);
    tensor_free(out);
    end();
}

static void test_padding_modes(void)
{
    /* A 1x1x1x4 row holding 0 1 2 3, padded by 2 on each side. */
    static const float expect_zeros[]     = { 0, 0, 0, 1, 2, 3, 0, 0 };
    static const float expect_reflect[]   = { 2, 1, 0, 1, 2, 3, 2, 1 };
    static const float expect_replicate[] = { 0, 0, 0, 1, 2, 3, 3, 3 };
    static const float expect_circular[]  = { 2, 3, 0, 1, 2, 3, 0, 1 };
    const float *expected[4] = { expect_zeros, expect_reflect,
                                 expect_replicate, expect_circular };
    const char *names[4] = { "zeros", "reflect", "replicate", "circular" };
    int m, i;

    begin("padding: all four border modes, known values");
    for (m = 0; m < 4; ++m) {
        Tensor *in = tensor_alloc(1, 1, 1, 4);
        Tensor *out;
        CHECK(in != NULL, "allocation failed");
        for (i = 0; i < 4; ++i) {
            in->data[i] = (float)i;
        }
        out = pad2d_serial(in, 0, 2, (pad_mode)m);
        CHECK(out != NULL, "pad returned NULL for %s", names[m]);
        CHECK(out->w == 8, "width %d for %s", out->w, names[m]);
        for (i = 0; i < 8; ++i) {
            CHECK(out->data[i] == expected[m][i],
                  "%s index %d: got %g want %g",
                  names[m], i, out->data[i], expected[m][i]);
        }
        tensor_free(in);
        tensor_free(out);
    }
    end();
}

static void test_padding_wider_than_axis(void)
{
    Tensor *in, *out;
    int i;

    /* numpy.pad refuses reflection wider than the axis; modular folding
     * handles it.  Every output value must still come from the input. */
    begin("padding: reflect wider than the axis itself");
    in = tensor_alloc(1, 1, 1, 3);
    CHECK(in != NULL, "allocation failed");
    in->data[0] = 10.0f; in->data[1] = 20.0f; in->data[2] = 30.0f;
    out = pad2d_serial(in, 0, 7, PAD_REFLECT);
    CHECK(out != NULL, "pad returned NULL");
    CHECK(out->w == 17, "width %d", out->w);
    for (i = 0; i < out->w; ++i) {
        const float v = out->data[i];
        CHECK(v == 10.0f || v == 20.0f || v == 30.0f,
              "index %d holds %g, not an input value", i, v);
    }
    tensor_free(in);
    tensor_free(out);
    end();
}

/* ------------------------------------------------------------------ */
/* PART 1 -- convolution                                              */
/* ------------------------------------------------------------------ */
static void test_conv_identity(void)
{
    Tensor *in, *out;
    float weights[9];
    int i;

    begin("convolution: identity kernel reproduces the input");
    in = tensor_alloc(1, 1, 6, 6);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 36; ++i) {
        in->data[i] = (float)(i * 3 % 17);
    }
    memset(weights, 0, sizeof(weights));
    weights[4] = 1.0f;                    /* centre tap only */

    out = conv2d_serial(in, weights, NULL, 1, 3, 3, 1, 1, 1, 1, PAD_ZEROS);
    CHECK(out != NULL, "conv returned NULL");
    CHECK(out->h == 6 && out->w == 6, "shape %dx%d", out->h, out->w);
    for (i = 0; i < 36; ++i) {
        CHECK(out->data[i] == in->data[i], "index %d: %g vs %g",
              i, out->data[i], in->data[i]);
    }
    tensor_free(in);
    tensor_free(out);
    end();
}

static void test_conv_known_values(void)
{
    Tensor *in, *out;
    float weights[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    int i;

    /* 3x3 input 0..8, 2x2 kernel of ones, no padding -> sums of 2x2 blocks. */
    begin("convolution: known 2x2 sum kernel");
    in = tensor_alloc(1, 1, 3, 3);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 9; ++i) {
        in->data[i] = (float)i;
    }
    out = conv2d_serial(in, weights, NULL, 1, 2, 2, 1, 0, 0, 1, PAD_ZEROS);
    CHECK(out != NULL, "conv returned NULL");
    CHECK(out->h == 2 && out->w == 2, "shape %dx%d", out->h, out->w);
    CHECK(out->data[0] == 8.0f,  "[0] = %g, want 8",  out->data[0]);
    CHECK(out->data[1] == 12.0f, "[1] = %g, want 12", out->data[1]);
    CHECK(out->data[2] == 20.0f, "[2] = %g, want 20", out->data[2]);
    CHECK(out->data[3] == 24.0f, "[3] = %g, want 24", out->data[3]);
    tensor_free(in);
    tensor_free(out);
    end();
}

static void test_conv_bias_and_stride(void)
{
    Tensor *in, *dense, *strided;
    float weights[9];
    float bias[1] = { 5.0f };
    int i, y, x;

    begin("convolution: bias offset and stride subsampling");
    in = tensor_alloc(1, 1, 9, 9);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 81; ++i) {
        in->data[i] = (float)(i % 7) - 3.0f;
    }
    for (i = 0; i < 9; ++i) {
        weights[i] = (float)(i + 1) * 0.1f;
    }

    /* Stride 2 must equal the stride-1 result sampled every other pixel. */
    dense = conv2d_serial(in, weights, bias, 1, 3, 3, 1, 0, 0, 1, PAD_ZEROS);
    strided = conv2d_serial(in, weights, bias, 1, 3, 3, 2, 0, 0, 1, PAD_ZEROS);
    CHECK(dense && strided, "conv returned NULL");
    CHECK(dense->h == 7 && strided->h == 4, "shapes %d / %d",
          dense->h, strided->h);
    for (y = 0; y < strided->h; ++y) {
        for (x = 0; x < strided->w; ++x) {
            const float s = strided->data[y * strided->w + x];
            const float d = dense->data[(2 * y) * dense->w + (2 * x)];
            CHECK(s == d, "(%d,%d): strided %g vs dense %g", y, x, s, d);
        }
    }
    tensor_free(in);
    tensor_free(dense);
    tensor_free(strided);
    end();
}

static void test_conv_dilation(void)
{
    Tensor *in, *dilated, *stuffed_out;
    float w3[9], w5[25];
    int i;

    /* Dilation d is equivalent to a kernel with d-1 zeros stuffed between
     * taps -- a definition-level check of the dilation arithmetic. */
    begin("convolution: dilation equals a zero-stuffed kernel");
    in = tensor_alloc(1, 1, 11, 11);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 121; ++i) {
        in->data[i] = (float)((i * 13) % 23) - 11.0f;
    }
    for (i = 0; i < 9; ++i) {
        w3[i] = (float)(i + 1);
    }
    memset(w5, 0, sizeof(w5));
    for (i = 0; i < 9; ++i) {
        w5[(i / 3) * 2 * 5 + (i % 3) * 2] = w3[i];
    }

    dilated = conv2d_serial(in, w3, NULL, 1, 3, 3, 1, 0, 0, 2, PAD_ZEROS);
    stuffed_out = conv2d_serial(in, w5, NULL, 1, 5, 5, 1, 0, 0, 1, PAD_ZEROS);
    CHECK(dilated && stuffed_out, "conv returned NULL");
    CHECK(dilated->h == stuffed_out->h && dilated->w == stuffed_out->w,
          "shapes differ: %dx%d vs %dx%d", dilated->h, dilated->w,
          stuffed_out->h, stuffed_out->w);
    for (i = 0; i < (int)tensor_elements(dilated); ++i) {
        CHECK(dilated->data[i] == stuffed_out->data[i],
              "index %d: %g vs %g", i, dilated->data[i], stuffed_out->data[i]);
    }
    tensor_free(in);
    tensor_free(dilated);
    tensor_free(stuffed_out);
    end();
}

/* ------------------------------------------------------------------ */
/* PART 1 -- pooling                                                  */
/* ------------------------------------------------------------------ */
static void test_pool_known_values(void)
{
    Tensor *in, *mx, *av;
    int i;

    begin("pooling: known 2x2 max and average");
    in = tensor_alloc(1, 1, 4, 4);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 16; ++i) {
        in->data[i] = (float)i;
    }
    mx = pool2d_serial(in, POOL_MAX, 2, 2, 2, 0, 0);
    av = pool2d_serial(in, POOL_AVG, 2, 2, 2, 0, 0);
    CHECK(mx && av, "pool returned NULL");
    CHECK(mx->h == 2 && mx->w == 2, "shape %dx%d", mx->h, mx->w);

    CHECK(mx->data[0] == 5.0f,  "max[0] = %g, want 5",  mx->data[0]);
    CHECK(mx->data[1] == 7.0f,  "max[1] = %g, want 7",  mx->data[1]);
    CHECK(mx->data[2] == 13.0f, "max[2] = %g, want 13", mx->data[2]);
    CHECK(mx->data[3] == 15.0f, "max[3] = %g, want 15", mx->data[3]);

    CHECK(av->data[0] == 2.5f,  "avg[0] = %g, want 2.5",  av->data[0]);
    CHECK(av->data[1] == 4.5f,  "avg[1] = %g, want 4.5",  av->data[1]);
    CHECK(av->data[2] == 10.5f, "avg[2] = %g, want 10.5", av->data[2]);
    CHECK(av->data[3] == 12.5f, "avg[3] = %g, want 12.5", av->data[3]);

    tensor_free(in);
    tensor_free(mx);
    tensor_free(av);
    end();
}

static void test_pool_negative_activations(void)
{
    Tensor *in, *out;
    int i;

    /* Padding a max-pool with 0 instead of -inf would return 0 here.
     * Clamping the window makes that impossible. */
    begin("pooling: padded max-pool never invents a zero");
    in = tensor_alloc(1, 1, 4, 4);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 16; ++i) {
        in->data[i] = -3.0f;
    }
    out = pool2d_serial(in, POOL_MAX, 2, 2, 2, 1, 1);
    CHECK(out != NULL, "pool returned NULL");
    for (i = 0; i < (int)tensor_elements(out); ++i) {
        CHECK(out->data[i] == -3.0f, "index %d = %g, want -3 (padding leaked)",
              i, out->data[i]);
    }
    tensor_free(in);
    tensor_free(out);
    end();
}

static void test_pool_bounds_and_mean(void)
{
    Tensor *in, *mx, *av, *gap;
    double mean = 0.0;
    int i;

    begin("pooling: max >= avg, and exact tiling preserves the mean");
    in = tensor_alloc(1, 2, 8, 8);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < (int)tensor_elements(in); ++i) {
        in->data[i] = (float)((i * 37) % 101) - 50.0f;
        mean += in->data[i];
    }
    mean /= (double)tensor_elements(in);

    mx = pool2d_serial(in, POOL_MAX, 2, 2, 2, 0, 0);
    av = pool2d_serial(in, POOL_AVG, 2, 2, 2, 0, 0);
    CHECK(mx && av, "pool returned NULL");
    for (i = 0; i < (int)tensor_elements(mx); ++i) {
        CHECK(mx->data[i] >= av->data[i] - 1e-6f,
              "index %d: max %g < avg %g", i, mx->data[i], av->data[i]);
    }
    {
        double pooled_mean = 0.0;
        for (i = 0; i < (int)tensor_elements(av); ++i) {
            pooled_mean += av->data[i];
        }
        pooled_mean /= (double)tensor_elements(av);
        CHECK_NEAR(pooled_mean, mean, 1e-4, "mean not preserved");
    }

    gap = global_avg_pool_serial(in);
    CHECK(gap != NULL, "global pool returned NULL");
    CHECK(gap->h == 1 && gap->w == 1 && gap->c == 2, "gap shape wrong");

    tensor_free(in);
    tensor_free(mx);
    tensor_free(av);
    tensor_free(gap);
    end();
}

/* ------------------------------------------------------------------ */
/* PART 2 -- greyscale                                                */
/* ------------------------------------------------------------------ */
static void test_grayscale_known_values(void)
{
    Image *rgb, *gray;

    begin("greyscale: BT.601 and BT.709 on primary colours");
    rgb = image_alloc(1, 3, 3);
    CHECK(rgb != NULL, "allocation failed");
    rgb->data[0] = 255; rgb->data[1] = 0;   rgb->data[2] = 0;     /* red   */
    rgb->data[3] = 0;   rgb->data[4] = 255; rgb->data[5] = 0;     /* green */
    rgb->data[6] = 0;   rgb->data[7] = 0;   rgb->data[8] = 255;   /* blue  */

    gray = grayscale_serial(rgb, GRAY_BT601);
    CHECK(gray != NULL, "greyscale returned NULL");
    CHECK(gray->channels == 1, "channels %d", gray->channels);
    CHECK_NEAR(gray->data[0], 255.0 * 0.299, 0.01, "red under BT.601");
    CHECK_NEAR(gray->data[1], 255.0 * 0.587, 0.01, "green under BT.601");
    CHECK_NEAR(gray->data[2], 255.0 * 0.114, 0.01, "blue under BT.601");
    image_free(gray);

    gray = grayscale_serial(rgb, GRAY_BT709);
    CHECK(gray != NULL, "greyscale returned NULL");
    CHECK_NEAR(gray->data[0], 255.0 * 0.2126, 0.01, "red under BT.709");
    image_free(gray);

    image_free(rgb);
    end();
}

static void test_grayscale_neutral_fixed_point(void)
{
    Image *rgb, *gray;
    int m;

    /* Neutral grey must survive every formula unchanged: the weights sum
     * to 1, and the gamma path must round-trip. */
    begin("greyscale: neutral grey is a fixed point of every method");
    rgb = image_alloc(1, 1, 3);
    CHECK(rgb != NULL, "allocation failed");
    rgb->data[0] = rgb->data[1] = rgb->data[2] = 128.0f;
    for (m = 0; m < GRAY_METHOD_COUNT; ++m) {
        gray = grayscale_serial(rgb, (gray_method)m);
        CHECK(gray != NULL, "greyscale returned NULL for method %d", m);
        CHECK_NEAR(gray->data[0], 128.0, 0.6, "method %s",
                   gray_method_name((gray_method)m));
        image_free(gray);
    }
    image_free(rgb);
    end();
}

/* ------------------------------------------------------------------ */
/* PART 2 -- normalisation                                            */
/* ------------------------------------------------------------------ */
static void test_normalize_minmax(void)
{
    Image *in, *out;
    int i;

    begin("normalisation: min-max maps the observed range onto [0,255]");
    in = image_alloc(4, 4, 1);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 16; ++i) {
        in->data[i] = 100.0f + (float)i;   /* range [100, 115] */
    }
    out = normalize_serial(in, NORM_MINMAX);
    CHECK(out != NULL, "normalize returned NULL");
    CHECK_NEAR(out->data[0], 0.0, 1e-3, "minimum should map to 0");
    CHECK_NEAR(out->data[15], 255.0, 1e-3, "maximum should map to 255");
    for (i = 1; i < 16; ++i) {
        CHECK(out->data[i] >= out->data[i - 1],
              "monotonicity broken at %d", i);
    }
    image_free(in);
    image_free(out);
    end();
}

static void test_normalize_constant_image(void)
{
    Image *in, *out;
    int m, i;

    /* Zero range and zero variance: must produce a defined value, not NaN. */
    begin("normalisation: constant image never produces NaN");
    in = image_alloc(8, 8, 1);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 64; ++i) {
        in->data[i] = 42.0f;
    }
    for (m = 0; m < NORM_METHOD_COUNT; ++m) {
        out = normalize_serial(in, (norm_method)m);
        CHECK(out != NULL, "normalize returned NULL for %d", m);
        for (i = 0; i < 64; ++i) {
            CHECK(out->data[i] == out->data[i],   /* NaN != NaN */
                  "%s produced NaN at %d", norm_method_name((norm_method)m), i);
            CHECK(fabsf(out->data[i]) < 1e6f,
                  "%s produced %g at %d",
                  norm_method_name((norm_method)m), out->data[i], i);
        }
        image_free(out);
    }
    image_free(in);
    end();
}

static void test_normalize_equalize_monotone(void)
{
    Image *in, *out;
    int i;

    /* Equalisation is a CDF mapping, so it must preserve pixel ORDER. */
    begin("normalisation: histogram equalisation is monotone");
    in = image_alloc(16, 16, 1);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 256; ++i) {
        in->data[i] = (float)(i % 200);
    }
    out = normalize_serial(in, NORM_EQUALIZE);
    CHECK(out != NULL, "normalize returned NULL");
    for (i = 0; i < 256; ++i) {
        int j;
        for (j = 0; j < 256; ++j) {
            if (in->data[i] < in->data[j]) {
                CHECK(out->data[i] <= out->data[j] + 1e-4f,
                      "order inverted: in %g->%g vs %g->%g",
                      in->data[i], out->data[i], in->data[j], out->data[j]);
            }
        }
    }
    image_free(in);
    image_free(out);
    end();
}

/* ------------------------------------------------------------------ */
/* PART 2 -- resize                                                   */
/* ------------------------------------------------------------------ */
static void test_resize_shapes(void)
{
    Image *in, *out;
    int m;

    begin("resize: output shape for every method");
    in = image_make_test_pattern(32, 3);
    CHECK(in != NULL, "pattern generation failed");
    for (m = 0; m < RESIZE_METHOD_COUNT; ++m) {
        out = resize_serial(in, 17, 23, (resize_method)m, 1);
        CHECK(out != NULL, "resize returned NULL for method %d", m);
        CHECK(out->height == 17 && out->width == 23 && out->channels == 3,
              "shape %dx%dx%d", out->height, out->width, out->channels);
        image_free(out);
    }
    image_free(in);
    end();
}

static void test_resize_partition_of_unity(void)
{
    Image *in, *out;
    int m, i;

    /*
     * The defining property of a correct resampler: the weights of every
     * output pixel sum to exactly 1, so a constant image resizes to the same
     * constant.  Without it, resizing silently changes brightness.
     */
    begin("resize: constant image preserved by every kernel");
    in = image_alloc(24, 24, 1);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 24 * 24; ++i) {
        in->data[i] = 77.0f;
    }
    for (m = 0; m < RESIZE_METHOD_COUNT; ++m) {
        int sizes[3][2] = { { 12, 12 }, { 48, 48 }, { 7, 19 } };
        int s;
        for (s = 0; s < 3; ++s) {
            out = resize_serial(in, sizes[s][0], sizes[s][1],
                                (resize_method)m, 1);
            CHECK(out != NULL, "resize returned NULL");
            for (i = 0; i < (int)image_elements(out); ++i) {
                CHECK_NEAR(out->data[i], 77.0, 1e-3,
                           "%s to %dx%d drifted at %d",
                           resize_method_name((resize_method)m),
                           sizes[s][0], sizes[s][1], i);
            }
            image_free(out);
        }
    }
    image_free(in);
    end();
}

static void test_resize_nearest_exact_upscale(void)
{
    Image *in, *out;
    static const float expect[16] = { 1, 1, 2, 2,  1, 1, 2, 2,
                                      3, 3, 4, 4,  3, 3, 4, 4 };
    int i;

    begin("resize: 2x nearest upscale repeats pixels exactly");
    in = image_alloc(2, 2, 1);
    CHECK(in != NULL, "allocation failed");
    in->data[0] = 1; in->data[1] = 2; in->data[2] = 3; in->data[3] = 4;
    out = resize_serial(in, 4, 4, RESIZE_NEAREST, 0);
    CHECK(out != NULL, "resize returned NULL");
    for (i = 0; i < 16; ++i) {
        CHECK(out->data[i] == expect[i], "index %d: %g want %g",
              i, out->data[i], expect[i]);
    }
    image_free(in);
    image_free(out);
    end();
}

static void test_resize_antialiasing(void)
{
    Image *in, *aa, *no_aa;
    double err_aa = 0.0, err_no = 0.0;
    unsigned int seed = 987654321u;
    int y, x, i;

    /*
     * Ground truth for an 8x reduction is the mean of each 8x8 block.  A
     * one-pixel-wide kernel reads only a couple of samples per output and
     * lands far from that mean; widening it by the downscale factor fixes it.
     *
     * The source is deterministic white noise rather than the structured
     * test pattern.  That matters: the pattern's checkerboard has 8-pixel
     * squares that align exactly with the 8x8 ground-truth blocks, so the
     * block mean would preserve full checker contrast while a correct
     * anti-aliased filter -- spanning two squares -- averages it away.  On
     * that input the "ground truth" punishes the right answer.  White noise
     * has no such alignment, so the block mean is genuinely what a correct
     * downscale should approach.
     */
    begin("resize: anti-aliasing reduces downscaling error");
    in = image_alloc(128, 128, 1);
    CHECK(in != NULL, "allocation failed");
    for (i = 0; i < 128 * 128; ++i) {
        seed = seed * 1103515245u + 12345u;
        in->data[i] = (float)((seed >> 16) & 0xff);
    }

    aa = resize_serial(in, 16, 16, RESIZE_BILINEAR, 1);
    no_aa = resize_serial(in, 16, 16, RESIZE_BILINEAR, 0);
    CHECK(aa && no_aa, "resize returned NULL");

    for (y = 0; y < 16; ++y) {
        for (x = 0; x < 16; ++x) {
            double truth = 0.0;
            int by, bx;
            for (by = 0; by < 8; ++by) {
                for (bx = 0; bx < 8; ++bx) {
                    truth += in->data[(y * 8 + by) * 128 + (x * 8 + bx)];
                }
            }
            truth /= 64.0;
            err_aa += fabs(aa->data[y * 16 + x] - truth);
            err_no += fabs(no_aa->data[y * 16 + x] - truth);
        }
    }
    CHECK(err_aa < 0.5 * err_no,
          "anti-aliased error %g not well below naive error %g",
          err_aa, err_no);
    image_free(in);
    image_free(aa);
    image_free(no_aa);
    end();
}

static void test_resize_antialiasing_tames_fine_detail(void)
{
    Image *in, *aa, *no_aa;
    double var_aa = 0.0, var_no = 0.0, mean_aa = 0.0, mean_no = 0.0;
    int y, x, i, n;

    /*
     * The complementary demonstration: a checkerboard finer than the output
     * grid carries no information the output can represent, so a correct
     * downscale must average it towards mid-grey.  Without anti-aliasing the
     * sampler locks onto one phase of the pattern and produces high-contrast
     * moire -- visible here as a much larger output variance.
     */
    begin("resize: anti-aliasing suppresses moire on fine detail");
    in = image_alloc(120, 120, 1);
    CHECK(in != NULL, "allocation failed");
    for (y = 0; y < 120; ++y) {
        for (x = 0; x < 120; ++x) {
            /* 3-pixel squares: far finer than the 11x11 output grid. */
            in->data[y * 120 + x] = (((y / 3) + (x / 3)) % 2) ? 255.0f : 0.0f;
        }
    }
    aa = resize_serial(in, 11, 11, RESIZE_BILINEAR, 1);
    no_aa = resize_serial(in, 11, 11, RESIZE_BILINEAR, 0);
    CHECK(aa && no_aa, "resize returned NULL");

    n = 11 * 11;
    for (i = 0; i < n; ++i) {
        mean_aa += aa->data[i];
        mean_no += no_aa->data[i];
    }
    mean_aa /= n;
    mean_no /= n;
    for (i = 0; i < n; ++i) {
        var_aa += (aa->data[i] - mean_aa) * (aa->data[i] - mean_aa);
        var_no += (no_aa->data[i] - mean_no) * (no_aa->data[i] - mean_no);
    }
    var_aa /= n;
    var_no /= n;

    CHECK(var_aa < var_no, "anti-aliased variance %g not below naive %g",
          var_aa, var_no);
    CHECK(fabs(mean_aa - 127.5) < 25.0,
          "anti-aliased mean %g far from mid-grey", mean_aa);

    image_free(in);
    image_free(aa);
    image_free(no_aa);
    end();
}

/* ------------------------------------------------------------------ */
/* PARALLEL CORRECTNESS -- the core of this suite                     */
/* ------------------------------------------------------------------ */
static void test_serial_matches_parallel(void)
{
    Image *img = image_make_test_pattern(96, 3);
    Tensor *ten;
    float kernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    int i, m;

    begin("parallel: OpenMP output is identical to serial output");
    CHECK(img != NULL, "pattern generation failed");
    ten = image_to_tensor(img);
    CHECK(ten != NULL, "conversion failed");
    for (i = 0; i < 9; ++i) {
        kernel[i] /= 16.0f;
    }

    for (m = 0; m < GRAY_METHOD_COUNT; ++m) {
        Image *a = grayscale_serial(img, (gray_method)m);
        Image *b = grayscale_omp(img, (gray_method)m);
        CHECK(images_identical(a, b), "greyscale %s differs",
              gray_method_name((gray_method)m));
        image_free(a); image_free(b);
    }
    for (m = 0; m < NORM_METHOD_COUNT; ++m) {
        Image *a = normalize_serial(img, (norm_method)m);
        Image *b = normalize_omp(img, (norm_method)m);
        CHECK(images_identical(a, b), "normalise %s differs",
              norm_method_name((norm_method)m));
        image_free(a); image_free(b);
    }
    for (m = 0; m < RESIZE_METHOD_COUNT; ++m) {
        Image *a = resize_serial(img, 40, 57, (resize_method)m, 1);
        Image *b = resize_omp(img, 40, 57, (resize_method)m, 1);
        CHECK(images_identical(a, b), "resize %s differs",
              resize_method_name((resize_method)m));
        image_free(a); image_free(b);
    }
    for (m = 0; m < PAD_MODE_COUNT; ++m) {
        Tensor *a = pad2d_serial(ten, 9, 13, (pad_mode)m);
        Tensor *b = pad2d_omp(ten, 9, 13, (pad_mode)m);
        CHECK(tensors_identical(a, b), "pad %s differs",
              pad_mode_name((pad_mode)m));
        tensor_free(a); tensor_free(b);
    }
    for (m = 0; m < POOL_TYPE_COUNT; ++m) {
        Tensor *a = pool2d_serial(ten, (pool_type)m, 3, 3, 2, 1, 1);
        Tensor *b = pool2d_omp(ten, (pool_type)m, 3, 3, 2, 1, 1);
        CHECK(tensors_identical(a, b), "pool %s differs",
              pool_type_name((pool_type)m));
        tensor_free(a); tensor_free(b);
    }
    {
        Image *a = filter_image_serial(img, kernel, 3, 3, PAD_REFLECT);
        Image *b = filter_image_omp(img, kernel, 3, 3, PAD_REFLECT);
        CHECK(images_identical(a, b), "image filter differs");
        image_free(a); image_free(b);
    }
    {
        /* Multi-channel CNN convolution with 8 filters. */
        float *w = (float *)malloc(sizeof(float) * 8 * 3 * 3 * 3);
        Tensor *a, *b;
        CHECK(w != NULL, "allocation failed");
        for (i = 0; i < 8 * 3 * 3 * 3; ++i) {
            w[i] = (float)((i * 29) % 13) - 6.0f;
        }
        a = conv2d_serial(ten, w, NULL, 8, 3, 3, 1, 1, 1, 1, PAD_REFLECT);
        b = conv2d_omp(ten, w, NULL, 8, 3, 3, 1, 1, 1, 1, PAD_REFLECT);
        CHECK(tensors_identical(a, b), "CNN convolution differs");
        tensor_free(a); tensor_free(b);
        free(w);
    }

    image_free(img);
    tensor_free(ten);
    end();
}

static void test_thread_count_invariance(void)
{
    Image *img = image_make_test_pattern(128, 3);
    Tensor *ten;
    Image *reference_gray = NULL, *reference_resize = NULL;
    Tensor *reference_pool = NULL;
    int threads[4] = { 1, 2, 3, 4 };
    int t;

    /*
     * THE most important test here.  A data race almost never crashes; it
     * quietly changes a handful of pixels, and the difference usually only
     * appears at some thread counts.  Sweeping the thread count and
     * demanding bit-identical output is the practical way to catch it.
     */
    begin("parallel: output is bit-identical at 1/2/3/4 threads");
    CHECK(img != NULL, "pattern generation failed");
    ten = image_to_tensor(img);
    CHECK(ten != NULL, "conversion failed");

    for (t = 0; t < 4; ++t) {
        Image *gray, *resized;
        Tensor *pooled;

        ip_set_threads(threads[t]);

        gray = grayscale_omp(img, GRAY_BT709);
        resized = resize_omp(img, 53, 71, RESIZE_BICUBIC, 1);
        pooled = pool2d_omp(ten, POOL_MAX, 3, 3, 2, 1, 1);
        CHECK(gray && resized && pooled, "operation failed at %d threads",
              threads[t]);

        if (t == 0) {
            reference_gray = gray;
            reference_resize = resized;
            reference_pool = pooled;
        } else {
            CHECK(images_identical(gray, reference_gray),
                  "greyscale differs at %d threads", threads[t]);
            CHECK(images_identical(resized, reference_resize),
                  "resize differs at %d threads", threads[t]);
            CHECK(tensors_identical(pooled, reference_pool),
                  "pooling differs at %d threads", threads[t]);
            image_free(gray);
            image_free(resized);
            tensor_free(pooled);
        }
    }

    ip_set_threads(ip_max_threads());
    image_free(reference_gray);
    image_free(reference_resize);
    tensor_free(reference_pool);
    image_free(img);
    tensor_free(ten);
    end();
}

static void test_reduction_stability(void)
{
    Tensor *ten;
    Image *img;
    float first = 0.0f;
    int t;

    /*
     * Global average pooling sums a whole plane, so with a different thread
     * count the additions combine in a different ORDER.  Floating-point
     * addition is not associative, so a tiny difference is legitimate here --
     * unlike every other kernel, where each output is produced by exactly one
     * thread in one fixed order.  A double accumulator keeps the drift far
     * below what any downstream use could notice; this test pins that down
     * rather than pretending the result is bit-exact.
     */
    begin("parallel: plane-wide reduction is stable across thread counts");
    img = image_make_test_pattern(200, 3);
    CHECK(img != NULL, "pattern generation failed");
    ten = image_to_tensor(img);
    CHECK(ten != NULL, "conversion failed");

    for (t = 1; t <= 4; ++t) {
        Tensor *gap;
        ip_set_threads(t);
        gap = global_avg_pool_omp(ten);
        CHECK(gap != NULL, "global pool failed at %d threads", t);
        if (t == 1) {
            first = gap->data[0];
        } else {
            CHECK_NEAR(gap->data[0], first, 1e-3,
                       "drifted at %d threads", t);
        }
        tensor_free(gap);
    }

    ip_set_threads(ip_max_threads());
    image_free(img);
    tensor_free(ten);
    end();
}

static void test_schedule_invariance(void)
{
    Image *img = image_make_test_pattern(96, 3);
    Image *reference = NULL;
    int s;

    /* The loop schedule decides WHICH thread runs which iteration.  It must
     * never decide the RESULT. */
    begin("parallel: output is identical under static/dynamic/guided");
    CHECK(img != NULL, "pattern generation failed");
    for (s = 0; s < SCHED_COUNT; ++s) {
        Image *out;
        imgproc_set_schedule((omp_schedule_kind)s, 0);
        out = resize_omp(img, 37, 61, RESIZE_BICUBIC, 1);
        CHECK(out != NULL, "resize failed under %s",
              schedule_name((omp_schedule_kind)s));
        if (s == 0) {
            reference = out;
        } else {
            CHECK(images_identical(out, reference),
                  "result differs under %s schedule",
                  schedule_name((omp_schedule_kind)s));
            image_free(out);
        }
    }
    imgproc_set_schedule(SCHED_STATIC, 0);
    image_free(reference);
    image_free(img);
    end();
}

/* ------------------------------------------------------------------ */
/* I/O                                                                */
/* ------------------------------------------------------------------ */
static void test_netpbm_round_trip(void)
{
    Image *rgb, *gray, *loaded;
    ip_status status;
    const char *ppm = "build/_test_rt.ppm";
    const char *pgm = "build/_test_rt.pgm";

    begin("I/O: PPM and PGM round-trip without loss");
    rgb = image_make_test_pattern(37, 3);
    CHECK(rgb != NULL, "pattern generation failed");
    CHECK(image_write_netpbm(rgb, ppm) == IP_OK, "PPM write failed");
    loaded = image_read_netpbm(ppm, &status);
    CHECK(loaded != NULL, "PPM read failed: %s", ip_status_string(status));
    CHECK(loaded->height == 37 && loaded->width == 37 && loaded->channels == 3,
          "PPM shape %dx%dx%d", loaded->height, loaded->width, loaded->channels);
    CHECK(image_max_abs_diff(rgb, loaded) <= 0.5,
          "PPM round-trip drift %g", image_max_abs_diff(rgb, loaded));
    image_free(loaded);

    gray = grayscale_serial(rgb, GRAY_BT601);
    CHECK(gray != NULL, "greyscale failed");
    CHECK(image_write_netpbm(gray, pgm) == IP_OK, "PGM write failed");
    loaded = image_read_netpbm(pgm, &status);
    CHECK(loaded != NULL, "PGM read failed: %s", ip_status_string(status));
    CHECK(loaded->channels == 1, "PGM channels %d", loaded->channels);
    image_free(loaded);

    image_free(rgb);
    image_free(gray);
    remove(ppm);
    remove(pgm);
    end();
}

static void test_io_rejects_bad_input(void)
{
    ip_status status = IP_OK;
    Image *img;

    begin("I/O: malformed and missing files are rejected cleanly");
    img = image_read_netpbm("build/_does_not_exist.ppm", &status);
    CHECK(img == NULL, "missing file should fail");
    CHECK(status == IP_ERR_IO, "expected IP_ERR_IO, got %s",
          ip_status_string(status));

    {
        FILE *fp = fopen("build/_test_bad.ppm", "wb");
        CHECK(fp != NULL, "could not create scratch file");
        fputs("P9\n1 1\n255\n", fp);
        fclose(fp);
    }
    img = image_read_netpbm("build/_test_bad.ppm", &status);
    CHECK(img == NULL, "bad magic should fail");
    CHECK(status == IP_ERR_FORMAT, "expected IP_ERR_FORMAT, got %s",
          ip_status_string(status));
    remove("build/_test_bad.ppm");
    end();
}

/* ------------------------------------------------------------------ */
/* Argument validation                                                */
/* ------------------------------------------------------------------ */
static void test_invalid_arguments(void)
{
    Tensor *ten = tensor_alloc(1, 1, 8, 8);
    Image *img = image_alloc(8, 8, 3);
    float w[9] = { 0 };

    begin("validation: invalid arguments return NULL, never crash");
    CHECK(ten && img, "allocation failed");

    CHECK(pad2d_serial(NULL, 1, 1, PAD_ZEROS) == NULL, "NULL input accepted");
    CHECK(pad2d_serial(ten, -1, 1, PAD_ZEROS) == NULL, "negative pad accepted");
    CHECK(pad2d_omp(ten, 1, 1, (pad_mode)99) == NULL, "bad mode accepted");

    CHECK(conv2d_serial(ten, w, NULL, 0, 3, 3, 1, 0, 0, 1, PAD_ZEROS) == NULL,
          "zero output channels accepted");
    CHECK(conv2d_serial(ten, NULL, NULL, 1, 3, 3, 1, 0, 0, 1, PAD_ZEROS) == NULL,
          "NULL weights accepted");
    CHECK(conv2d_serial(ten, w, NULL, 1, 3, 3, 0, 0, 0, 1, PAD_ZEROS) == NULL,
          "zero stride accepted");
    CHECK(conv2d_serial(ten, w, NULL, 1, 99, 99, 1, 0, 0, 1, PAD_ZEROS) == NULL,
          "oversized kernel accepted");

    /* PyTorch's rule: pooling padding may not exceed half the kernel. */
    CHECK(pool2d_serial(ten, POOL_MAX, 2, 2, 2, 2, 2) == NULL,
          "excessive pooling padding accepted");

    CHECK(resize_serial(img, 0, 8, RESIZE_BILINEAR, 1) == NULL,
          "zero output height accepted");
    CHECK(resize_serial(NULL, 4, 4, RESIZE_BILINEAR, 1) == NULL,
          "NULL image accepted");

    CHECK(grayscale_serial(NULL, GRAY_BT601) == NULL, "NULL image accepted");
    CHECK(normalize_serial(img, (norm_method)42) == NULL, "bad method accepted");

    /* Even-sized filter kernels have no well-defined centre. */
    CHECK(filter_image_serial(img, w, 2, 2, PAD_ZEROS) == NULL,
          "even kernel accepted");

    CHECK(conv_output_size(8, 3, 1, 0, 0, 1) == 6, "output size arithmetic");
    CHECK(conv_output_size(3, 5, 1, 0, 0, 1) == -1, "kernel larger than input");

    tensor_free(ten);
    image_free(img);
    end();
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("\n============================================================\n");
    printf(" MTCS-103P Mini Project -- correctness test suite\n");
    printf("============================================================\n");
    printf("  OpenMP      : %s\n", ip_openmp_enabled() ? "enabled" : "disabled");
    printf("  max threads : %d\n\n", ip_max_threads());

    printf("Part 1 -- CNN primitives\n");
    test_padding_shapes();
    test_padding_modes();
    test_padding_wider_than_axis();
    test_conv_identity();
    test_conv_known_values();
    test_conv_bias_and_stride();
    test_conv_dilation();
    test_pool_known_values();
    test_pool_negative_activations();
    test_pool_bounds_and_mean();

    printf("\nPart 2 -- Image processing\n");
    test_grayscale_known_values();
    test_grayscale_neutral_fixed_point();
    test_normalize_minmax();
    test_normalize_constant_image();
    test_normalize_equalize_monotone();
    test_resize_shapes();
    test_resize_partition_of_unity();
    test_resize_nearest_exact_upscale();
    test_resize_antialiasing();
    test_resize_antialiasing_tames_fine_detail();

    printf("\nParallel correctness\n");
    test_serial_matches_parallel();
    test_thread_count_invariance();
    test_reduction_stability();
    test_schedule_invariance();

    printf("\nI/O and validation\n");
    test_netpbm_round_trip();
    test_io_rejects_bad_input();
    test_invalid_arguments();

    printf("\n------------------------------------------------------------\n");
    if (g_failures == 0) {
        printf(" ALL TESTS PASSED  (%d assertions)\n", g_checks);
    } else {
        printf(" %d FAILURE(S) out of %d assertions\n", g_failures, g_checks);
    }
    printf("------------------------------------------------------------\n\n");
    return g_failures == 0 ? 0 : 1;
}
