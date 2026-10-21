/*
 * normalize.c -- PART 2: intensity normalisation.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHY NORMALISE
 * ---------------------------------------------------------------------------
 * Raw pixels live in [0, 255].  Fed to a network, the first layer's
 * pre-activations have a large, input-dependent scale and a large positive
 * mean; gradients are correspondingly large and badly conditioned, saturating
 * activations and forcing a tiny learning rate.  Normalisation puts every
 * input feature on a comparable, roughly zero-centred scale.  It is the first
 * transform in essentially every vision pipeline.
 *
 *   min-max    x' = (x - min) / (max - min)          bounded, outlier-sensitive
 *   z-score    x' = (x - mean) / stddev              unbounded, exposure-robust
 *   equalise   x' = cdf(x)                           flattens the histogram
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS THE INTERESTING FILE FOR HPC
 * ---------------------------------------------------------------------------
 * The other kernels are embarrassingly parallel: every output depends on one
 * local neighbourhood and nothing else.  Normalisation is not.  All three
 * schemes need a GLOBAL property of the image -- its extrema, its moments, or
 * its histogram -- before a single output pixel can be written.  That forces
 * a two-phase structure and demands real parallel-reduction machinery:
 *
 *   Phase 1 (reduce)  : compute the global statistic cooperatively
 *   Phase 2 (map)     : apply the resulting point transform, embarrassingly
 *
 * Three different reduction techniques are demonstrated, one per scheme:
 *
 *   NORM_MINMAX    `reduction(min:)` / `reduction(max:)`  -- OpenMP 3.1+
 *   NORM_ZSCORE    `reduction(+:)` on two accumulators simultaneously
 *   NORM_EQUALIZE  per-thread PRIVATE histograms, merged in a second step
 *
 * The histogram case is the one worth studying.  The naive version --
 * `#pragma omp atomic` on a shared 256-bin array -- is catastrophically slow:
 * every thread hammers the same few cache lines, so the lines ping-pong
 * between cores and the kernel runs SLOWER than serial.  Giving each thread
 * its own padded histogram removes both the atomics and the false sharing;
 * the merge is over nthreads*256 values, which is nothing.  See report
 * section 5.4 for the measured difference.
 */

#include "imgproc.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define HIST_BINS 256

/*
 * Each thread's histogram is padded to a whole number of 64-byte cache lines
 * so that thread t's bins can never share a line with thread t+1's.  Without
 * this padding the merge array would still cause false sharing even though
 * the bins themselves are private.
 */
#define HIST_STRIDE (((HIST_BINS * (int)sizeof(long) + 63) / 64) * 64 / (int)sizeof(long))

/* ------------------------------------------------------------------ */
/* Phase 1 variants: global statistics                                */
/* ------------------------------------------------------------------ */
static void stats_minmax_serial(const float *data, long n,
                                float *out_min, float *out_max)
{
    float lo = FLT_MAX, hi = -FLT_MAX;
    long i;

    for (i = 0; i < n; ++i) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }
    *out_min = lo;
    *out_max = hi;
}

static void stats_minmax_omp(const float *data, long n,
                             float *out_min, float *out_max)
{
    float lo = FLT_MAX, hi = -FLT_MAX;
    long i;

    /* min/max reductions require OpenMP 3.1 (2011); gcc has had them for
     * over a decade.  Each thread keeps a private lo/hi and the runtime
     * combines them in a tree at the end of the region. */
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(data) firstprivate(n) reduction(min:lo) reduction(max:hi)
#endif
    for (i = 0; i < n; ++i) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }
    *out_min = lo;
    *out_max = hi;
}

static void stats_moments_serial(const float *data, long n,
                                 double *out_mean, double *out_std)
{
    double sum = 0.0, sumsq = 0.0;
    long i;

    for (i = 0; i < n; ++i) {
        const double v = (double)data[i];
        sum += v;
        sumsq += v * v;
    }
    {
        const double mean = sum / (double)n;
        double var = sumsq / (double)n - mean * mean;
        if (var < 0.0) var = 0.0;        /* guard against catastrophic
                                          * cancellation on flat images */
        *out_mean = mean;
        *out_std = sqrt(var);
    }
}

static void stats_moments_omp(const float *data, long n,
                              double *out_mean, double *out_std)
{
    double sum = 0.0, sumsq = 0.0;
    long i;

    /* Two reduction clauses on one loop: the data is read once and both
     * moments fall out, so this stays a single memory pass. */
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(data) firstprivate(n) reduction(+:sum) reduction(+:sumsq)
#endif
    for (i = 0; i < n; ++i) {
        const double v = (double)data[i];
        sum += v;
        sumsq += v * v;
    }
    {
        const double mean = sum / (double)n;
        double var = sumsq / (double)n - mean * mean;
        if (var < 0.0) var = 0.0;
        *out_mean = mean;
        *out_std = sqrt(var);
    }
}

static inline int bin_of(float value)
{
    int b = (int)(value + 0.5f);
    if (b < 0) b = 0;
    if (b > HIST_BINS - 1) b = HIST_BINS - 1;
    return b;
}

static void histogram_serial(const float *data, long n, long *hist)
{
    long i;

    memset(hist, 0, HIST_BINS * sizeof(long));
    for (i = 0; i < n; ++i) {
        hist[bin_of(data[i])]++;
    }
}

static void histogram_omp(const float *data, long n, long *hist)
{
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
    long *local = (long *)calloc((size_t)nthreads * HIST_STRIDE, sizeof(long));
    int t, b;
    long i;

    if (!local) {                        /* fall back rather than fail */
        histogram_serial(data, n, hist);
        return;
    }

#pragma omp parallel default(none) shared(data, local) firstprivate(n)
    {
        long *mine = local + (size_t)omp_get_thread_num() * HIST_STRIDE;
        long k;

        /* No atomics here: `mine` is this thread's own padded slab.
         *
         * The increment goes through a volatile lvalue on purpose.  Apple
         * clang 21 (Xcode 26) at -O2 and above miscompiles this exact
         * pattern -- a schedule(runtime) loop scattering increments into a
         * per-thread array -- and silently drops counts: on a 96x96x3 test
         * image, 27,648 pixels produced a histogram summing to 19,633.  It
         * is deterministic, so it is not a race, and GCC, Apple clang at
         * -O0/-O1 and every other kernel are unaffected.  Forcing each
         * increment to be a real load and store removes the bad transform
         * and costs nothing measurable: the loop is a read-modify-write of
         * memory either way.  Found by the macOS CI job via the
         * serial == OpenMP test; see report section 6.6. */
#pragma omp for schedule(runtime)
        for (k = 0; k < n; ++k) {
            volatile long *slot = &mine[bin_of(data[k])];
            *slot += 1;
        }
    }

    /* Merge: nthreads * 256 additions, negligible next to n. */
    memset(hist, 0, HIST_BINS * sizeof(long));
    for (t = 0; t < nthreads; ++t) {
        const long *mine = local + (size_t)t * HIST_STRIDE;
        for (b = 0; b < HIST_BINS; ++b) {
            hist[b] += mine[b];
        }
    }
    free(local);
    (void)i;
#else
    histogram_serial(data, n, hist);
#endif
}

/* ------------------------------------------------------------------ */
/* Phase 2: build a 256-entry lookup table from the histogram          */
/* ------------------------------------------------------------------ */
/*
 * Equalisation maps each intensity through the cumulative distribution
 * function.  Since the CDF of a random variable is uniformly distributed,
 * the output histogram comes out approximately flat, which maximises global
 * contrast.  Being a CDF, the map is monotone -- pixel ORDER is preserved.
 *
 * Collapsing it to a 256-entry LUT means phase 2 is a table lookup per
 * pixel instead of a search, and the LUT is small enough to stay in L1 in
 * every thread.
 */
static void build_equalize_lut(const long *hist, long total, float *lut)
{
    long cdf = 0, cdf_min = 0;
    int b;

    for (b = 0; b < HIST_BINS; ++b) {
        if (hist[b] != 0) {
            cdf_min = hist[b];
            break;
        }
    }
    if (total <= cdf_min) {              /* constant image: identity map */
        for (b = 0; b < HIST_BINS; ++b) {
            lut[b] = (float)b;
        }
        return;
    }
    for (b = 0; b < HIST_BINS; ++b) {
        cdf += hist[b];
        lut[b] = (float)(cdf - cdf_min) * 255.0f / (float)(total - cdf_min);
    }
}

/* ------------------------------------------------------------------ */
/* Phase 2 kernels: the point transforms                              */
/* ------------------------------------------------------------------ */
/*
 * The serial and OpenMP versions are written as SEPARATE functions rather
 * than one function with `#pragma omp parallel for if(parallel)`.
 *
 * This is not stylistic.  With an `if` clause GCC still emits the full
 * libgomp loop protocol -- GOMP_loop_runtime_start / _next / _end -- even
 * when the condition is false, and that call overhead is charged to the
 * "serial" baseline.  Measured on a 512x512x3 image, the `if(0)` version of
 * the min-max map ran roughly 7x slower than the same loop written without
 * any directive, which would have made every speedup figure in the report
 * meaningless.  Keeping the two paths textually separate guarantees the
 * baseline is what it claims to be: ordinary C.
 */
static void apply_affine_serial(const float *restrict src, float *restrict dst,
                                long n, float scale, float offset)
{
    long i;
    for (i = 0; i < n; ++i) {
        dst[i] = src[i] * scale + offset;
    }
}

static void apply_affine_omp(const float *restrict src, float *restrict dst,
                             long n, float scale, float offset)
{
    long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(src, dst) firstprivate(n, scale, offset)
#endif
    for (i = 0; i < n; ++i) {
        dst[i] = src[i] * scale + offset;
    }
}

static void apply_lut_serial(const float *restrict src, float *restrict dst,
                             long n, const float *restrict lut)
{
    long i;
    for (i = 0; i < n; ++i) {
        dst[i] = lut[bin_of(src[i])];
    }
}

static void apply_lut_omp(const float *restrict src, float *restrict dst,
                          long n, const float *restrict lut)
{
    long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) default(none) \
        shared(src, dst, lut) firstprivate(n)
#endif
    for (i = 0; i < n; ++i) {
        dst[i] = lut[bin_of(src[i])];
    }
}

/* ------------------------------------------------------------------ */
/* Driver                                                             */
/* ------------------------------------------------------------------ */
static Image *normalize_impl(const Image *in, norm_method method, int parallel)
{
    Image *out;
    long n;

    if (!in || method < 0 || method >= NORM_METHOD_COUNT) {
        return NULL;
    }
    out = image_alloc(in->height, in->width, in->channels);
    if (!out) {
        return NULL;
    }
    n = (long)image_elements(in);

    if (method == NORM_MINMAX) {
        float lo, hi, scale, offset;

        if (parallel) {
            stats_minmax_omp(in->data, n, &lo, &hi);
        } else {
            stats_minmax_serial(in->data, n, &lo, &hi);
        }
        /* A constant image has zero range.  Return black rather than NaN --
         * a defined, documented answer beats an infectious NaN. */
        if (hi - lo < 1e-12f) {
            scale = 0.0f;
            offset = 0.0f;
        } else {
            scale = 255.0f / (hi - lo);
            offset = -lo * scale;
        }

        if (parallel) {
            apply_affine_omp(in->data, out->data, n, scale, offset);
        } else {
            apply_affine_serial(in->data, out->data, n, scale, offset);
        }

    } else if (method == NORM_ZSCORE) {
        double mean, stddev;
        float scale, offset;

        if (parallel) {
            stats_moments_omp(in->data, n, &mean, &stddev);
        } else {
            stats_moments_serial(in->data, n, &mean, &stddev);
        }
        /*
         * A z-score is centred on 0 and unbounded, which cannot be stored in
         * an 8-bit image.  The result is re-centred on 128 and scaled so that
         * +/- 2 standard deviations spans the display range -- the standard
         * way of *visualising* a standardised image.  The statistical
         * transform is unchanged; only the viewing window is applied.
         */
        if (stddev < 1e-12) {
            scale = 0.0f;
            offset = 128.0f;
        } else {
            scale = (float)(64.0 / stddev);
            offset = (float)(128.0 - 64.0 * mean / stddev);
        }

        if (parallel) {
            apply_affine_omp(in->data, out->data, n, scale, offset);
        } else {
            apply_affine_serial(in->data, out->data, n, scale, offset);
        }

    } else {                             /* NORM_EQUALIZE */
        long hist[HIST_BINS];
        float lut[HIST_BINS];

        if (parallel) {
            histogram_omp(in->data, n, hist);
        } else {
            histogram_serial(in->data, n, hist);
        }
        build_equalize_lut(hist, n, lut);

        if (parallel) {
            apply_lut_omp(in->data, out->data, n, lut);
        } else {
            apply_lut_serial(in->data, out->data, n, lut);
        }
    }

    return out;
}

Image *normalize_serial(const Image *in, norm_method method)
{
    return normalize_impl(in, method, 0);
}

Image *normalize_omp(const Image *in, norm_method method)
{
    return normalize_impl(in, method, 1);
}
