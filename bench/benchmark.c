/*
 * benchmark.c -- strong-scaling study of every parallel kernel.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * ---------------------------------------------------------------------------
 * WHAT IS MEASURED AND WHY
 * ---------------------------------------------------------------------------
 * This is a STRONG scaling study: the problem size is held fixed and the
 * thread count is increased.  The quantities reported are
 *
 *   Speedup        S(p) = T_serial / T(p)
 *   Efficiency     E(p) = S(p) / p
 *   Karp-Flatt     e    = (1/S(p) - 1/p) / (1 - 1/p)
 *
 * T_serial is the genuinely sequential implementation -- code compiled with
 * no OpenMP directives on the path at all -- NOT the parallel version run
 * with one thread.  Benchmarking against a 1-thread parallel run silently
 * hides the cost of the OpenMP runtime and inflates every speedup figure.
 *
 * The Karp-Flatt metric is the experimentally determined serial fraction.
 * It is more informative than speedup alone: if e stays CONSTANT as p grows,
 * the loss of efficiency is genuine Amdahl serial work; if e GROWS with p,
 * the loss comes from parallel overhead -- scheduling, synchronisation,
 * memory contention.  The two have completely different remedies, and the
 * report uses this to explain why the memory-bound kernels flatten out.
 *
 * ---------------------------------------------------------------------------
 * METHODOLOGY
 * ---------------------------------------------------------------------------
 *   - One untimed warm-up run per configuration, so the measurement is not
 *     polluted by first-touch page faults and a cold cache.
 *   - The BEST of N timed runs is kept.  A slow run is always external
 *     interference; the fastest is the closest estimate of the code's cost.
 *   - Results are printed as a table and optionally written as CSV for
 *     plotting.
 */

#include "imgproc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Workload definitions                                               */
/* ------------------------------------------------------------------ */
typedef enum {
    WL_GRAYSCALE,
    WL_GRAYSCALE_GAMMA,
    WL_NORM_MINMAX,
    WL_NORM_EQUALIZE,
    WL_RESIZE,
    WL_PAD,
    WL_POOL,
    WL_CONV3,
    WL_CONV5,
    WL_CONV_CNN,
    WL_COUNT
} workload_id;

typedef struct {
    const char *name;
    const char *note;        /* what this kernel stresses */
} WorkloadInfo;

static const WorkloadInfo WORKLOADS[WL_COUNT] = {
    { "grayscale",       "elementwise map, memory bound" },
    { "grayscale_gamma", "elementwise map with powf, compute bound" },
    { "norm_minmax",     "min/max reduction + map" },
    { "norm_equalize",   "private histograms + LUT map" },
    { "resize_bicubic",  "two separable passes, barrier between" },
    { "pad_reflect",     "pure gather, bandwidth bound" },
    { "pool_max_2x2",    "windowed reduction, bandwidth bound" },
    { "conv_3x3",        "depthwise image filter" },
    { "conv_5x5",        "depthwise image filter, more arithmetic" },
    { "conv_cnn_16ch",   "multi-channel CNN convolution, compute bound" },
};

/* The CNN workload needs weights; allocated once and reused. */
typedef struct {
    Image *image;
    Tensor *tensor;
    float *cnn_weights;
    int cnn_out_channels;
    float kernel3[9];
    float kernel5[25];
} Fixtures;

/* ------------------------------------------------------------------ */
/* One run of one workload                                            */
/* ------------------------------------------------------------------ */
/* Returns elapsed seconds, or -1.0 on failure. */
static double run_once(workload_id id, const Fixtures *fx, int parallel)
{
    double t0, elapsed;
    void *result = NULL;

    t0 = ip_wtime();
    switch (id) {
    case WL_GRAYSCALE:
        result = parallel ? (void *)grayscale_omp(fx->image, GRAY_BT601)
                          : (void *)grayscale_serial(fx->image, GRAY_BT601);
        break;
    case WL_GRAYSCALE_GAMMA:
        result = parallel ? (void *)grayscale_omp(fx->image, GRAY_LUMINANCE)
                          : (void *)grayscale_serial(fx->image, GRAY_LUMINANCE);
        break;
    case WL_NORM_MINMAX:
        result = parallel ? (void *)normalize_omp(fx->image, NORM_MINMAX)
                          : (void *)normalize_serial(fx->image, NORM_MINMAX);
        break;
    case WL_NORM_EQUALIZE:
        result = parallel ? (void *)normalize_omp(fx->image, NORM_EQUALIZE)
                          : (void *)normalize_serial(fx->image, NORM_EQUALIZE);
        break;
    case WL_RESIZE:
        result = parallel
            ? (void *)resize_omp(fx->image, fx->image->height / 2,
                                 fx->image->width / 2, RESIZE_BICUBIC, 1)
            : (void *)resize_serial(fx->image, fx->image->height / 2,
                                    fx->image->width / 2, RESIZE_BICUBIC, 1);
        break;
    case WL_PAD:
        result = parallel ? (void *)pad2d_omp(fx->tensor, 32, 32, PAD_REFLECT)
                          : (void *)pad2d_serial(fx->tensor, 32, 32, PAD_REFLECT);
        break;
    case WL_POOL:
        result = parallel ? (void *)pool2d_omp(fx->tensor, POOL_MAX, 2, 2, 2, 0, 0)
                          : (void *)pool2d_serial(fx->tensor, POOL_MAX, 2, 2, 2, 0, 0);
        break;
    case WL_CONV3:
        result = parallel
            ? (void *)filter_image_omp(fx->image, fx->kernel3, 3, 3, PAD_REFLECT)
            : (void *)filter_image_serial(fx->image, fx->kernel3, 3, 3, PAD_REFLECT);
        break;
    case WL_CONV5:
        result = parallel
            ? (void *)filter_image_omp(fx->image, fx->kernel5, 5, 5, PAD_REFLECT)
            : (void *)filter_image_serial(fx->image, fx->kernel5, 5, 5, PAD_REFLECT);
        break;
    case WL_CONV_CNN:
        result = parallel
            ? (void *)conv2d_omp(fx->tensor, fx->cnn_weights, NULL,
                                 fx->cnn_out_channels, 3, 3, 1, 1, 1, 1, PAD_ZEROS)
            : (void *)conv2d_serial(fx->tensor, fx->cnn_weights, NULL,
                                    fx->cnn_out_channels, 3, 3, 1, 1, 1, 1, PAD_ZEROS);
        break;
    default:
        return -1.0;
    }
    elapsed = ip_wtime() - t0;

    if (!result) {
        return -1.0;
    }
    /* Image and Tensor results are freed with different functions. */
    if (id == WL_PAD || id == WL_POOL || id == WL_CONV_CNN) {
        tensor_free((Tensor *)result);
    } else {
        image_free((Image *)result);
    }
    return elapsed;
}

/* Best of `repeat` timed runs, after one untimed warm-up. */
static double time_best(workload_id id, const Fixtures *fx, int parallel,
                        int repeat)
{
    double best = 1e30;
    int r;

    if (run_once(id, fx, parallel) < 0.0) {
        return -1.0;
    }
    for (r = 0; r < repeat; ++r) {
        double t = run_once(id, fx, parallel);
        if (t < 0.0) {
            return -1.0;
        }
        if (t < best) {
            best = t;
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* Fixtures                                                           */
/* ------------------------------------------------------------------ */
static void fixtures_free(Fixtures *fx)
{
    image_free(fx->image);
    tensor_free(fx->tensor);
    free(fx->cnn_weights);
    memset(fx, 0, sizeof(*fx));
}

static int fixtures_build(Fixtures *fx, int size, int cnn_channels)
{
    int i;
    unsigned int seed = 12345u;

    memset(fx, 0, sizeof(*fx));

    fx->image = image_make_test_pattern(size, 3);
    if (!fx->image) {
        return 0;
    }
    fx->tensor = image_to_tensor(fx->image);
    if (!fx->tensor) {
        fixtures_free(fx);
        return 0;
    }

    for (i = 0; i < 9; ++i) {
        fx->kernel3[i] = 1.0f / 9.0f;
    }
    for (i = 0; i < 25; ++i) {
        fx->kernel5[i] = 1.0f / 25.0f;
    }

    /* Deterministic pseudo-random CNN weights: a fixed LCG rather than
     * rand(), so the benchmark is byte-for-byte reproducible on any libc. */
    fx->cnn_out_channels = cnn_channels;
    fx->cnn_weights = (float *)malloc((size_t)cnn_channels * fx->tensor->c
                                      * 3 * 3 * sizeof(float));
    if (!fx->cnn_weights) {
        fixtures_free(fx);
        return 0;
    }
    {
        const long total = (long)cnn_channels * fx->tensor->c * 9;
        long k;
        for (k = 0; k < total; ++k) {
            seed = seed * 1103515245u + 12345u;
            fx->cnn_weights[k] =
                ((float)((seed >> 16) & 0x7fffu) / 16383.5f) - 1.0f;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                          */
/* ------------------------------------------------------------------ */
static double karp_flatt(double speedup, int p)
{
    if (p <= 1 || speedup <= 0.0) {
        return 0.0;
    }
    return (1.0 / speedup - 1.0 / (double)p) / (1.0 - 1.0 / (double)p);
}

static void print_header(void)
{
    printf("\n%-18s %8s %9s %11s %11s %9s %7s %9s\n",
           "kernel", "sched", "threads", "serial(ms)", "omp(ms)",
           "speedup", "eff", "karp-flatt");
    printf("%-18s %8s %9s %11s %11s %9s %7s %9s\n",
           "------------------", "--------", "---------", "-----------",
           "-----------", "---------", "-------", "---------");
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
static void usage(void)
{
    fprintf(stderr,
        "benchmark -- strong-scaling study (MTCS-103P mini project)\n\n"
        "Usage: benchmark [options]\n"
        "  --quick             one small size, default schedule (a smoke test)\n"
        "  --full              every kernel, every thread count, all 3 schedules\n"
        "  --size N            image edge length (default 1024)\n"
        "  --repeat N          timed repetitions, best kept (default 5)\n"
        "  --channels N        CNN output channels for conv_cnn (default 16)\n"
        "  --chunk N           OpenMP chunk size (0 = runtime default)\n"
        "  --csv FILE          also write results as CSV\n"
        "  --help              this message\n");
}

int main(int argc, char **argv)
{
    int size = 1024, repeat = 5, channels = 16, chunk = 0;
    int full = 0, quick = 0;
    const char *csv_path = NULL;
    FILE *csv = NULL;
    Fixtures fx;
    int thread_list[32];
    int n_threads = 0, max_threads, i, s, w;
    int schedule_list[3] = { SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED };
    int n_schedules = 1;
    double serial_time[WL_COUNT];

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[i], "--full") == 0) {
            full = 1;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
            channels = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            chunk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unrecognised option '%s'\n", argv[i]);
            usage();
            return 2;
        }
    }
    if (quick) {
        size = 512;
        repeat = 3;
    }
    if (size < 32 || repeat < 1 || channels < 1) {
        fprintf(stderr, "error: invalid parameters\n");
        return 2;
    }

    max_threads = ip_max_threads();

    /* Thread counts: powers of two up to the core count, plus the core
     * count itself if it is not a power of two. */
    for (i = 1; i <= max_threads; i *= 2) {
        thread_list[n_threads++] = i;
    }
    if (n_threads == 0 || thread_list[n_threads - 1] != max_threads) {
        thread_list[n_threads++] = max_threads;
    }
    if (full) {
        n_schedules = 3;
    }

    printf("============================================================\n");
    printf(" MTCS-103P Mini Project -- OpenMP strong-scaling study\n");
    printf("============================================================\n");
    printf("  imgproc version : %s\n", IMGPROC_VERSION);
    printf("  OpenMP          : %s (spec %ld)\n",
           ip_openmp_enabled() ? "enabled" : "DISABLED", ip_openmp_version());
    printf("  cores available : %d\n", max_threads);
    printf("  image size      : %d x %d x 3  (%.1f MB as float)\n",
           size, size, (double)size * size * 3 * sizeof(float) / 1048576.0);
    printf("  CNN conv        : %d output channels, 3x3\n", channels);
    printf("  repetitions     : %d (best kept, after 1 warm-up)\n", repeat);
    printf("  chunk size      : %d %s\n", chunk,
           chunk == 0 ? "(runtime default: 1 for dynamic/guided)" : "(explicit)");
    if (!ip_openmp_enabled()) {
        printf("\n  WARNING: built without -fopenmp; all speedups will be 1.0\n");
    }

    if (!fixtures_build(&fx, size, channels)) {
        fprintf(stderr, "error: could not allocate benchmark fixtures\n");
        return 1;
    }

    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) {
            fprintf(stderr, "error: cannot write '%s'\n", csv_path);
            fixtures_free(&fx);
            return 1;
        }
        fprintf(csv, "kernel,note,size,schedule,threads,serial_ms,omp_ms,"
                     "speedup,efficiency,karp_flatt\n");
        /* Header and rows both use RFC 4180 quoting for the note column. */
    }

    /* Serial baselines, measured once with the runtime pinned to 1 thread so
     * nothing else is competing for the cores. */
    ip_set_threads(1);
    printf("\n  measuring sequential baselines ...\n");
    for (w = 0; w < WL_COUNT; ++w) {
        serial_time[w] = time_best((workload_id)w, &fx, 0, repeat);
        if (serial_time[w] < 0.0) {
            fprintf(stderr, "error: workload '%s' failed\n", WORKLOADS[w].name);
            fixtures_free(&fx);
            if (csv) fclose(csv);
            return 1;
        }
        printf("    %-18s %9.3f ms\n", WORKLOADS[w].name,
               serial_time[w] * 1e3);
    }

    print_header();

    for (s = 0; s < n_schedules; ++s) {
        imgproc_set_schedule((omp_schedule_kind)schedule_list[s], chunk);
        for (w = 0; w < WL_COUNT; ++w) {
            for (i = 0; i < n_threads; ++i) {
                const int p = thread_list[i];
                double omp_time, speedup, efficiency, e;

                ip_set_threads(p);
                omp_time = time_best((workload_id)w, &fx, 1, repeat);
                if (omp_time < 0.0) {
                    continue;
                }
                speedup = serial_time[w] / omp_time;
                efficiency = speedup / (double)p;
                e = karp_flatt(speedup, p);

                printf("%-18s %8s %9d %11.3f %11.3f %9.2f %6.1f%% %9.4f\n",
                       WORKLOADS[w].name,
                       schedule_name((omp_schedule_kind)schedule_list[s]),
                       p, serial_time[w] * 1e3, omp_time * 1e3,
                       speedup, efficiency * 100.0, e);

                if (csv) {
                    /* The note contains commas, so it must be quoted or
                     * the column count changes from row to row. */
                    fprintf(csv, "%s,\"%s\",%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                            WORKLOADS[w].name, WORKLOADS[w].note, size,
                            schedule_name((omp_schedule_kind)schedule_list[s]),
                            p, serial_time[w] * 1e3, omp_time * 1e3,
                            speedup, efficiency, e);
                }
            }
        }
    }

    if (csv) {
        fclose(csv);
        printf("\n  CSV written to %s\n", csv_path);
    }
    printf("\n  Note: efficiency above 100%% is possible and is not an error.\n");
    printf("  Splitting the image across threads gives each one a smaller\n");
    printf("  working set, so more of it fits in that core's private L2\n");
    printf("  cache -- a genuine superlinear cache effect.  See report 5.5.\n\n");

    fixtures_free(&fx);
    return 0;
}
