/*
 * main.c -- command-line driver for the parallel image-processing library.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * Every operation runs BOTH the serial and the OpenMP version by default and
 * reports the measured speedup, so any command doubles as a micro-benchmark.
 * With --json the same numbers are emitted as a single JSON object, which is
 * how the graphical front-end drives this binary.
 *
 * Usage:
 *   imgproc info
 *   imgproc gen       OUT.ppm        --size 512 [--gray]
 *   imgproc grayscale IN.ppm OUT.pgm --method bt601|bt709|average|luminance
 *   imgproc normalize IN.ppm OUT.ppm --method minmax|zscore|equalize
 *   imgproc resize    IN.ppm OUT.ppm --size H W [--method ...] [--no-antialias]
 *   imgproc pad       IN.ppm OUT.ppm --amount 32 [--mode zeros|reflect|...]
 *   imgproc pool      IN.ppm OUT.ppm [--type max|avg] [--size 2] [--stride 2]
 *   imgproc conv      IN.ppm OUT.ppm --kernel box|gauss|sobelx|sobely|laplacian|sharpen
 *   imgproc convert   IN OUT         (format conversion only, no timing)
 *
 * IN and OUT may be .ppm/.pgm, .png, .jpg/.jpeg, .bmp or .tga; the format
 * is chosen from the extension.
 *
 * Common options:
 *   --threads N     number of OpenMP threads (default: all available)
 *   --repeat N      time N runs and report the best (default: 3)
 *   --schedule S    static | dynamic | guided
 *   --json          machine-readable output
 *   --quiet         suppress the human-readable banner
 */

#include "imgproc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Built-in convolution kernels                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *name;
    int size;
    float data[25];
} NamedKernel;

static const NamedKernel KERNELS[] = {
    { "box", 3, { 1.f/9, 1.f/9, 1.f/9,
                  1.f/9, 1.f/9, 1.f/9,
                  1.f/9, 1.f/9, 1.f/9 } },
    { "gauss", 5, { 1.f/273,  4.f/273,  7.f/273,  4.f/273, 1.f/273,
                    4.f/273, 16.f/273, 26.f/273, 16.f/273, 4.f/273,
                    7.f/273, 26.f/273, 41.f/273, 26.f/273, 7.f/273,
                    4.f/273, 16.f/273, 26.f/273, 16.f/273, 4.f/273,
                    1.f/273,  4.f/273,  7.f/273,  4.f/273, 1.f/273 } },
    { "sobelx", 3, { -1, 0, 1,
                     -2, 0, 2,
                     -1, 0, 1 } },
    { "sobely", 3, { -1, -2, -1,
                      0,  0,  0,
                      1,  2,  1 } },
    { "laplacian", 3, { 1,  1, 1,
                        1, -8, 1,
                        1,  1, 1 } },
    { "sharpen", 3, {  0, -1,  0,
                      -1,  5, -1,
                       0, -1,  0 } },
    { "emboss", 3, { -2, -1, 0,
                     -1,  1, 1,
                      0,  1, 2 } },
};
static const int KERNEL_COUNT = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));

static const NamedKernel *find_kernel(const char *name)
{
    int i;
    for (i = 0; i < KERNEL_COUNT; ++i) {
        if (strcmp(KERNELS[i].name, name) == 0) {
            return &KERNELS[i];
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Option parsing                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    int threads;
    int repeat;
    int json;
    int quiet;
    int antialias;
    int amount;
    int size_h, size_w;
    int pool_size, pool_stride;
    int gen_size, gen_gray;
    omp_schedule_kind schedule;
    pad_mode pad;
    gray_method gray;
    norm_method norm;
    resize_method resize;
    pool_type pool;
    const char *kernel;
} Options;

static void options_defaults(Options *o)
{
    memset(o, 0, sizeof(*o));
    o->threads = 0;                      /* 0 => leave runtime default */
    o->repeat = 3;
    o->antialias = 1;
    o->amount = 32;
    o->size_h = 256;
    o->size_w = 256;
    o->pool_size = 2;
    o->pool_stride = 2;
    o->gen_size = 512;
    o->schedule = SCHED_STATIC;
    o->pad = PAD_REFLECT;
    o->gray = GRAY_BT601;
    o->norm = NORM_MINMAX;
    o->resize = RESIZE_BILINEAR;
    o->pool = POOL_MAX;
    o->kernel = "gauss";
}

static int match_name(const char *value, const char *const *names, int count)
{
    int i;
    for (i = 0; i < count; ++i) {
        if (strcmp(value, names[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static const char *const PAD_NAMES[]    = { "zeros", "reflect", "replicate", "circular" };
static const char *const GRAY_NAMES[]   = { "bt601", "bt709", "average", "luminance" };
static const char *const NORM_NAMES[]   = { "minmax", "zscore", "equalize" };
static const char *const RESIZE_NAMES[] = { "nearest", "bilinear", "bicubic" };
static const char *const POOL_NAMES[]   = { "max", "avg" };
static const char *const SCHED_NAMES[]  = { "static", "dynamic", "guided" };

/* Returns the index after the last consumed argument, or -1 on error. */
static int parse_options(int argc, char **argv, int start, Options *o)
{
    int i = start;

    while (i < argc) {
        const char *arg = argv[i];

        if (strcmp(arg, "--threads") == 0 && i + 1 < argc) {
            o->threads = atoi(argv[++i]);
        } else if (strcmp(arg, "--repeat") == 0 && i + 1 < argc) {
            o->repeat = atoi(argv[++i]);
            if (o->repeat < 1) o->repeat = 1;
        } else if (strcmp(arg, "--json") == 0) {
            o->json = 1;
        } else if (strcmp(arg, "--quiet") == 0) {
            o->quiet = 1;
        } else if (strcmp(arg, "--no-antialias") == 0) {
            o->antialias = 0;
        } else if (strcmp(arg, "--gray") == 0) {
            o->gen_gray = 1;
        } else if (strcmp(arg, "--amount") == 0 && i + 1 < argc) {
            o->amount = atoi(argv[++i]);
        } else if (strcmp(arg, "--stride") == 0 && i + 1 < argc) {
            o->pool_stride = atoi(argv[++i]);
        } else if (strcmp(arg, "--size") == 0 && i + 1 < argc) {
            /* --size N        (pooling window / generated image)
             * --size H W      (resize target) */
            o->pool_size = atoi(argv[i + 1]);
            o->gen_size = atoi(argv[i + 1]);
            o->size_h = atoi(argv[i + 1]);
            ++i;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                o->size_w = atoi(argv[++i]);
            } else {
                o->size_w = o->size_h;
            }
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            int k = match_name(argv[++i], PAD_NAMES, 4);
            if (k < 0) { fprintf(stderr, "error: unknown pad mode '%s'\n", argv[i]); return -1; }
            o->pad = (pad_mode)k;
        } else if (strcmp(arg, "--schedule") == 0 && i + 1 < argc) {
            int k = match_name(argv[++i], SCHED_NAMES, 3);
            if (k < 0) { fprintf(stderr, "error: unknown schedule '%s'\n", argv[i]); return -1; }
            o->schedule = (omp_schedule_kind)k;
        } else if (strcmp(arg, "--type") == 0 && i + 1 < argc) {
            int k = match_name(argv[++i], POOL_NAMES, 2);
            if (k < 0) { fprintf(stderr, "error: unknown pool type '%s'\n", argv[i]); return -1; }
            o->pool = (pool_type)k;
        } else if (strcmp(arg, "--kernel") == 0 && i + 1 < argc) {
            o->kernel = argv[++i];
        } else if (strcmp(arg, "--method") == 0 && i + 1 < argc) {
            /* The valid set depends on the command; try each in turn. */
            const char *value = argv[++i];
            int k;
            if ((k = match_name(value, GRAY_NAMES, 4)) >= 0)   o->gray = (gray_method)k;
            else if ((k = match_name(value, NORM_NAMES, 3)) >= 0)  o->norm = (norm_method)k;
            else if ((k = match_name(value, RESIZE_NAMES, 3)) >= 0) o->resize = (resize_method)k;
            else { fprintf(stderr, "error: unknown method '%s'\n", value); return -1; }
        } else {
            fprintf(stderr, "error: unrecognised option '%s'\n", arg);
            return -1;
        }
        ++i;
    }
    return i;
}

/* ------------------------------------------------------------------ */
/* Timing helper                                                      */
/* ------------------------------------------------------------------ */
/*
 * The BEST of `repeat` runs is reported, not the mean.  A slow run is always
 * caused by external interference (scheduler preemption, another process,
 * a cold cache); the fastest run is the closest estimate of the code's own
 * cost.  This is standard practice in benchmarking.
 */
typedef struct {
    double serial_s;
    double omp_s;
    double speedup;
    double efficiency;
    int threads;
} Timing;

static void report(const Options *o, const char *operation, const char *detail,
                   const Timing *t, const Image *out_img, const Tensor *out_ten)
{
    int oh = 0, ow = 0, oc = 0;

    if (out_img) {
        oh = out_img->height; ow = out_img->width; oc = out_img->channels;
    } else if (out_ten) {
        oh = out_ten->h; ow = out_ten->w; oc = out_ten->c;
    }

    if (o->json) {
        printf("{\"operation\":\"%s\",\"detail\":\"%s\","
               "\"threads\":%d,\"serial_ms\":%.4f,\"omp_ms\":%.4f,"
               "\"speedup\":%.4f,\"efficiency\":%.4f,"
               "\"out_height\":%d,\"out_width\":%d,\"out_channels\":%d}\n",
               operation, detail, t->threads,
               t->serial_s * 1e3, t->omp_s * 1e3,
               t->speedup, t->efficiency, oh, ow, oc);
    } else if (!o->quiet) {
        printf("  operation : %s (%s)\n", operation, detail);
        printf("  output    : %d x %d x %d\n", oh, ow, oc);
        printf("  serial    : %8.3f ms\n", t->serial_s * 1e3);
        printf("  openmp    : %8.3f ms   (%d thread%s)\n",
               t->omp_s * 1e3, t->threads, t->threads == 1 ? "" : "s");
        printf("  speedup   : %8.2fx\n", t->speedup);
        printf("  efficiency: %7.1f%%\n", t->efficiency * 100.0);
    }
}

static void finish_timing(Timing *t, int threads)
{
    t->threads = threads;
    t->speedup = (t->omp_s > 0.0) ? t->serial_s / t->omp_s : 0.0;
    t->efficiency = (threads > 0) ? t->speedup / (double)threads : 0.0;
}

/* ------------------------------------------------------------------ */
/* info                                                               */
/* ------------------------------------------------------------------ */
static int cmd_info(const Options *o)
{
    if (o->json) {
        printf("{\"version\":\"%s\",\"openmp\":%s,\"openmp_spec\":%ld,"
               "\"max_threads\":%d}\n",
               IMGPROC_VERSION, ip_openmp_enabled() ? "true" : "false",
               ip_openmp_version(), ip_max_threads());
        return 0;
    }
    printf("imgproc %s  --  MTCS-103P mini project\n", IMGPROC_VERSION);
    printf("  OpenMP       : %s", ip_openmp_enabled() ? "enabled" : "DISABLED");
    if (ip_openmp_enabled()) {
        printf(" (specification %ld)", ip_openmp_version());
    }
    printf("\n");
    printf("  max threads  : %d\n", ip_max_threads());
    printf("\n");
    printf("  pad modes    : zeros reflect replicate circular\n");
    printf("  gray methods : bt601 bt709 average luminance\n");
    printf("  norm methods : minmax zscore equalize\n");
    printf("  resize       : nearest bilinear bicubic\n");
    printf("  pool types   : max avg\n");
    printf("  conv kernels : ");
    {
        int i;
        for (i = 0; i < KERNEL_COUNT; ++i) {
            printf("%s%s", KERNELS[i].name, i + 1 < KERNEL_COUNT ? " " : "\n");
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Command dispatch                                                   */
/* ------------------------------------------------------------------ */
static void usage(void)
{
    fprintf(stderr,
        "imgproc " IMGPROC_VERSION " -- parallel image processing (OpenMP)\n"
        "MTCS-103P High Performance Computing Laboratory, mini project\n\n"
        "Usage:\n"
        "  imgproc info\n"
        "  imgproc gen       OUT.ppm        [--size 512] [--gray]\n"
        "  imgproc grayscale IN.ppm OUT.pgm [--method bt601|bt709|average|luminance]\n"
        "  imgproc normalize IN.ppm OUT.ppm [--method minmax|zscore|equalize]\n"
        "  imgproc resize    IN.ppm OUT.ppm --size H W [--method nearest|bilinear|bicubic]\n"
        "                                   [--no-antialias]\n"
        "  imgproc pad       IN.ppm OUT.ppm [--amount 32] [--mode zeros|reflect|replicate|circular]\n"
        "  imgproc pool      IN.ppm OUT.ppm [--type max|avg] [--size 2] [--stride 2]\n"
        "  imgproc conv      IN.ppm OUT.ppm [--kernel box|gauss|sobelx|sobely|laplacian|sharpen|emboss]\n"
        "  imgproc convert   IN     OUT     (format conversion only)\n\n"
        "Images: .ppm .pgm .png .jpg .jpeg .bmp .tga (chosen by extension)\n\n"
        "Common options:\n"
        "  --threads N    OpenMP threads          --repeat N   timed repetitions (best is kept)\n"
        "  --schedule S   static|dynamic|guided   --json       machine-readable output\n"
        "  --quiet        suppress the banner\n");
}

int main(int argc, char **argv)
{
    Options o;
    const char *command;
    ip_status status = IP_OK;
    Image *input = NULL, *result = NULL;
    Tensor *tensor_result = NULL;
    Timing timing;
    char detail[128];
    int rc = 0, r, threads;

    if (argc < 2) {
        usage();
        return 2;
    }
    command = argv[1];
    options_defaults(&o);
    memset(&timing, 0, sizeof(timing));
    detail[0] = '\0';

    if (strcmp(command, "--help") == 0 || strcmp(command, "-h") == 0 ||
        strcmp(command, "help") == 0) {
        usage();
        return 0;
    }

    /* ---- info ---------------------------------------------------- */
    if (strcmp(command, "info") == 0) {
        if (parse_options(argc, argv, 2, &o) < 0) return 2;
        return cmd_info(&o);
    }

    /* ---- gen (no input image) ------------------------------------ */
    if (strcmp(command, "gen") == 0) {
        if (argc < 3) { usage(); return 2; }
        if (parse_options(argc, argv, 3, &o) < 0) return 2;
        result = image_make_test_pattern(o.gen_size, o.gen_gray ? 1 : 3);
        if (!result) {
            fprintf(stderr, "error: could not generate a %dx%d pattern\n",
                    o.gen_size, o.gen_size);
            return 1;
        }
        status = image_write(result, argv[2]);
        if (status != IP_OK) {
            fprintf(stderr, "error: writing '%s': %s\n", argv[2],
                    ip_status_string(status));
            image_free(result);
            return 1;
        }
        if (o.json) {
            printf("{\"operation\":\"gen\",\"out_height\":%d,\"out_width\":%d,"
                   "\"out_channels\":%d}\n",
                   result->height, result->width, result->channels);
        } else if (!o.quiet) {
            printf("wrote %s (%d x %d x %d)\n", argv[2],
                   result->height, result->width, result->channels);
        }
        image_free(result);
        return 0;
    }

    /* ---- convert (format conversion, no kernels) ----------------- */
    if (strcmp(command, "convert") == 0) {
        if (argc < 4) { usage(); return 2; }
        if (parse_options(argc, argv, 4, &o) < 0) return 2;
        input = image_read(argv[2], &status);
        if (!input) {
            fprintf(stderr, "error: reading '%s': %s\n", argv[2],
                    ip_status_string(status));
            return 1;
        }
        status = image_write(input, argv[3]);
        if (status != IP_OK) {
            fprintf(stderr, "error: writing '%s': %s\n", argv[3],
                    ip_status_string(status));
            image_free(input);
            return 1;
        }
        if (o.json) {
            printf("{\"operation\":\"convert\",\"out_height\":%d,"
                   "\"out_width\":%d,\"out_channels\":%d}\n",
                   input->height, input->width, input->channels);
        } else if (!o.quiet) {
            printf("wrote %s (%d x %d x %d)\n", argv[3],
                   input->height, input->width, input->channels);
        }
        image_free(input);
        return 0;
    }

    /* ---- operations taking IN and OUT ---------------------------- */
    if (argc < 4) {
        usage();
        return 2;
    }
    if (parse_options(argc, argv, 4, &o) < 0) {
        return 2;
    }

    if (o.threads > 0) {
        ip_set_threads(o.threads);
    }
    threads = ip_max_threads();
    imgproc_set_schedule(o.schedule, 0);

    input = image_read(argv[2], &status);
    if (!input) {
        fprintf(stderr, "error: reading '%s': %s\n", argv[2],
                ip_status_string(status));
        return 1;
    }

    if (!o.quiet && !o.json) {
        printf("\nimgproc %s\n", IMGPROC_VERSION);
        printf("  input     : %s (%d x %d x %d)\n", argv[2],
               input->height, input->width, input->channels);
        printf("  schedule  : %s\n", schedule_name(o.schedule));
    }

    /* Each branch times the serial path, then the OpenMP path, keeping the
     * best of `repeat` runs, and leaves the OpenMP result for writing. */
#define TIME_BLOCK(SERIAL_CALL, OMP_CALL, FREE_CALL, RESULT_VAR)        \
    do {                                                                \
        double best_s = 1e30, best_p = 1e30, t0, dt;                    \
        for (r = 0; r < o.repeat; ++r) {                                \
            void *tmp;                                                  \
            t0 = ip_wtime();                                            \
            tmp = (void *)(SERIAL_CALL);                                \
            dt = ip_wtime() - t0;                                       \
            if (dt < best_s) best_s = dt;                               \
            if (!tmp) { fprintf(stderr, "error: operation failed\n");    \
                        rc = 1; goto done; }                            \
            FREE_CALL(tmp);                                             \
        }                                                               \
        for (r = 0; r < o.repeat; ++r) {                                \
            double t1;                                                  \
            if (RESULT_VAR) { FREE_CALL(RESULT_VAR); RESULT_VAR = NULL; }\
            t0 = ip_wtime();                                            \
            RESULT_VAR = (OMP_CALL);                                    \
            t1 = ip_wtime() - t0;                                       \
            if (t1 < best_p) best_p = t1;                               \
            if (!RESULT_VAR) { fprintf(stderr, "error: operation failed\n"); \
                               rc = 1; goto done; }                     \
        }                                                               \
        timing.serial_s = best_s;                                       \
        timing.omp_s = best_p;                                          \
    } while (0)

    if (strcmp(command, "grayscale") == 0) {
        snprintf(detail, sizeof(detail), "%s", gray_method_name(o.gray));
        TIME_BLOCK(grayscale_serial(input, o.gray),
                   grayscale_omp(input, o.gray),
                   image_free, result);

    } else if (strcmp(command, "normalize") == 0) {
        snprintf(detail, sizeof(detail), "%s", norm_method_name(o.norm));
        TIME_BLOCK(normalize_serial(input, o.norm),
                   normalize_omp(input, o.norm),
                   image_free, result);

    } else if (strcmp(command, "resize") == 0) {
        snprintf(detail, sizeof(detail), "%s %dx%d%s",
                 resize_method_name(o.resize), o.size_h, o.size_w,
                 o.antialias ? "" : " no-aa");
        TIME_BLOCK(resize_serial(input, o.size_h, o.size_w, o.resize, o.antialias),
                   resize_omp(input, o.size_h, o.size_w, o.resize, o.antialias),
                   image_free, result);

    } else if (strcmp(command, "conv") == 0) {
        const NamedKernel *k = find_kernel(o.kernel);
        if (!k) {
            fprintf(stderr, "error: unknown kernel '%s'\n", o.kernel);
            rc = 2;
            goto done;
        }
        snprintf(detail, sizeof(detail), "%s %dx%d", k->name, k->size, k->size);
        TIME_BLOCK(filter_image_serial(input, k->data, k->size, k->size, o.pad),
                   filter_image_omp(input, k->data, k->size, k->size, o.pad),
                   image_free, result);

    } else if (strcmp(command, "pad") == 0 || strcmp(command, "pool") == 0) {
        Tensor *in_tensor = image_to_tensor(input);
        if (!in_tensor) {
            fprintf(stderr, "error: out of memory\n");
            rc = 1;
            goto done;
        }
        if (strcmp(command, "pad") == 0) {
            snprintf(detail, sizeof(detail), "%s %dpx",
                     pad_mode_name(o.pad), o.amount);
            TIME_BLOCK(pad2d_serial(in_tensor, o.amount, o.amount, o.pad),
                       pad2d_omp(in_tensor, o.amount, o.amount, o.pad),
                       tensor_free, tensor_result);
        } else {
            snprintf(detail, sizeof(detail), "%s %dx%d stride %d",
                     pool_type_name(o.pool), o.pool_size, o.pool_size,
                     o.pool_stride);
            TIME_BLOCK(pool2d_serial(in_tensor, o.pool, o.pool_size,
                                     o.pool_size, o.pool_stride, 0, 0),
                       pool2d_omp(in_tensor, o.pool, o.pool_size,
                                  o.pool_size, o.pool_stride, 0, 0),
                       tensor_free, tensor_result);
        }
        tensor_free(in_tensor);
        if (tensor_result) {
            result = tensor_to_image(tensor_result);
        }

    } else {
        fprintf(stderr, "error: unknown command '%s'\n\n", command);
        usage();
        rc = 2;
        goto done;
    }

#undef TIME_BLOCK

    finish_timing(&timing, threads);

    if (result) {
        status = image_write(result, argv[3]);
        if (status != IP_OK) {
            fprintf(stderr, "error: writing '%s': %s\n", argv[3],
                    ip_status_string(status));
            rc = 1;
            goto done;
        }
    }
    report(&o, command, detail, &timing, result, tensor_result);

done:
    image_free(input);
    image_free(result);
    tensor_free(tensor_result);
    return rc;
}
