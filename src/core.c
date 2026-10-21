/*
 * core.c -- allocation, layout conversion, enumeration names, timing.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 */

/* posix_memalign() and clock_gettime() are POSIX, not ISO C99, so the
 * feature-test macro must be set before any system header is included. */
#define _POSIX_C_SOURCE 200112L

#include "imgproc.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Status strings                                                     */
/* ------------------------------------------------------------------ */
const char *ip_status_string(ip_status status)
{
    switch (status) {
    case IP_OK:         return "ok";
    case IP_ERR_ALLOC:  return "out of memory";
    case IP_ERR_ARG:    return "invalid argument";
    case IP_ERR_SHAPE:  return "incompatible shapes";
    case IP_ERR_IO:     return "file I/O error";
    case IP_ERR_FORMAT: return "unsupported or malformed file format";
    default:            return "unknown error";
    }
}

/* ------------------------------------------------------------------ */
/* Aligned allocation                                                 */
/* ------------------------------------------------------------------ */
/*
 * Buffers are 64-byte aligned, which is the cache-line size on every x86-64
 * CPU this is likely to run on.  Two reasons:
 *   1. Row starts land on line boundaries, so the vectoriser can use aligned
 *      loads instead of unaligned ones.
 *   2. Threads writing to different rows cannot end up sharing a cache line
 *      at a row boundary, which is one of the ways false sharing creeps in.
 */
#define IP_ALIGN 64

static void *ip_aligned_alloc(size_t bytes)
{
    void *ptr = NULL;
    size_t rounded = (bytes + IP_ALIGN - 1) / IP_ALIGN * IP_ALIGN;
    if (rounded == 0) {
        rounded = IP_ALIGN;
    }
    if (posix_memalign(&ptr, IP_ALIGN, rounded) != 0) {
        return NULL;
    }
    return ptr;
}

/* ------------------------------------------------------------------ */
/* Image                                                              */
/* ------------------------------------------------------------------ */
size_t image_elements(const Image *image)
{
    if (!image) {
        return 0;
    }
    return (size_t)image->height * image->width * image->channels;
}

Image *image_alloc(int height, int width, int channels)
{
    Image *image;

    if (height <= 0 || width <= 0 || channels <= 0) {
        return NULL;
    }
    image = (Image *)calloc(1, sizeof(Image));
    if (!image) {
        return NULL;
    }
    image->height = height;
    image->width = width;
    image->channels = channels;
    image->data = (float *)ip_aligned_alloc(
        (size_t)height * width * channels * sizeof(float));
    if (!image->data) {
        free(image);
        return NULL;
    }
    return image;
}

void image_free(Image *image)
{
    if (!image) {
        return;
    }
    free(image->data);
    free(image);
}

Image *image_clone(const Image *src)
{
    Image *copy;

    if (!src) {
        return NULL;
    }
    copy = image_alloc(src->height, src->width, src->channels);
    if (!copy) {
        return NULL;
    }
    memcpy(copy->data, src->data, image_elements(src) * sizeof(float));
    return copy;
}

/* ------------------------------------------------------------------ */
/* Tensor                                                             */
/* ------------------------------------------------------------------ */
size_t tensor_elements(const Tensor *tensor)
{
    if (!tensor) {
        return 0;
    }
    return (size_t)tensor->n * tensor->c * tensor->h * tensor->w;
}

Tensor *tensor_alloc(int n, int c, int h, int w)
{
    Tensor *tensor;

    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        return NULL;
    }
    tensor = (Tensor *)calloc(1, sizeof(Tensor));
    if (!tensor) {
        return NULL;
    }
    tensor->n = n;
    tensor->c = c;
    tensor->h = h;
    tensor->w = w;
    tensor->data = (float *)ip_aligned_alloc(
        (size_t)n * c * h * w * sizeof(float));
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    return tensor;
}

void tensor_free(Tensor *tensor)
{
    if (!tensor) {
        return;
    }
    free(tensor->data);
    free(tensor);
}

/* ------------------------------------------------------------------ */
/* Layout conversion                                                  */
/* ------------------------------------------------------------------ */
Tensor *image_to_tensor(const Image *image)
{
    Tensor *tensor;
    int y, x, c;

    if (!image) {
        return NULL;
    }
    tensor = tensor_alloc(1, image->channels, image->height, image->width);
    if (!tensor) {
        return NULL;
    }
    /* HWC -> CHW.  A strided gather; done once per run, so it is not on any
     * hot path and is left sequential for clarity. */
    for (c = 0; c < image->channels; ++c) {
        for (y = 0; y < image->height; ++y) {
            const float *row = image->data
                + ((size_t)y * image->width) * image->channels + c;
            float *dst = tensor->data
                + ((size_t)c * image->height + y) * image->width;
            for (x = 0; x < image->width; ++x) {
                dst[x] = row[(size_t)x * image->channels];
            }
        }
    }
    return tensor;
}

Image *tensor_to_image(const Tensor *tensor)
{
    Image *image;
    int y, x, c;

    if (!tensor || tensor->n != 1) {
        return NULL;
    }
    image = image_alloc(tensor->h, tensor->w, tensor->c);
    if (!image) {
        return NULL;
    }
    for (c = 0; c < tensor->c; ++c) {
        for (y = 0; y < tensor->h; ++y) {
            const float *src = tensor->data
                + ((size_t)c * tensor->h + y) * tensor->w;
            float *row = image->data
                + ((size_t)y * tensor->w) * tensor->c + c;
            for (x = 0; x < tensor->w; ++x) {
                row[(size_t)x * tensor->c] = src[x];
            }
        }
    }
    return image;
}

/* ------------------------------------------------------------------ */
/* Enumeration names                                                  */
/* ------------------------------------------------------------------ */
const char *pad_mode_name(pad_mode mode)
{
    switch (mode) {
    case PAD_ZEROS:     return "zeros";
    case PAD_REFLECT:   return "reflect";
    case PAD_REPLICATE: return "replicate";
    case PAD_CIRCULAR:  return "circular";
    default:            return "?";
    }
}

const char *gray_method_name(gray_method method)
{
    switch (method) {
    case GRAY_BT601:     return "bt601";
    case GRAY_BT709:     return "bt709";
    case GRAY_AVERAGE:   return "average";
    case GRAY_LUMINANCE: return "luminance";
    default:             return "?";
    }
}

const char *norm_method_name(norm_method method)
{
    switch (method) {
    case NORM_MINMAX:   return "minmax";
    case NORM_ZSCORE:   return "zscore";
    case NORM_EQUALIZE: return "equalize";
    default:            return "?";
    }
}

const char *resize_method_name(resize_method method)
{
    switch (method) {
    case RESIZE_NEAREST:  return "nearest";
    case RESIZE_BILINEAR: return "bilinear";
    case RESIZE_BICUBIC:  return "bicubic";
    default:              return "?";
    }
}

const char *pool_type_name(pool_type type)
{
    switch (type) {
    case POOL_MAX: return "max";
    case POOL_AVG: return "avg";
    default:       return "?";
    }
}

const char *schedule_name(omp_schedule_kind kind)
{
    switch (kind) {
    case SCHED_STATIC:  return "static";
    case SCHED_DYNAMIC: return "dynamic";
    case SCHED_GUIDED:  return "guided";
    default:            return "?";
    }
}

/* ------------------------------------------------------------------ */
/* Run-time schedule selection                                        */
/* ------------------------------------------------------------------ */
/*
 * The kernels use `schedule(runtime)`, and this function sets the schedule
 * the runtime will apply.  That lets the benchmark harness sweep
 * static/dynamic/guided without recompiling, which is what makes the
 * schedule comparison in the report reproducible.
 */
void imgproc_set_schedule(omp_schedule_kind kind, int chunk)
{
#ifdef _OPENMP
    omp_sched_t sched;

    switch (kind) {
    case SCHED_DYNAMIC: sched = omp_sched_dynamic; break;
    case SCHED_GUIDED:  sched = omp_sched_guided;  break;
    case SCHED_STATIC:
    default:            sched = omp_sched_static;  break;
    }
    omp_set_schedule(sched, chunk);
#else
    (void)kind;
    (void)chunk;
#endif
}

/* ------------------------------------------------------------------ */
/* Timing and thread control                                          */
/* ------------------------------------------------------------------ */
double ip_wtime(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int ip_max_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void ip_set_threads(int threads)
{
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
}

int ip_openmp_enabled(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

long ip_openmp_version(void)
{
#ifdef _OPENMP
    return (long)_OPENMP;
#else
    return 0L;
#endif
}
