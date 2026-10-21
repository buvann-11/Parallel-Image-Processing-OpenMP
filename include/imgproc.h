/*
 * imgproc.h -- Parallel image processing and CNN primitives with OpenMP.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 * Sri Sathya Sai Institute of Higher Learning
 *
 * ---------------------------------------------------------------------------
 * DATA LAYOUTS
 * ---------------------------------------------------------------------------
 * Two layouts are used, each where it is conventional:
 *
 *   Image  : HWC, channels last   -- image[(y * width + x) * channels + c]
 *            The layout used by PPM/PNG files and by every image library.
 *            Channels of one pixel are adjacent, so a per-pixel operation
 *            (e.g. greyscale) reads one contiguous cache line.
 *
 *   Tensor : NCHW, channels first -- tensor[((n * C + c) * H + y) * W + x]
 *            The layout used by PyTorch, Caffe and cuDNN.  A whole channel
 *            plane is contiguous, which is what a convolution kernel wants
 *            when it sweeps a window across one channel at a time.
 *
 * Choosing the right layout per operation is itself a performance decision;
 * see report/REPORT.md section 4.2.
 *
 * ---------------------------------------------------------------------------
 * PARALLELISATION CONTRACT
 * ---------------------------------------------------------------------------
 * Every kernel below exists in two forms:
 *
 *   <name>_serial()    -- plain C, no OpenMP directives at all.
 *   <name>_omp()       -- OpenMP parallel version.
 *
 * The serial version is a genuine sequential baseline, not the parallel
 * version run with one thread.  Speedup is always reported against it, so the
 * OpenMP runtime's own overhead is included in the measurement rather than
 * hidden -- measuring against a 1-thread parallel run is a well-known way to
 * make speedup numbers look better than they are.
 *
 * Every _omp() kernel writes to disjoint output elements, so no critical
 * sections, atomics or locks appear on any hot path.  The two operations that
 * genuinely need combination across threads -- statistics and histograms --
 * use OpenMP reductions and per-thread private buffers respectively.
 */

#ifndef IMGPROC_H
#define IMGPROC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Version                                                            */
/* ------------------------------------------------------------------ */
#define IMGPROC_VERSION "1.0.0"

/* ------------------------------------------------------------------ */
/* Error codes                                                        */
/* ------------------------------------------------------------------ */
typedef enum {
    IP_OK = 0,
    IP_ERR_ALLOC = -1,      /* out of memory                           */
    IP_ERR_ARG = -2,        /* invalid argument                        */
    IP_ERR_SHAPE = -3,      /* incompatible shapes                     */
    IP_ERR_IO = -4,         /* file could not be read or written       */
    IP_ERR_FORMAT = -5      /* unsupported or malformed file format    */
} ip_status;

/* Human-readable text for an ip_status value. */
const char *ip_status_string(ip_status status);

/* ------------------------------------------------------------------ */
/* Image (HWC, float, values conventionally in [0, 255])              */
/* ------------------------------------------------------------------ */
typedef struct {
    int height;
    int width;
    int channels;   /* 1 = greyscale, 3 = RGB */
    float *data;    /* height * width * channels floats, row-major HWC */
} Image;

Image      *image_alloc(int height, int width, int channels);
void        image_free(Image *image);
Image      *image_clone(const Image *src);
size_t      image_elements(const Image *image);

/* Indexing helper: value of channel c at (y, x). */
static inline float image_at(const Image *im, int y, int x, int c)
{
    return im->data[((size_t)y * im->width + x) * im->channels + c];
}

/* ------------------------------------------------------------------ */
/* Tensor (NCHW, float) -- the CNN feature-map type                   */
/* ------------------------------------------------------------------ */
typedef struct {
    int n;          /* batch size      */
    int c;          /* channels        */
    int h;          /* height          */
    int w;          /* width           */
    float *data;    /* n * c * h * w floats, NCHW order */
} Tensor;

Tensor     *tensor_alloc(int n, int c, int h, int w);
void        tensor_free(Tensor *tensor);
size_t      tensor_elements(const Tensor *tensor);

/* Layout conversions between the two worlds. */
Tensor     *image_to_tensor(const Image *image);      /* HWC -> 1CHW */
Image      *tensor_to_image(const Tensor *tensor);    /* 1CHW -> HWC */

/* ------------------------------------------------------------------ */
/* Image file I/O -- Netpbm natively; PNG/JPEG/BMP/TGA via stb        */
/* ------------------------------------------------------------------ */
/*
 * Netpbm is used deliberately: the format is a ~20-line header plus raw
 * bytes, so the C side needs no external image library at all (no libpng,
 * no libjpeg, nothing to install on the evaluator's machine).  Common
 * formats (PNG, JPEG, ...) are handled by image_read()/image_write() below
 * through the vendored, dependency-free stb headers.
 */
Image      *image_read_netpbm(const char *path, ip_status *status);
ip_status   image_write_netpbm(const Image *image, const char *path);

/*
 * Format-dispatching I/O (image_formats.c).  The codec is chosen from the
 * extension: .ppm/.pgm use the Netpbm code above; .png/.jpg/.jpeg/.bmp/.tga
 * use the vendored stb single-header libraries (third_party/stb).  Decoded
 * pixels land in the same float [0,255] HWC layout, so every kernel is
 * format-agnostic.  RGBA and grey+alpha inputs drop their alpha channel.
 */
Image      *image_read(const char *path, ip_status *status);
ip_status   image_write(const Image *image, const char *path);
int         image_format_supported(const char *path);

/* Deterministic synthetic test images -- no input files required. */
Image      *image_make_test_pattern(int size, int channels);

/* ------------------------------------------------------------------ */
/* Enumerations                                                       */
/* ------------------------------------------------------------------ */

/* Border handling for padding and convolution. */
typedef enum {
    PAD_ZEROS = 0,      /* 0 0 | a b c d | 0 0   -- the CNN default        */
    PAD_REFLECT,        /* c b | a b c d | c b   -- mirror, edge not repeated */
    PAD_REPLICATE,      /* a a | a b c d | d d   -- clamp to edge          */
    PAD_CIRCULAR,       /* c d | a b c d | a b   -- wrap around            */
    PAD_MODE_COUNT
} pad_mode;

/* Greyscale conversion formula. */
typedef enum {
    GRAY_BT601 = 0,     /* 0.299 R + 0.587 G + 0.114 B  (SDTV; OpenCV/PIL) */
    GRAY_BT709,         /* 0.2126 R + 0.7152 G + 0.0722 B (HDTV / sRGB)    */
    GRAY_AVERAGE,       /* (R + G + B) / 3   -- perceptually wrong         */
    GRAY_LUMINANCE,     /* BT.709 with sRGB gamma linearisation            */
    GRAY_METHOD_COUNT
} gray_method;

/* Intensity normalisation scheme. */
typedef enum {
    NORM_MINMAX = 0,    /* rescale observed range to [0, 255]              */
    NORM_ZSCORE,        /* zero mean, unit variance, re-centred on 128     */
    NORM_EQUALIZE,      /* histogram equalisation                          */
    NORM_METHOD_COUNT
} norm_method;

/* Resampling kernel. */
typedef enum {
    RESIZE_NEAREST = 0,
    RESIZE_BILINEAR,
    RESIZE_BICUBIC,     /* Keys cubic, a = -0.5 (Catmull-Rom)              */
    RESIZE_METHOD_COUNT
} resize_method;

/* Pooling reduction. */
typedef enum {
    POOL_MAX = 0,
    POOL_AVG,
    POOL_TYPE_COUNT
} pool_type;

/* OpenMP loop schedule, selectable at run time for the scaling study. */
typedef enum {
    SCHED_STATIC = 0,
    SCHED_DYNAMIC,
    SCHED_GUIDED,
    SCHED_COUNT
} omp_schedule_kind;

/* Name <-> enum helpers (used by the CLI and the GUI). */
const char *pad_mode_name(pad_mode mode);
const char *gray_method_name(gray_method method);
const char *norm_method_name(norm_method method);
const char *resize_method_name(resize_method method);
const char *pool_type_name(pool_type type);
const char *schedule_name(omp_schedule_kind kind);

/* Select the OpenMP schedule used by subsequent _omp() calls. */
void        imgproc_set_schedule(omp_schedule_kind kind, int chunk);

/* ------------------------------------------------------------------ */
/* PART 1 -- CNN primitives                                           */
/* ------------------------------------------------------------------ */

/* Output spatial size of a sliding-window operation along one axis. */
int conv_output_size(int in_size, int kernel, int stride,
                     int pad_before, int pad_after, int dilation);

/* --- Padding ------------------------------------------------------ */
/* Pads the spatial dimensions of an NCHW tensor.  Returns a new tensor
 * of shape (n, c, h + pad_h*2, w + pad_w*2), or NULL on failure. */
Tensor *pad2d_serial(const Tensor *in, int pad_h, int pad_w, pad_mode mode);
Tensor *pad2d_omp   (const Tensor *in, int pad_h, int pad_w, pad_mode mode);

/* --- Convolution -------------------------------------------------- */
/*
 * Cross-correlation, the operation every DL framework calls "convolution":
 *
 *   y[n,f,p,q] = bias[f] + sum_{c,i,j} w[f,c,i,j]
 *                * x[n, c, p*stride + i*dilation, q*stride + j*dilation]
 *
 * weights : (out_channels, in->c, kh, kw), contiguous
 * bias    : out_channels floats, or NULL for no bias
 */
Tensor *conv2d_serial(const Tensor *in, const float *weights, const float *bias,
                      int out_channels, int kh, int kw,
                      int stride, int pad_h, int pad_w, int dilation,
                      pad_mode mode);
Tensor *conv2d_omp   (const Tensor *in, const float *weights, const float *bias,
                      int out_channels, int kh, int kw,
                      int stride, int pad_h, int pad_w, int dilation,
                      pad_mode mode);

/* --- Pooling ------------------------------------------------------ */
Tensor *pool2d_serial(const Tensor *in, pool_type type,
                      int kh, int kw, int stride, int pad_h, int pad_w);
Tensor *pool2d_omp   (const Tensor *in, pool_type type,
                      int kh, int kw, int stride, int pad_h, int pad_w);

/* Global average pooling: (N, C, H, W) -> (N, C, 1, 1). */
Tensor *global_avg_pool_serial(const Tensor *in);
Tensor *global_avg_pool_omp   (const Tensor *in);

/* ------------------------------------------------------------------ */
/* PART 2 -- Image processing                                         */
/* ------------------------------------------------------------------ */

/* --- Greyscale ---------------------------------------------------- */
/* RGB (3-channel) -> greyscale (1-channel).  A 1-channel input is copied. */
Image *grayscale_serial(const Image *in, gray_method method);
Image *grayscale_omp   (const Image *in, gray_method method);

/* --- Normalisation ------------------------------------------------ */
Image *normalize_serial(const Image *in, norm_method method);
Image *normalize_omp   (const Image *in, norm_method method);

/* --- Resize ------------------------------------------------------- */
/*
 * Separable resampling.  When `antialias` is non-zero and the image is being
 * reduced, the reconstruction kernel is widened by the downscaling factor so
 * that every input pixel contributes -- without this, decimation aliases.
 */
Image *resize_serial(const Image *in, int out_h, int out_w,
                     resize_method method, int antialias);
Image *resize_omp   (const Image *in, int out_h, int out_w,
                     resize_method method, int antialias);

/* --- Convenience: filter an image with one 2-D kernel (depthwise) -- */
Image *filter_image_serial(const Image *in, const float *kernel, int kh, int kw,
                           pad_mode mode);
Image *filter_image_omp   (const Image *in, const float *kernel, int kh, int kw,
                           pad_mode mode);

/* ------------------------------------------------------------------ */
/* Timing                                                             */
/* ------------------------------------------------------------------ */
/* Wall-clock seconds.  Uses omp_get_wtime() when compiled with OpenMP,
 * clock_gettime(CLOCK_MONOTONIC) otherwise. */
double ip_wtime(void);

/* Number of threads the OpenMP runtime would use for a parallel region. */
int    ip_max_threads(void);

/* Set the number of threads for subsequent parallel regions. */
void   ip_set_threads(int threads);

/* Non-zero when the library was compiled with -fopenmp. */
int    ip_openmp_enabled(void);

/* OpenMP specification version the compiler reported (e.g. 201511), or 0. */
long   ip_openmp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* IMGPROC_H */
