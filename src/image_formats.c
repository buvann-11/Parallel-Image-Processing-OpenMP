/*
 * image_formats.c -- format-dispatching image I/O.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * image_read() / image_write() choose the codec from the file extension:
 *
 *   .ppm .pgm             -> the project's own Netpbm code (image_io.c)
 *   .png .jpg .jpeg .bmp  -> stb_image / stb_image_write
 *   .tga                     (Sean Barrett, public domain; third_party/stb)
 *
 * Every decoder hands back 8-bit samples, which are widened to the float
 * [0, 255] working range used by all kernels -- so a PNG and the equivalent
 * PPM produce bit-identical results downstream.  Grey+alpha and RGBA files
 * are reduced to 1 and 3 channels respectively: no kernel here models
 * transparency.
 */

#include "imgproc.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define JPEG_QUALITY 95

typedef enum { FMT_NETPBM, FMT_PNG, FMT_JPEG, FMT_BMP, FMT_TGA, FMT_UNKNOWN }
    file_format;

static file_format format_from_path(const char *path)
{
    const char *dot = strrchr(path, '.');
    char ext[8];
    size_t i;

    if (!dot || strlen(dot + 1) >= sizeof(ext)) {
        return FMT_UNKNOWN;
    }
    for (i = 0; dot[1 + i] != '\0'; ++i) {
        ext[i] = (char)tolower((unsigned char)dot[1 + i]);
    }
    ext[i] = '\0';

    if (!strcmp(ext, "ppm") || !strcmp(ext, "pgm") || !strcmp(ext, "pnm")) {
        return FMT_NETPBM;
    }
    if (!strcmp(ext, "png")) return FMT_PNG;
    if (!strcmp(ext, "jpg") || !strcmp(ext, "jpeg")) return FMT_JPEG;
    if (!strcmp(ext, "bmp")) return FMT_BMP;
    if (!strcmp(ext, "tga")) return FMT_TGA;
    return FMT_UNKNOWN;
}

int image_format_supported(const char *path)
{
    return path != NULL && format_from_path(path) != FMT_UNKNOWN;
}

Image *image_read(const char *path, ip_status *status)
{
    int width = 0, height = 0, file_channels = 0, channels;
    unsigned char *pixels;
    Image *image;
    size_t count, i;

    if (status) *status = IP_OK;
    if (!path) {
        if (status) *status = IP_ERR_ARG;
        return NULL;
    }

    /* Netpbm keeps its own reader: it is the reference path the tests and
     * the NumPy cross-validation were built against. */
    if (format_from_path(path) == FMT_NETPBM) {
        return image_read_netpbm(path, status);
    }

    /* Anything else: let stb sniff the content (it does not trust the
     * extension), keeping grey images grey and dropping alpha. */
    if (!stbi_info(path, &width, &height, &file_channels)) {
        /* Not an image stb understands -- maybe Netpbm without the usual
         * extension. */
        return image_read_netpbm(path, status);
    }
    channels = (file_channels <= 2) ? 1 : 3;

    pixels = stbi_load(path, &width, &height, &file_channels, channels);
    if (!pixels) {
        if (status) *status = IP_ERR_FORMAT;
        return NULL;
    }

    image = image_alloc(height, width, channels);
    if (!image) {
        stbi_image_free(pixels);
        if (status) *status = IP_ERR_ALLOC;
        return NULL;
    }
    count = (size_t)width * (size_t)height * (size_t)channels;
    for (i = 0; i < count; ++i) {
        image->data[i] = (float)pixels[i];
    }
    stbi_image_free(pixels);
    return image;
}

ip_status image_write(const Image *image, const char *path)
{
    unsigned char *raw;
    size_t count, i;
    int ok = 0;
    const file_format fmt = path ? format_from_path(path) : FMT_UNKNOWN;

    if (!image || !path) {
        return IP_ERR_ARG;
    }
    if (fmt == FMT_NETPBM) {
        return image_write_netpbm(image, path);
    }
    if (fmt == FMT_UNKNOWN) {
        return IP_ERR_FORMAT;
    }
    if (image->channels != 1 && image->channels != 3) {
        return IP_ERR_FORMAT;
    }

    count = image_elements(image);
    raw = (unsigned char *)malloc(count);
    if (!raw) {
        return IP_ERR_ALLOC;
    }
    /* Same rounding and clamping as the Netpbm writer, so the formats agree
     * byte for byte on lossless output. */
    for (i = 0; i < count; ++i) {
        const float v = image->data[i];
        if (!(v > 0.0f)) {
            raw[i] = 0;
        } else if (v >= 255.0f) {
            raw[i] = 255;
        } else {
            raw[i] = (unsigned char)(v + 0.5f);
        }
    }

    switch (fmt) {
    case FMT_PNG:
        ok = stbi_write_png(path, image->width, image->height, image->channels,
                            raw, image->width * image->channels);
        break;
    case FMT_JPEG:
        ok = stbi_write_jpg(path, image->width, image->height, image->channels,
                            raw, JPEG_QUALITY);
        break;
    case FMT_BMP:
        ok = stbi_write_bmp(path, image->width, image->height, image->channels,
                            raw);
        break;
    case FMT_TGA:
        ok = stbi_write_tga(path, image->width, image->height, image->channels,
                            raw);
        break;
    default:
        ok = 0;
        break;
    }
    free(raw);
    return ok ? IP_OK : IP_ERR_IO;
}
