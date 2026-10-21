/*
 * image_io.c -- Netpbm (PGM "P5" / PPM "P6") reader and writer, plus
 *               deterministic synthetic test patterns.
 *
 * MTCS-103P High Performance Computing Laboratory -- Mini Project
 *
 * Netpbm was chosen so that the C code has ZERO external dependencies: the
 * header is a few ASCII tokens and the body is raw bytes.  The evaluator
 * needs nothing but a C compiler.  PNG/JPEG input is handled by converting
 * to PPM in the Python front-end, and Tk displays PPM natively, so nothing
 * is lost.
 */

#include "imgproc.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Header parsing                                                     */
/* ------------------------------------------------------------------ */
/*
 * Netpbm headers allow whitespace and '#' comments between any two tokens,
 * so the tokens cannot simply be fscanf'd in one call.
 */
static int read_token(FILE *fp, long *value)
{
    int ch;

    for (;;) {
        ch = fgetc(fp);
        if (ch == EOF) {
            return 0;
        }
        if (ch == '#') {                     /* comment runs to end of line */
            while (ch != '\n' && ch != EOF) {
                ch = fgetc(fp);
            }
            continue;
        }
        if (!isspace(ch)) {
            break;
        }
    }
    if (!isdigit(ch)) {
        return 0;
    }

    *value = 0;
    while (isdigit(ch)) {
        *value = *value * 10 + (ch - '0');
        if (*value > (1L << 30)) {           /* refuse absurd dimensions */
            return 0;
        }
        ch = fgetc(fp);
    }
    return 1;
}

Image *image_read_netpbm(const char *path, ip_status *status)
{
    FILE *fp = NULL;
    Image *image = NULL;
    unsigned char *raw = NULL;
    char magic[3] = {0};
    long width = 0, height = 0, maxval = 0;
    int channels;
    size_t count, i;

    if (status) {
        *status = IP_OK;
    }
    if (!path) {
        if (status) *status = IP_ERR_ARG;
        return NULL;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        if (status) *status = IP_ERR_IO;
        return NULL;
    }

    if (fread(magic, 1, 2, fp) != 2) {
        if (status) *status = IP_ERR_FORMAT;
        goto fail;
    }
    if (magic[0] != 'P' || (magic[1] != '5' && magic[1] != '6')) {
        if (status) *status = IP_ERR_FORMAT;
        goto fail;
    }
    channels = (magic[1] == '6') ? 3 : 1;

    if (!read_token(fp, &width) || !read_token(fp, &height) ||
        !read_token(fp, &maxval)) {
        if (status) *status = IP_ERR_FORMAT;
        goto fail;
    }
    if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
        /* 16-bit Netpbm exists but is deliberately out of scope. */
        if (status) *status = IP_ERR_FORMAT;
        goto fail;
    }
    /* NOTE: read_token() already consumed the single whitespace byte that
     * terminates the maxval token, which is exactly the separator the format
     * requires before the raw data.  Consuming another byte here would eat
     * the first pixel. */

    count = (size_t)width * height * channels;

    /* Never trust the header's dimensions: a truncated or malicious file can
     * claim 99999 x 99999 and make us try to allocate tens of gigabytes.
     * Check that the file really holds that many bytes before allocating. */
    {
        long here = ftell(fp);
        long end;
        if (here < 0 || fseek(fp, 0, SEEK_END) != 0 || (end = ftell(fp)) < 0 ||
            (unsigned long)(end - here) < (unsigned long)count ||
            fseek(fp, here, SEEK_SET) != 0) {
            if (status) *status = IP_ERR_FORMAT;
            goto fail;
        }
    }

    raw = (unsigned char *)malloc(count);
    if (!raw) {
        if (status) *status = IP_ERR_ALLOC;
        goto fail;
    }
    if (fread(raw, 1, count, fp) != count) {
        if (status) *status = IP_ERR_FORMAT;
        goto fail;
    }

    image = image_alloc((int)height, (int)width, channels);
    if (!image) {
        if (status) *status = IP_ERR_ALLOC;
        goto fail;
    }
    /* Scale to the canonical [0, 255] working range. */
    if (maxval == 255) {
        for (i = 0; i < count; ++i) {
            image->data[i] = (float)raw[i];
        }
    } else {
        const float scale = 255.0f / (float)maxval;
        for (i = 0; i < count; ++i) {
            image->data[i] = (float)raw[i] * scale;
        }
    }

    free(raw);
    fclose(fp);
    return image;

fail:
    free(raw);
    if (fp) {
        fclose(fp);
    }
    image_free(image);
    return NULL;
}

ip_status image_write_netpbm(const Image *image, const char *path)
{
    FILE *fp;
    unsigned char *raw;
    size_t count, i;

    if (!image || !path) {
        return IP_ERR_ARG;
    }
    if (image->channels != 1 && image->channels != 3) {
        return IP_ERR_FORMAT;
    }

    count = image_elements(image);
    raw = (unsigned char *)malloc(count);
    if (!raw) {
        return IP_ERR_ALLOC;
    }
    /* Round-half-away-from-zero and clamp; never let a float wrap a byte. */
    for (i = 0; i < count; ++i) {
        float v = image->data[i];
        if (!(v > 0.0f)) {            /* also catches NaN */
            raw[i] = 0;
        } else if (v >= 255.0f) {
            raw[i] = 255;
        } else {
            raw[i] = (unsigned char)(v + 0.5f);
        }
    }

    fp = fopen(path, "wb");
    if (!fp) {
        free(raw);
        return IP_ERR_IO;
    }
    fprintf(fp, "P%c\n# written by imgproc " IMGPROC_VERSION "\n%d %d\n255\n",
            image->channels == 3 ? '6' : '5', image->width, image->height);
    if (fwrite(raw, 1, count, fp) != count) {
        free(raw);
        fclose(fp);
        return IP_ERR_IO;
    }

    free(raw);
    if (fclose(fp) != 0) {
        return IP_ERR_IO;
    }
    return IP_OK;
}

/* ------------------------------------------------------------------ */
/* Synthetic test pattern                                             */
/* ------------------------------------------------------------------ */
/*
 * A deterministic pattern combining three structures whose correct filtered
 * output can be reasoned about rather than merely eyeballed:
 *
 *   - a checkerboard : known edge positions; aliases visibly if a downscale
 *                      skips anti-aliasing,
 *   - a disc         : curved edges in every orientation at once,
 *   - a linear ramp  : a signal a Gaussian blur must leave unchanged.
 *
 * Each colour channel carries different structure, so a channel-mixing bug
 * in the greyscale or resize code is immediately visible.
 */
Image *image_make_test_pattern(int size, int channels)
{
    Image *image;
    int y, x;
    const int square = (size >= 16) ? size / 16 : 1;
    const float centre = (float)(size - 1) * 0.5f;
    const float radius = (float)size / 3.0f;

    if (size <= 0 || (channels != 1 && channels != 3)) {
        return NULL;
    }
    image = image_alloc(size, size, channels);
    if (!image) {
        return NULL;
    }

    for (y = 0; y < size; ++y) {
        for (x = 0; x < size; ++x) {
            const float dy = (float)y - centre;
            const float dx = (float)x - centre;
            const float dist = sqrtf(dy * dy + dx * dx);

            /* Anti-aliased disc: one-pixel soft boundary, so the pattern
             * itself contains no aliasing that could be mistaken for a bug. */
            float disc = radius + 0.5f - dist;
            disc = (disc < 0.0f) ? 0.0f : (disc > 1.0f ? 1.0f : disc);

            const float ramp = 255.0f * (float)x / (float)(size > 1 ? size - 1 : 1);
            const float board = (((y / square) + (x / square)) % 2) ? 255.0f : 0.0f;

            float *px = image->data + ((size_t)y * size + x) * channels;
            if (channels == 3) {
                px[0] = ramp;               /* R: horizontal gradient   */
                px[1] = disc * 255.0f;      /* G: disc                  */
                px[2] = board;              /* B: checkerboard          */
            } else {
                /* Blend all three so a single channel still exercises
                 * edges, curves and gradients. */
                px[0] = 0.45f * ramp + 0.40f * disc * 255.0f + 0.15f * board;
            }
        }
    }
    return image;
}
