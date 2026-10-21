/*
 * stb_impl.c -- the single translation unit that compiles the stb
 *               single-header libraries.
 *
 * stb_image.h and stb_image_write.h are by Sean Barrett and contributors,
 * https://github.com/nothings/stb, released into the public domain (or MIT,
 * at the user's choice) -- see third_party/stb/LICENSE.  They are vendored
 * unmodified.  This file is compiled with relaxed warnings (see Makefile) so
 * that the project's own code can stay at -Wall -Wextra -Wpedantic.
 *
 * Only the decoders the project needs are enabled, which keeps the binary
 * small and the attack surface of the image parser narrow.
 */

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_FAILURE_USERMSG

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
