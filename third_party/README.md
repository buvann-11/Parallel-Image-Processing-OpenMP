# Third-party code

| Component | Version | Upstream | License | Used for |
|---|---|---|---|---|
| `stb/stb_image.h` | v2.30 | <https://github.com/nothings/stb> | Public domain / MIT | Decoding PNG, JPEG, BMP, TGA |
| `stb/stb_image_write.h` | v1.16 | <https://github.com/nothings/stb> | Public domain / MIT | Encoding PNG, JPEG, BMP, TGA |

The headers are vendored **unmodified**. They are compiled in exactly one
translation unit, `stb/stb_impl.c`, with warnings relaxed (see the Makefile), so
the project's own sources stay at `-Wall -Wextra -Wpedantic`. Only the PNG,
JPEG, BMP and TGA decoders are enabled.

No third-party code is used in any kernel: padding, convolution, pooling,
greyscale, normalisation and resize are implemented from scratch in `src/`.
