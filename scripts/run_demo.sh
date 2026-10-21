#!/bin/sh
# ============================================================================
# run_demo.sh -- generate one sample output for every operation.
#
# MTCS-103P High Performance Computing Laboratory -- Mini Project
#
# Produces data/demo/*.ppm so the results can be inspected without running
# the GUI.  Timing for each operation is printed as it goes.
#
# Usage:  make demo        or        sh scripts/run_demo.sh
# ============================================================================

set -e

BIN=bin/imgproc
OUT=data/demo
SIZE=${SIZE:-512}

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run 'make' first." >&2
    exit 1
fi

mkdir -p "$OUT"

echo
echo "Generating demonstration outputs (${SIZE}x${SIZE}) into $OUT/ ..."
echo

"$BIN" gen "$OUT/00_input.ppm" --size "$SIZE" --quiet
echo "  input pattern -> $OUT/00_input.ppm"

run() {
    label=$1
    shift
    printf '  %-28s' "$label"
    "$BIN" "$@" --quiet --json | \
        sed -n 's/.*"serial_ms":\([0-9.]*\).*"omp_ms":\([0-9.]*\).*"speedup":\([0-9.]*\).*/serial \1 ms   omp \2 ms   speedup \3x/p'
}

echo
echo "Part 1 -- CNN primitives"
run "pad, zeros"        pad  "$OUT/00_input.ppm" "$OUT/10_pad_zeros.ppm"      --amount 64 --mode zeros
run "pad, reflect"      pad  "$OUT/00_input.ppm" "$OUT/11_pad_reflect.ppm"    --amount 64 --mode reflect
run "pad, replicate"    pad  "$OUT/00_input.ppm" "$OUT/12_pad_replicate.ppm"  --amount 64 --mode replicate
run "pad, circular"     pad  "$OUT/00_input.ppm" "$OUT/13_pad_circular.ppm"   --amount 64 --mode circular
run "max pool 4x4"      pool "$OUT/00_input.ppm" "$OUT/14_pool_max.ppm"       --type max --size 4 --stride 4
run "avg pool 4x4"      pool "$OUT/00_input.ppm" "$OUT/15_pool_avg.ppm"       --type avg --size 4 --stride 4
run "convolve, gauss"   conv "$OUT/00_input.ppm" "$OUT/16_conv_gauss.ppm"     --kernel gauss
run "convolve, sobel x" conv "$OUT/00_input.ppm" "$OUT/17_conv_sobelx.ppm"    --kernel sobelx
run "convolve, laplace" conv "$OUT/00_input.ppm" "$OUT/18_conv_laplacian.ppm" --kernel laplacian
run "convolve, sharpen" conv "$OUT/00_input.ppm" "$OUT/19_conv_sharpen.ppm"   --kernel sharpen

echo
echo "Part 2 -- Image processing"
run "greyscale, BT.601"   grayscale "$OUT/00_input.ppm" "$OUT/20_gray_bt601.pgm"  --method bt601
run "greyscale, BT.709"   grayscale "$OUT/00_input.ppm" "$OUT/21_gray_bt709.pgm"  --method bt709
run "greyscale, average"  grayscale "$OUT/00_input.ppm" "$OUT/22_gray_average.pgm" --method average
run "greyscale, gamma"    grayscale "$OUT/00_input.ppm" "$OUT/23_gray_lum.pgm"    --method luminance
run "normalize, min-max"  normalize "$OUT/00_input.ppm" "$OUT/30_norm_minmax.ppm" --method minmax
run "normalize, z-score"  normalize "$OUT/00_input.ppm" "$OUT/31_norm_zscore.ppm" --method zscore
run "normalize, equalise" normalize "$OUT/00_input.ppm" "$OUT/32_norm_equalize.ppm" --method equalize
run "resize, nearest"     resize "$OUT/00_input.ppm" "$OUT/40_resize_nearest.ppm"  --size 128 128 --method nearest
run "resize, bilinear"    resize "$OUT/00_input.ppm" "$OUT/41_resize_bilinear.ppm" --size 128 128 --method bilinear
run "resize, bicubic"     resize "$OUT/00_input.ppm" "$OUT/42_resize_bicubic.ppm"  --size 128 128 --method bicubic
run "resize, no anti-alias" resize "$OUT/00_input.ppm" "$OUT/43_resize_noaa.ppm"   --size 128 128 --method bilinear --no-antialias

echo
echo "Done.  $(ls -1 "$OUT" | wc -l) files in $OUT/"
echo "View them with any image viewer, or through the GUI (make gui)."
echo
