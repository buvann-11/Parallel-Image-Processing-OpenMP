#!/bin/sh
# test_cli.sh -- end-to-end tests of bin/imgproc and the image-format layer.
#
# MTCS-103P High Performance Computing Laboratory -- Mini Project
#
# The C test suite exercises the kernels directly; this script checks the
# shipped binary the way a user drives it:
#   * PNG/BMP/TGA round-trips are lossless and agree byte-for-byte with PPM,
#   * JPEG decodes to the right shape (it is lossy, so only that),
#   * hostile input (oversized headers, corrupt files, impossible window or
#     output sizes) is rejected cleanly instead of crashing,
#   * every operation gives identical output from a PPM and a PNG input,
#   * the CLI output is identical at 1, 2 and 4 threads and under all three
#     schedules -- the end-to-end version of the race check.
set -eu

BIN=${BIN:-./bin/imgproc}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/imgproc_cli.XXXXXX")
trap 'rm -rf "$TMP"' EXIT INT TERM

pass=0
fail=0
ok()  { pass=$((pass + 1)); printf '  [ ok ] %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  [FAIL] %s\n' "$1"; }
same() { if cmp -s "$2" "$3"; then ok "$1"; else bad "$1"; fi; }

echo "------------------------------------------------------------"
echo " CLI and image-format tests"
echo "------------------------------------------------------------"

"$BIN" gen "$TMP/in.ppm" --size 257 --quiet
"$BIN" gen "$TMP/gray.pgm" --size 129 --gray --quiet

# ---- lossless round trips -------------------------------------------------
for ext in png bmp tga; do
    "$BIN" convert "$TMP/in.ppm" "$TMP/rt.$ext" --quiet
    "$BIN" convert "$TMP/rt.$ext" "$TMP/rt_$ext.ppm" --quiet
    same "RGB  ppm -> $ext -> ppm is lossless" "$TMP/in.ppm" "$TMP/rt_$ext.ppm"
done
"$BIN" convert "$TMP/gray.pgm" "$TMP/gray.png" --quiet
"$BIN" convert "$TMP/gray.png" "$TMP/gray_rt.pgm" --quiet
same "grey pgm -> png -> pgm stays 1-channel and lossless" \
     "$TMP/gray.pgm" "$TMP/gray_rt.pgm"

# ---- JPEG (lossy): shape only --------------------------------------------
"$BIN" convert "$TMP/in.ppm" "$TMP/in.jpg" --quiet
shape=$("$BIN" convert "$TMP/in.jpg" "$TMP/jpg.ppm" --json)
case "$shape" in
    *'"out_height":257,"out_width":257,"out_channels":3'*) ok "JPEG decodes to 257 x 257 x 3" ;;
    *) bad "JPEG decodes to 257 x 257 x 3 (got: $shape)" ;;
esac

# ---- unsupported extension is rejected cleanly ----------------------------
if "$BIN" convert "$TMP/in.ppm" "$TMP/out.xyz" --quiet 2>/dev/null; then
    bad "unknown output extension is rejected"
else
    ok "unknown output extension is rejected"
fi

# ---- hostile and invalid input is rejected without crashing --------------
printf 'P6\n99999 99999\n255\nxx' > "$TMP/huge.ppm"
if "$BIN" conv "$TMP/huge.ppm" "$TMP/o.ppm" --quiet 2>/dev/null; then
    bad "header claiming 99999x99999 on a tiny file is rejected"
else
    ok "header claiming 99999x99999 on a tiny file is rejected"
fi
printf 'not an image' > "$TMP/junk.png"
if "$BIN" conv "$TMP/junk.png" "$TMP/o.ppm" --quiet 2>/dev/null; then
    bad "corrupt PNG is rejected"
else
    ok "corrupt PNG is rejected"
fi
"$BIN" gen "$TMP/tiny.ppm" --size 5 --quiet
rc=0; "$BIN" pool "$TMP/tiny.ppm" "$TMP/o.ppm" --size 8 --stride 8 --quiet 2>/dev/null || rc=$?
if [ "$rc" -eq 2 ]; then ok "pooling window larger than the image is a usage error"
else bad "pooling window larger than the image is a usage error (rc=$rc)"; fi
rc=0; "$BIN" resize "$TMP/tiny.ppm" "$TMP/o.ppm" --size 0 4 --quiet 2>/dev/null || rc=$?
if [ "$rc" -eq 2 ]; then ok "resize to zero size is a usage error"
else bad "resize to zero size is a usage error (rc=$rc)"; fi

# ---- every operation: PPM input == PNG input ------------------------------
"$BIN" convert "$TMP/in.ppm" "$TMP/in.png" --quiet
run_op() {  # name, then the op arguments with IN/OUT placeholders
    name=$1; shift
    op=$1; shift
    "$BIN" "$op" "$TMP/in.ppm" "$TMP/a.ppm" "$@" --repeat 1 --quiet
    "$BIN" "$op" "$TMP/in.png" "$TMP/b.ppm" "$@" --repeat 1 --quiet
    same "$name: PNG input gives the same result as PPM input" \
         "$TMP/a.ppm" "$TMP/b.ppm"
}
run_op grayscale  grayscale --method bt709
run_op normalize  normalize --method equalize
run_op resize     resize    --size 100 160 --method bicubic
run_op pad        pad       --amount 40 --mode reflect
run_op pool       pool      --type max --size 2 --stride 2
run_op conv       conv      --kernel gauss

# ---- thread count and schedule must not change the output -----------------
for op in "conv --kernel sobelx" "normalize --method equalize" \
          "resize --size 97 131 --method bicubic" "pool --type avg --size 3 --stride 2"; do
    set -- $op
    cmd=$1; shift
    "$BIN" "$cmd" "$TMP/in.ppm" "$TMP/t1.ppm" "$@" --threads 1 --repeat 1 --quiet
    all_same=1
    for t in 2 4; do
        for s in static dynamic guided; do
            "$BIN" "$cmd" "$TMP/in.ppm" "$TMP/tn.ppm" "$@" --threads $t \
                --schedule $s --repeat 1 --quiet
            cmp -s "$TMP/t1.ppm" "$TMP/tn.ppm" || all_same=0
        done
    done
    if [ $all_same -eq 1 ]; then
        ok "$cmd: identical at 1/2/4 threads x static/dynamic/guided"
    else
        bad "$cmd: output depends on thread count or schedule"
    fi
done

echo "------------------------------------------------------------"
if [ $fail -eq 0 ]; then
    echo " ALL CLI TESTS PASSED  ($pass checks)"
    echo "------------------------------------------------------------"
    exit 0
fi
echo " $fail of $((pass + fail)) CLI checks FAILED"
echo "------------------------------------------------------------"
exit 1
