#!/bin/sh
# make_gallery.sh -- docs/screenshots/operations_gallery.png: every operation
# applied to the sample photograph (data/samples/astronaut.png, NASA, public
# domain).  Needs ImageMagick (`convert`, `montage`) for the contact sheet only.
set -eu
B=./bin/imgproc
A=data/samples/astronaut.png
G=$(mktemp -d "${TMPDIR:-/tmp}/gallery.XXXXXX")
trap 'rm -rf "$G"' EXIT
q="--quiet --repeat 1"

$B convert   $A $G/01.png $q
$B pad       $A $G/02.png --amount 64 --mode reflect $q
$B conv      $A $G/03.png --kernel gauss $q
$B conv      $A $G/04.png --kernel sobelx $q
$B conv      $A $G/05.png --kernel emboss $q
$B pool      $A $G/06.png --type max --size 4 --stride 4 $q
$B grayscale $A $G/07.png --method luminance $q
$B normalize $A $G/08.png --method equalize $q
$B resize    $A $G/09.png --size 96 96 --method bicubic $q

for i in 01 02 03 04 05 07 08; do
    convert $G/$i.png -resize 300x300! -colorspace sRGB -type TrueColor $G/r$i.png
done
# Downsampled outputs are enlarged with nearest-neighbour so the real
# resolution stays visible instead of being smoothed by the preview.
for i in 06 09; do
    convert $G/$i.png -filter point -resize 300x300! -colorspace sRGB -type TrueColor $G/r$i.png
done

mkdir -p docs/screenshots
montage -font DejaVu-Sans -pointsize 15 \
    -label "Input (PNG, 512x512)"                $G/r01.png \
    -label "Pad: reflect, 64 px"                 $G/r02.png \
    -label "Conv: Gaussian 5x5"                  $G/r03.png \
    -label "Conv: Sobel x"                       $G/r04.png \
    -label "Conv: emboss"                        $G/r05.png \
    -label "Max-pool 4x4, stride 4 (128x128)"    $G/r06.png \
    -label "Greyscale: linear luminance"         $G/r07.png \
    -label "Normalize: histogram equalisation"   $G/r08.png \
    -label "Resize: bicubic 512 -> 96"           $G/r09.png \
    -tile 3x3 -geometry 300x300+10+10 -background white \
    docs/screenshots/operations_gallery.png
echo "wrote docs/screenshots/operations_gallery.png"
