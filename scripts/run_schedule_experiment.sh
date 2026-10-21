#!/bin/sh
# ============================================================================
# run_schedule_experiment.sh -- the chunk-size experiment.
#
# MTCS-103P High Performance Computing Laboratory -- Mini Project
#
# WHY THIS EXPERIMENT EXISTS
# --------------------------
# `schedule(dynamic)` is often presented as the safe choice for load
# balancing.  With the DEFAULT chunk size it is nothing of the sort.  OpenMP
# defines the default chunk for dynamic and guided as 1, so every single loop
# iteration goes through the runtime's work-dispatch path.  For a loop whose
# body is one multiply-add on one pixel, that dispatch costs far more than
# the work it hands out.
#
# This script measures the same ten kernels twice -- once with the default
# chunk and once with an explicit large chunk -- so the effect is a number in
# the report rather than an assertion.
#
# The result cuts both ways, which is the interesting part:
#
#   * FLAT per-element loops (greyscale, normalisation) collapse with
#     chunk = 1 and recover completely with a large chunk.
#   * ROW-based loops (convolution, pooling, resize) are fine with chunk = 1,
#     because one iteration is already a whole row of real work -- and they
#     get WORSE with a large chunk, because an image only has ~1000 rows, so
#     a chunk of 8192 hands the entire loop to one thread.
#
# There is no single correct chunk size; it has to match the granularity of
# the loop body.
#
# Usage:  sh scripts/run_schedule_experiment.sh
# Output: results/chunk_default.csv, results/chunk_large.csv
# ============================================================================

set -e

BIN=bin/benchmark
SIZE=${SIZE:-1024}
REPEAT=${REPEAT:-3}
LARGE_CHUNK=${LARGE_CHUNK:-8192}

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run 'make' first." >&2
    exit 1
fi

mkdir -p results

echo
echo "Experiment 1/2: default chunk size (1 for dynamic and guided)"
"$BIN" --full --size "$SIZE" --repeat "$REPEAT" --chunk 0 \
       --csv results/chunk_default.csv > /dev/null

echo "Experiment 2/2: explicit chunk size of $LARGE_CHUNK"
"$BIN" --full --size "$SIZE" --repeat "$REPEAT" --chunk "$LARGE_CHUNK" \
       --csv results/chunk_large.csv > /dev/null

echo
echo "Wrote results/chunk_default.csv and results/chunk_large.csv"
echo "Summarise them with:  python3 bench/compare_chunks.py"
echo
