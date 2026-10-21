# ============================================================================
# MTCS-103P High Performance Computing Laboratory -- Mini Project
# Parallel Image Processing and CNN Primitives with OpenMP
# ============================================================================
#
#   make              build everything (library, CLI, benchmark, tests)
#   make test         build and run the correctness tests
#   make check        test + verify + sanitizers: everything CI runs
#   make bench        run the full scaling study, writing results/*.csv
#   make quickbench   a short benchmark, for checking the setup works
#   make demo         generate the sample output images
#   make gui          launch the graphical front-end (operations + test runner)
#   make verify       cross-check the C output against the NumPy reference
#   make serial       build with OpenMP DISABLED (for a sanity comparison)
#   make asan         run the tests under AddressSanitizer + UBSan
#   make tsan         run the tests under ThreadSanitizer (data-race check)
#   make figures      regenerate every figure from the current results
#   make report       regenerate the figures and render every report PDF
#   make results      the full pipeline: bench + experiments + figures + PDF
#   make install      install bin/imgproc to $(PREFIX)/bin
#   make clean        remove build artefacts
#
# Requires: a C99 compiler with OpenMP.
#   Linux : gcc (any recent version) -- nothing else.
#   macOS : `brew install libomp` and the default Apple clang works, or
#           `brew install gcc` and `make CC=gcc-14`.  Detected automatically.
# Image decoding uses the vendored stb headers (third_party/stb); there are
# no libraries to install.
# ============================================================================

CC       ?= cc
PYTHON   ?= python3
PREFIX   ?= /usr/local

CSTD     := -std=c99
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
OPT      ?= -O3 -funroll-loops
INCLUDE  := -Iinclude -isystem third_party/stb
LDLIBS   := -lm

# -ffast-math is deliberately NOT used.  It would let the compiler reassociate
# floating-point sums, which changes results between thread counts and would
# undermine the correctness comparison against the NumPy reference.

# --- OpenMP flags, per platform --------------------------------------------
# Apple clang understands OpenMP only via `-Xpreprocessor -fopenmp` plus
# Homebrew's libomp; every other compiler takes plain -fopenmp.
UNAME_S  := $(shell uname -s)
CC_IS_APPLE_CLANG := $(shell $(CC) --version 2>/dev/null | grep -c "Apple clang")

ifeq ($(UNAME_S)$(CC_IS_APPLE_CLANG),Darwin1)
  LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
  ifeq ($(LIBOMP_PREFIX),)
    $(warning Apple clang needs libomp for OpenMP:  brew install libomp)
  endif
  # -isystem: Homebrew's omp.h trips -Wpedantic (C23 enum ranges); it is not our code.
  OPENMP_CFLAGS  := -Xpreprocessor -fopenmp -isystem $(LIBOMP_PREFIX)/include
  OPENMP_LDFLAGS := -L$(LIBOMP_PREFIX)/lib -lomp
else
  OPENMP_CFLAGS  := -fopenmp
  OPENMP_LDFLAGS := -fopenmp
endif

CFLAGS   := $(CSTD) $(WARN) $(OPT) $(OPENMP_CFLAGS) $(INCLUDE) $(EXTRA_CFLAGS)
LDFLAGS  := $(OPENMP_LDFLAGS) $(EXTRA_LDFLAGS)

# stb is third-party code: compiled with relaxed warnings so that ours can
# stay strict, and never with -Werror.
STB_CFLAGS := -std=c99 -O2 -w -Ithird_party/stb $(EXTRA_CFLAGS)

SRC_DIR   := src
OBJ_DIR   := build
BIN_DIR   := bin

LIB_SRCS  := $(wildcard $(SRC_DIR)/*.c)
LIB_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS)) \
             $(OBJ_DIR)/stb_impl.o

TARGETS   := $(BIN_DIR)/imgproc $(BIN_DIR)/benchmark $(BIN_DIR)/test_correctness

.PHONY: all test check bench quickbench demo gui verify serial asan tsan \
        figures report results install clean help dirs

all: dirs $(TARGETS)
	@echo ""
	@echo "Build complete.  Binaries in $(BIN_DIR)/"
	@echo "  $(BIN_DIR)/imgproc           command-line driver"
	@echo "  $(BIN_DIR)/benchmark         scaling study"
	@echo "  $(BIN_DIR)/test_correctness  correctness tests"
	@echo ""

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) results data

# --- library objects --------------------------------------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c include/imgproc.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/stb_impl.o: third_party/stb/stb_impl.c third_party/stb/stb_image.h \
                       third_party/stb/stb_image_write.h | dirs
	$(CC) $(STB_CFLAGS) -c $< -o $@

# --- executables ------------------------------------------------------------
$(BIN_DIR)/imgproc: cli/main.c $(LIB_OBJS) | dirs
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/benchmark: bench/benchmark.c $(LIB_OBJS) | dirs
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/test_correctness: tests/test_correctness.c $(LIB_OBJS) | dirs
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# --- convenience targets ----------------------------------------------------
test: $(BIN_DIR)/test_correctness $(BIN_DIR)/imgproc
	@./$(BIN_DIR)/test_correctness
	@sh tests/test_cli.sh

check: test verify asan tsan
	@echo ""
	@echo "make check: all stages passed."

bench: $(BIN_DIR)/benchmark
	@./$(BIN_DIR)/benchmark --full --csv results/scaling.csv
	@echo ""
	@echo "Now generate the plots with:  $(PYTHON) bench/plot_results.py"

quickbench: $(BIN_DIR)/benchmark
	@./$(BIN_DIR)/benchmark --quick

demo: $(BIN_DIR)/imgproc
	@sh scripts/run_demo.sh

gui: all
	@$(PYTHON) gui/app.py

verify: $(BIN_DIR)/imgproc
	@$(PYTHON) tools/verify_against_reference.py

# Build with OpenMP switched off, to confirm the code is correct sequentially
# and to measure the cost of the OpenMP runtime itself.
serial: dirs
	$(CC) $(STB_CFLAGS) -c third_party/stb/stb_impl.c -o $(OBJ_DIR)/stb_impl_serial.o
	$(CC) $(CSTD) $(WARN) $(OPT) $(INCLUDE) cli/main.c $(LIB_SRCS) \
		$(OBJ_DIR)/stb_impl_serial.o -o $(BIN_DIR)/imgproc_serial $(LDLIBS)
	@echo "Built $(BIN_DIR)/imgproc_serial (no OpenMP)"

# --- sanitizers -------------------------------------------------------------
# Each builds into its own directory so it never mixes with the -O3 objects.
#   asan : out-of-bounds reads/writes, use-after-free, undefined behaviour.
#   tsan : data races between OpenMP threads.  GCC's libgomp is not built
#          for TSan, so it hides OpenMP's barriers and floods the log with
#          false positives.  This target therefore uses clang + LLVM libomp
#          with Archer, LLVM's OpenMP-aware TSan tool, which reports only
#          genuine races.  Needs clang, libomp and the clang TSan runtime
#          (Ubuntu: apt install clang libomp-dev libclang-rt-dev).
SAN_SRCS := cli/main.c $(LIB_SRCS)
TSAN_CC  ?= clang
# Archer and libomp live in the LLVM lib directory, two levels above clang's
# resource directory (Ubuntu: /usr/lib/llvm-N/lib; Homebrew: $(brew --prefix llvm)/lib).
LLVM_LIBDIR ?= $(abspath $(shell $(TSAN_CC) -print-resource-dir 2>/dev/null)/../..)
ARCHER      ?= $(firstword $(wildcard $(LLVM_LIBDIR)/libarcher.so $(LLVM_LIBDIR)/libarcher.dylib))
OMP_LIBDIR  := $(LLVM_LIBDIR)

asan: dirs
	@mkdir -p build/asan
	$(CC) $(STB_CFLAGS) -g -fsanitize=address,undefined \
		-c third_party/stb/stb_impl.c -o build/asan/stb_impl.o
	$(CC) $(CSTD) $(WARN) -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined -fno-sanitize-recover=undefined \
		$(OPENMP_CFLAGS) $(INCLUDE) tests/test_correctness.c $(LIB_SRCS) \
		build/asan/stb_impl.o -o build/asan/test_correctness \
		-fsanitize=address,undefined $(LDFLAGS) $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=1 ./build/asan/test_correctness
	@echo "asan: clean"

tsan: dirs
	@mkdir -p build/tsan
	$(TSAN_CC) -std=c99 -O1 -w -g -fsanitize=thread -Ithird_party/stb \
		-c third_party/stb/stb_impl.c -o build/tsan/stb_impl.o
	$(TSAN_CC) $(CSTD) $(WARN) -O1 -g -fsanitize=thread -fopenmp $(INCLUDE) \
		tests/test_correctness.c $(LIB_SRCS) build/tsan/stb_impl.o \
		-o build/tsan/test_correctness $(LDLIBS)
	OMP_TOOL_LIBRARIES=$(ARCHER) LD_LIBRARY_PATH=$(OMP_LIBDIR) \
		TSAN_OPTIONS="suppressions=$(CURDIR)/tests/tsan.supp halt_on_error=1" \
		OMP_NUM_THREADS=4 ./build/tsan/test_correctness
	@echo "tsan: no data races reported"

# --- figures and report -----------------------------------------------------
figures: demo
	@mkdir -p reports/figures
	@$(PYTHON) tools/make_contact_sheets.py
	@$(PYTHON) bench/plot_results.py
	@if [ -f results/chunk_default.csv ]; then \
		$(PYTHON) bench/compare_chunks.py; \
	else \
		echo "  (run 'sh scripts/run_schedule_experiment.sh' for the chunk figure)"; \
	fi

report: figures
	@$(PYTHON) tools/make_individual_reports.py

# Everything needed to refresh the report on a new machine.
results: all
	@$(MAKE) --no-print-directory bench
	@sh scripts/run_schedule_experiment.sh
	@$(MAKE) --no-print-directory report
	@echo ""
	@echo "All results regenerated.  Update section 5 of reports/REPORT.md"
	@echo "with the new numbers, then re-run 'make report'."

install: $(BIN_DIR)/imgproc
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(BIN_DIR)/imgproc $(DESTDIR)$(PREFIX)/bin/imgproc

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned."

help:
	@sed -n '4,30p' Makefile
