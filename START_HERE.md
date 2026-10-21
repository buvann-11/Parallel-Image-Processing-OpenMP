# Start here

**MTCS-103P High Performance Computing Laboratory — Mini Project P1: Image Processing (2)**
B. Sai Buvann (26010303001) · M. Sai Praneeth (26010303004)

CNN padding, convolution and pooling, plus greyscale, normalisation and resize —
written from scratch in C99, parallelised with OpenMP, verified and benchmarked.

## 1. Read the reports

| File | What it is |
|---|---|
| `reports/26010303001_Report.pdf` | Part 1 — padding, convolution, pooling (B. Sai Buvann) |
| `reports/26010303004_Report.pdf` | Part 2 — greyscale, normalisation, resize (M. Sai Praneeth) |

## 2. Run it — one click

| Your machine | Do this |
|---|---|
| macOS / Linux | double-click `RUN_LINUX_MAC.sh`, or run `./RUN_LINUX_MAC.sh` |
| Windows | double-click `RUN_WINDOWS.bat` |
| Any (terminal) | `python3 run_project.py` |

A menu appears:

```
  1. build    Compile the library, CLI, benchmark and tests
  2. verify   Run every verification stage (tests, cross-check, race check)
  3. demo     Apply every operation to the sample photograph
  4. bench    Quick strong-scaling benchmark
  5. gui      Open the graphical front-end
  6. all      build + verify + demo + bench
```

Choose **6 (all)** to see everything, or **5 (gui)** for the graphical interface.
If the build reports a missing compiler, see `REQUIREMENTS.md` (one install command).

## 3. What each part proves

| Command | Shows |
|---|---|
| `verify` / `python3 verification.py` | 41,119 C assertions, 18 image-format checks, every operation bit-identical across thread counts and schedules (no data races), serial vs OpenMP timing, and agreement with an independent NumPy implementation |
| `demo` | every operation applied to a real photograph → `outputs/` |
| `bench` | speedup, efficiency and Karp–Flatt serial fraction per kernel |
| `gui` | **Operations** tab: run any operation interactively. **Test suite** tab: run each test stage with one click (PASS/FAIL badges) |

## 4. Where things are

```
START_HERE.md          this file
README.md              full project overview, with screenshots
REQUIREMENTS.md        what to install, per operating system
RUN_LINUX_MAC.sh       one-click launcher (macOS / Linux)
RUN_WINDOWS.bat        one-click launcher (Windows)
run_project.py         the launcher itself: build / verify / demo / bench / gui
verification.py        every verification stage, one summary table
Makefile               the same tasks for `make` users (make, make test, make check ...)

src/  include/         the library: Part 1 (padding, convolution, pooling)
                       and Part 2 (greyscale, normalisation, resize)
cli/                   command-line driver   (bin/imgproc after building)
gui/                   graphical front-end (Tkinter)
tests/                 C test suite, CLI tests, sanitizer config
bench/                 strong-scaling benchmark and plotting
reports/               report PDFs, Markdown source, figures
results/               raw benchmark data (CSV)
data/                  sample photograph and sample outputs
docs/screenshots/      GUI screenshots
reference/             independent NumPy implementation (verification only)
tools/  scripts/       report, figure and experiment helpers
third_party/stb/       PNG/JPEG reader/writer (public domain)
```
