# Requirements

Only a **C compiler with OpenMP** is needed to build and run the project.
Python 3 runs the one-click launcher and the GUI. Everything else is optional.

| Needed for | Requirement |
|---|---|
| Building and running (required) | C99 compiler with OpenMP: GCC ≥ 4.9, or Clang with libomp |
| Launcher, verification, GUI | Python 3.8+ (Tkinter for the GUI — included with python.org installers) |
| NumPy cross-validation (optional) | `pip install numpy` |
| Regenerating figures and the PDF (optional) | `pip install matplotlib markdown`, plus `wkhtmltopdf` or `pandoc` |
| Race detection `make tsan` (optional, Linux) | `clang`, `libomp-dev`, `libclang-rt-dev` |

No image library is needed: PNG/JPEG support is built in (`third_party/stb`).

## Install, per operating system

**Ubuntu / Debian**
```bash
sudo apt install build-essential python3 python3-tk
pip install numpy            # optional
```

**macOS**
```bash
xcode-select --install       # Apple's compiler, if not already installed
brew install libomp          # OpenMP for Apple clang
pip3 install numpy           # optional
```
(Alternatively `brew install gcc` — the launcher prefers Homebrew GCC when present.)

**Windows**
1. Install [MSYS2](https://www.msys2.org), open the *MSYS2 UCRT64* terminal and run
   `pacman -S mingw-w64-ucrt-x86_64-gcc`.
2. Add `C:\msys64\ucrt64\bin` to your `PATH`.
3. Install Python 3 from python.org (tick "Add to PATH").
4. Double-click `RUN_WINDOWS.bat`.

## Check your setup

```bash
python3 run_project.py build     # compiles everything into bin/
python3 verification.py          # should end with: RESULT: ALL STAGES PASSED
```
