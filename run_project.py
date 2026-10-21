#!/usr/bin/env python3
"""One-command launcher for the MTCS-103P mini project.

Builds the C code with whatever OpenMP compiler is available and runs any
part of the project -- no `make` required, so it works the same on Linux,
macOS and Windows (MinGW-w64).

    python3 run_project.py            interactive menu
    python3 run_project.py all        build + verify + demo + benchmark
    python3 run_project.py build      compile everything into bin/
    python3 run_project.py verify     full verification (see verification.py)
    python3 run_project.py demo       run every operation on the sample photo
    python3 run_project.py bench      quick strong-scaling benchmark
    python3 run_project.py gui        launch the graphical front-end
    python3 run_project.py clean      remove build output

RUN_LINUX_MAC.sh and RUN_WINDOWS.bat simply call this script.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"
BUILD = ROOT / "build"
EXE = ".exe" if os.name == "nt" else ""

WARN = ["-Wall", "-Wextra", "-Wpedantic", "-Wshadow", "-Wstrict-prototypes"]
OPT = ["-O3", "-funroll-loops"]

TARGETS = {
    "imgproc": ROOT / "cli" / "main.c",
    "benchmark": ROOT / "bench" / "benchmark.c",
    "test_correctness": ROOT / "tests" / "test_correctness.c",
}


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.name != "nt" or "WT_SESSION" in os.environ


C = _supports_colour()


def head(text: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n {text}\n{line}" if not C else
          f"\n\033[1;36m{line}\n {text}\n{line}\033[0m")


def ok(text: str) -> None:
    print(f"  \033[32m✔\033[0m {text}" if C else f"  [ok] {text}")


def fail(text: str) -> None:
    print(f"  \033[31m✘\033[0m {text}" if C else f"  [FAIL] {text}")


def run(cmd: list[str], **kw) -> int:
    return subprocess.call(cmd, cwd=ROOT, **kw)


# ---------------------------------------------------------------------------
# compiler detection
# ---------------------------------------------------------------------------
def _brew_prefix(formula: str) -> str | None:
    if not shutil.which("brew"):
        return None
    try:
        out = subprocess.run(["brew", "--prefix", formula], capture_output=True,
                             text=True, timeout=30)
    except Exception:
        return None
    path = out.stdout.strip()
    return path if out.returncode == 0 and Path(path).exists() else None


def _is_apple_clang(cc: str) -> bool:
    try:
        out = subprocess.run([cc, "--version"], capture_output=True, text=True)
        return "Apple clang" in out.stdout
    except Exception:
        return False


def find_compiler() -> tuple[str, list[str], list[str]]:
    """Return (cc, openmp_cflags, openmp_ldflags)."""
    candidates = [os.environ["CC"]] if os.environ.get("CC") else []
    if platform.system() == "Darwin":
        # Homebrew GCC first (libgomp), then Apple clang + libomp.
        for v in range(16, 9, -1):
            candidates.append(f"gcc-{v}")
        candidates += ["clang", "cc"]
    else:
        candidates += ["gcc", "cc", "clang"]

    for cc in candidates:
        if not shutil.which(cc):
            continue
        if _is_apple_clang(cc):
            omp = _brew_prefix("libomp")
            if not omp:
                continue
            return (cc, ["-Xpreprocessor", "-fopenmp", "-isystem", f"{omp}/include"],
                    [f"-L{omp}/lib", "-lomp"])
        return cc, ["-fopenmp"], ["-fopenmp"]

    print("\nNo C compiler with OpenMP was found.  See REQUIREMENTS.md:")
    print("  Linux  : sudo apt install build-essential")
    print("  macOS  : brew install libomp      (or: brew install gcc)")
    print("  Windows: install MSYS2, then  pacman -S mingw-w64-ucrt-x86_64-gcc")
    sys.exit(1)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def build(quiet: bool = False) -> bool:
    head("Building (C99 + OpenMP)")
    cc, omp_c, omp_l = find_compiler()
    print(f"  compiler : {cc}  ({platform.system()} {platform.machine()})")
    BIN.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    inc = ["-Iinclude", "-isystem", "third_party/stb"]

    objects = []
    stb = BUILD / "stb_impl.o"
    if run([cc, "-std=c99", "-O2", "-w", "-Ithird_party/stb", "-c",
            "third_party/stb/stb_impl.c", "-o", str(stb)]) != 0:
        fail("third_party/stb")
        return False
    objects.append(str(stb))
    for src in sorted((ROOT / "src").glob("*.c")):
        obj = BUILD / (src.stem + ".o")
        if run([cc, "-std=c99", *WARN, *OPT, *omp_c, *inc, "-c",
                str(src.relative_to(ROOT)), "-o", str(obj)]) != 0:
            fail(f"compile {src.name}")
            return False
        objects.append(str(obj))
    ok(f"library: {len(objects) - 1} source files + stb")

    for name, main in TARGETS.items():
        out = BIN / (name + EXE)
        if run([cc, "-std=c99", *WARN, *OPT, *omp_c, *inc,
                str(main.relative_to(ROOT)), *objects, "-o", str(out),
                *omp_l, "-lm"]) != 0:
            fail(f"link {name}")
            return False
        ok(f"bin/{name}{EXE}")
    return True


def ensure_built() -> bool:
    if all((BIN / (n + EXE)).exists() for n in TARGETS):
        return True
    return build()


def verify() -> bool:
    if not ensure_built():
        return False
    return run([sys.executable, str(ROOT / "verification.py")]) == 0


def demo() -> bool:
    if not ensure_built():
        return False
    head("Demo: every operation on the sample photograph")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    src = ROOT / "data" / "samples" / "astronaut.png"
    imgproc = str(BIN / ("imgproc" + EXE))
    jobs = [
        ("pad_reflect.png", ["pad", "--amount", "64", "--mode", "reflect"]),
        ("conv_gauss.png", ["conv", "--kernel", "gauss"]),
        ("conv_sobelx.png", ["conv", "--kernel", "sobelx"]),
        ("conv_emboss.png", ["conv", "--kernel", "emboss"]),
        ("pool_max_4x4.png", ["pool", "--type", "max", "--size", "4", "--stride", "4"]),
        ("pool_avg_4x4.png", ["pool", "--type", "avg", "--size", "4", "--stride", "4"]),
        ("grey_bt709.png", ["grayscale", "--method", "bt709"]),
        ("grey_luminance.png", ["grayscale", "--method", "luminance"]),
        ("norm_equalize.png", ["normalize", "--method", "equalize"]),
        ("norm_minmax.png", ["normalize", "--method", "minmax"]),
        ("resize_bicubic_256.png", ["resize", "--size", "256", "256", "--method", "bicubic"]),
    ]
    print(f"  {'output':<26}{'serial ms':>11}{'openmp ms':>11}{'speedup':>9}")
    import json
    good = True
    for name, args in jobs:
        proc = subprocess.run([imgproc, args[0], str(src), str(out_dir / name),
                               *args[1:], "--json", "--quiet"],
                              cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"{name}: {proc.stderr.strip()}")
            good = False
            continue
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        print(f"  {name:<26}{info['serial_ms']:>11.2f}{info['omp_ms']:>11.2f}"
              f"{info['speedup']:>8.2f}x")
    print(f"\n  Images written to {out_dir.relative_to(ROOT)}/")
    return good


def bench() -> bool:
    if not ensure_built():
        return False
    head("Quick strong-scaling benchmark")
    return run([str(BIN / ("benchmark" + EXE)), "--quick"]) == 0


def gui() -> bool:
    if not ensure_built():
        return False
    try:
        import tkinter  # noqa: F401
    except ImportError:
        fail("Tkinter is not installed for this Python (see REQUIREMENTS.md)")
        return False
    return run([sys.executable, str(ROOT / "gui" / "app.py")]) == 0


def clean() -> bool:
    for d in (BIN, BUILD):
        shutil.rmtree(d, ignore_errors=True)
    ok("removed bin/ and build/")
    return True


def all_steps() -> bool:
    results = [("build", build()), ("verify", verify()), ("demo", demo()),
               ("bench", bench())]
    head("Summary")
    for name, good in results:
        (ok if good else fail)(name)
    return all(g for _, g in results)


COMMANDS = {
    "build": (build, "Compile the library, CLI, benchmark and tests"),
    "verify": (verify, "Run every verification stage (tests, cross-check, race check)"),
    "demo": (demo, "Apply every operation to the sample photograph"),
    "bench": (bench, "Quick strong-scaling benchmark"),
    "gui": (gui, "Open the graphical front-end"),
    "all": (all_steps, "build + verify + demo + bench"),
    "clean": (clean, "Remove build output"),
}


def menu() -> int:
    head("MTCS-103P Mini Project -- Parallel Image Processing with OpenMP")
    names = list(COMMANDS)
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name:<7} {COMMANDS[name][1]}")
    print("  q. quit")
    choice = input("\nChoose: ").strip().lower()
    if choice in ("q", ""):
        return 0
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        choice = names[int(choice) - 1]
    if choice not in COMMANDS:
        print("Unknown choice.")
        return 2
    return 0 if COMMANDS[choice][0]() else 1


def main() -> int:
    os.chdir(ROOT)
    if len(sys.argv) < 2:
        return menu()
    cmd = sys.argv[1].lower()
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if cmd not in COMMANDS:
        print(f"unknown command '{cmd}'\n{__doc__}")
        return 2
    return 0 if COMMANDS[cmd][0]() else 1


if __name__ == "__main__":
    sys.exit(main())
