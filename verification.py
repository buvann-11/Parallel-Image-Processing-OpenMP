#!/usr/bin/env python3
"""Run every verification stage of the project and print one summary.

    python3 verification.py

Portable (Linux, macOS, Windows): needs only the built binaries in bin/
(`python3 run_project.py build` or `make`).  Stages:

  1. C correctness suite      27 tests, 41,119 assertions (bin/test_correctness)
  2. CLI & image formats      lossless PNG/BMP/TGA round trips, JPEG decode,
                              PNG input == PPM input for every operation
  3. Thread & schedule check  every operation bit-identical at 1, 2, 4 and
                              max threads under static, dynamic and guided
  4. Serial vs OpenMP         speedup of each operation on a 1024x1024 image
  5. NumPy cross-validation   C output vs an independent NumPy implementation
                              (skipped, not failed, if NumPy is missing)

Exit status is 0 only if every stage that ran passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ".exe" if os.name == "nt" else ""
IMGPROC = str(ROOT / "bin" / ("imgproc" + EXE))
TESTS = str(ROOT / "bin" / ("test_correctness" + EXE))

OPS = {
    "pad reflect":        ["pad", "--amount", "40", "--mode", "reflect"],
    "conv gauss 5x5":     ["conv", "--kernel", "gauss"],
    "conv sobelx 3x3":    ["conv", "--kernel", "sobelx"],
    "pool max 2x2":       ["pool", "--type", "max", "--size", "2", "--stride", "2"],
    "pool avg 3x3/s2":    ["pool", "--type", "avg", "--size", "3", "--stride", "2"],
    "greyscale bt709":    ["grayscale", "--method", "bt709"],
    "greyscale luminance": ["grayscale", "--method", "luminance"],
    "normalize minmax":   ["normalize", "--method", "minmax"],
    "normalize zscore":   ["normalize", "--method", "zscore"],
    "normalize equalize": ["normalize", "--method", "equalize"],
    "resize bilinear":    ["resize", "--size", "100", "160", "--method", "bilinear"],
    "resize bicubic":     ["resize", "--size", "97", "131", "--method", "bicubic"],
}


def cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([IMGPROC, *args], cwd=ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"imgproc {' '.join(args)}: {proc.stderr.strip()}")
    return proc


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def stage(self, title: str) -> None:
        print(f"\n--- {title} " + "-" * max(0, 58 - len(title)))

    def check(self, name: str, good: bool) -> bool:
        print(f"  [{' ok ' if good else 'FAIL'}] {name}")
        return good

    def add(self, stage: str, status: str, detail: str) -> None:
        self.rows.append((stage, status, detail))


def stage_c_suite(r: Report) -> None:
    r.stage("1. C correctness suite")
    proc = subprocess.run([TESTS], cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in proc.stdout.splitlines() if "ASSERTIONS" in l.upper()
            or "PASSED" in l or "FAIL" in l]
    for line in tail[-3:]:
        print("  " + line.strip())
    good = proc.returncode == 0
    r.add("C correctness suite", "PASS" if good else "FAIL",
          tail[-1].strip() if tail else "")


def stage_formats(r: Report, tmp: Path) -> None:
    r.stage("2. CLI & image formats")
    passed = total = 0
    cli("gen", str(tmp / "in.ppm"), "--size", "257", "--quiet")
    cli("gen", str(tmp / "g.pgm"), "--size", "129", "--gray", "--quiet")
    for ext in ("png", "bmp", "tga"):
        cli("convert", str(tmp / "in.ppm"), str(tmp / f"rt.{ext}"), "--quiet")
        cli("convert", str(tmp / f"rt.{ext}"), str(tmp / f"rt_{ext}.ppm"), "--quiet")
        total += 1
        passed += r.check(f"ppm -> {ext} -> ppm is lossless",
                          digest(tmp / "in.ppm") == digest(tmp / f"rt_{ext}.ppm"))
    cli("convert", str(tmp / "g.pgm"), str(tmp / "g.png"), "--quiet")
    cli("convert", str(tmp / "g.png"), str(tmp / "g2.pgm"), "--quiet")
    total += 1
    passed += r.check("greyscale pgm -> png -> pgm stays 1-channel, lossless",
                      digest(tmp / "g.pgm") == digest(tmp / "g2.pgm"))
    cli("convert", str(tmp / "in.ppm"), str(tmp / "in.jpg"), "--quiet")
    info = json.loads(cli("convert", str(tmp / "in.jpg"), str(tmp / "j.ppm"),
                          "--json").stdout.strip().splitlines()[-1])
    total += 1
    passed += r.check("JPEG decodes to 257 x 257 x 3",
                      (info["out_height"], info["out_width"], info["out_channels"])
                      == (257, 257, 3))
    total += 1
    passed += r.check("unknown file extension is rejected",
                      cli("convert", str(tmp / "in.ppm"), str(tmp / "x.xyz"),
                          "--quiet", check=False).returncode != 0)
    cli("convert", str(tmp / "in.ppm"), str(tmp / "in.png"), "--quiet")
    for name, args in OPS.items():
        cli(args[0], str(tmp / "in.ppm"), str(tmp / "a.ppm"), *args[1:],
            "--repeat", "1", "--quiet")
        cli(args[0], str(tmp / "in.png"), str(tmp / "b.ppm"), *args[1:],
            "--repeat", "1", "--quiet")
        total += 1
        passed += r.check(f"{name}: PNG input == PPM input",
                          digest(tmp / "a.ppm") == digest(tmp / "b.ppm"))
    r.add("CLI & image formats", "PASS" if passed == total else "FAIL",
          f"{passed}/{total} checks")


def stage_threads(r: Report, tmp: Path, max_threads: int) -> None:
    r.stage("3. Thread-count & schedule invariance (race check)")
    threads = sorted({1, 2, 4, max_threads})
    passed = 0
    for name, args in OPS.items():
        hashes = set()
        for t in threads:
            for s in ("static", "dynamic", "guided"):
                out = tmp / "t.ppm"
                cli(args[0], str(tmp / "in.ppm"), str(out), *args[1:],
                    "--threads", str(t), "--schedule", s, "--repeat", "1", "--quiet")
                hashes.add(digest(out))
        passed += r.check(f"{name}: identical across {len(threads)} thread counts x 3 schedules",
                          len(hashes) == 1)
    r.add("Thread & schedule invariance", "PASS" if passed == len(OPS) else "FAIL",
          f"{passed}/{len(OPS)} operations, threads {threads}")


def stage_speedup(r: Report, tmp: Path, max_threads: int) -> None:
    r.stage(f"4. Serial vs OpenMP, 1024x1024x3, {max_threads} threads")
    cli("gen", str(tmp / "big.ppm"), "--size", "1024", "--quiet")
    print(f"  {'operation':<22}{'serial ms':>11}{'openmp ms':>11}{'speedup':>9}")
    best = ("", 0.0)
    for name, args in OPS.items():
        info = json.loads(cli(args[0], str(tmp / "big.ppm"), str(tmp / "o.ppm"),
                              *args[1:], "--repeat", "3", "--json", "--quiet")
                          .stdout.strip().splitlines()[-1])
        print(f"  {name:<22}{info['serial_ms']:>11.2f}{info['omp_ms']:>11.2f}"
              f"{info['speedup']:>8.2f}x")
        if info["speedup"] > best[1]:
            best = (name, info["speedup"])
    r.add("Serial vs OpenMP timing", "INFO",
          f"best {best[1]:.2f}x ({best[0]}) on {max_threads} threads")


def stage_numpy(r: Report) -> None:
    r.stage("5. Cross-validation against the NumPy reference")
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("  NumPy not installed -- skipped (pip install numpy)")
        r.add("NumPy cross-validation", "SKIP", "NumPy not installed")
        return
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "verify_against_reference.py"),
                           "--binary", IMGPROC], cwd=ROOT, capture_output=True, text=True)
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    summary = next((l for l in reversed(lines) if "checks" in l), lines[-1] if lines else "")
    summary = summary.split(" -- ")[0]
    print("  " + summary)
    r.add("NumPy cross-validation", "PASS" if proc.returncode == 0 else "FAIL", summary)


def main() -> int:
    if not Path(IMGPROC).exists() or not Path(TESTS).exists():
        print("Binaries not found -- build first:  python3 run_project.py build")
        return 1
    max_threads = int(json.loads(cli("info", "--json").stdout.strip().splitlines()[-1])
                      ["max_threads"])
    print("=" * 64)
    print(" MTCS-103P Mini Project -- full verification")
    print(f" {max_threads} OpenMP threads available")
    print("=" * 64)

    r = Report()
    with tempfile.TemporaryDirectory(prefix="imgproc_verify_") as d:
        tmp = Path(d)
        for stage in (lambda: stage_c_suite(r),
                      lambda: stage_formats(r, tmp),
                      lambda: stage_threads(r, tmp, max_threads),
                      lambda: stage_speedup(r, tmp, max_threads),
                      lambda: stage_numpy(r)):
            try:
                stage()
            except Exception as exc:  # report and carry on
                print(f"  error: {exc}")
                r.add("(stage error)", "FAIL", str(exc)[:60])

    print("\n" + "=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    for stage, status, detail in r.rows:
        print(f"  {status:<5} {stage:<30} {detail}")
    failed = any(s == "FAIL" for _, s, _ in r.rows)
    print("=" * 64)
    print(" RESULT: " + ("FAILED" if failed else "ALL STAGES PASSED"))
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
