#!/usr/bin/env python3
"""Capture the README/report screenshots of the GUI, reproducibly.

Drives gui/app.py programmatically -- loads the sample photograph, runs an
operation, runs the full test suite -- and grabs the window with ImageMagick
`import`.  Needs an X display; on a headless machine run it under Xvfb:

    xvfb-run -s "-screen 0 980x780x24" python3 tools/take_screenshots.py

Writes docs/screenshots/gui_operations.png and gui_test_suite.png.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

import app  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
SAMPLE = ROOT / "data" / "samples" / "astronaut.png"
W, H = 980, 780


def grab(root: tk.Tk, name: str) -> None:
    root.update()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    subprocess.run(["import", "-window", "root", "-crop", f"{W}x{H}+0+0",
                    "+repage", str(target)], check=True)
    print(f"wrote {target.relative_to(ROOT)}")


def main() -> int:
    root = tk.Tk()
    root.title("imgproc -- parallel image processing (OpenMP)  |  MTCS-103P")
    root.geometry(f"{W}x{H}+0+0")
    app._styles()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    tabs = ttk.Notebook(root)
    tabs.grid(row=0, column=0, sticky="nsew")
    ops = app.Application(tabs)
    suite = app.TestSuite(tabs, ops.max_threads)
    tabs.add(ops, text="  Operations  ")
    tabs.add(suite, text="  Test suite  ")

    # --- 1. Operations tab: Sobel edge filter on the sample photograph ------
    ops.source = ops._convert_to_netpbm(SAMPLE)
    ops._show(ops.canvas_in, ops.source, which="in")
    ops.op_var.set("Convolve")
    ops._on_operation()
    ops.variant_var.set("sobelx")
    ops.threads_var.set(ops.max_threads)

    def after_op() -> None:
        if str(ops.run_button["state"]) == "disabled":
            root.after(200, after_op)
            return
        root.after(400, lambda: (grab(root, "gui_operations.png"),
                                 start_suite()))

    def start_suite() -> None:
        tabs.select(1)
        suite.size_var.set("1024")
        suite.sweep_var.set("conv gauss 5x5")
        suite.run([k for k, *_ in suite.STAGES])
        root.after(500, after_suite)

    def after_suite() -> None:
        if suite._busy:
            root.after(300, after_suite)
            return
        # Scroll the log to the sweep table, the most informative part.
        idx = suite.log.search("=== Thread-count sweep ===", "1.0")
        if idx:
            suite.log.see("end")
            suite.log.yview(idx)
        root.after(400, lambda: (grab(root, "gui_test_suite.png"),
                                 root.destroy()))

    root.after(600, ops.run_operation)
    root.after(900, after_op)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
