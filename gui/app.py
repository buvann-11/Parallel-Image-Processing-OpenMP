#!/usr/bin/env python3
"""Graphical front-end for the parallel image-processing library.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Two tabs:

* **Operations** -- load an image (PPM/PGM/PNG/JPEG/BMP/TGA), pick any of the
  operations, choose the thread count and the OpenMP schedule, and run it.
  The before/after images appear side by side together with the measured
  serial time, parallel time, speedup and efficiency.

* **Test suite** -- one click runs each of the project's test stages against
  the submitted code: the C correctness suite, the CLI/format tests, the
  NumPy cross-validation, a thread-count sweep (speedup table + bit-identical
  output check) and the quick benchmark.  Output streams live into the
  window and each stage gets a PASS/FAIL badge.

Design notes
------------
* The GUI does NOT reimplement anything.  It shells out to ``bin/imgproc``
  and parses the JSON that binary prints.  All the numerical work, and all
  the OpenMP, stays in the C code where it belongs; this is a thin driver.

* It depends only on the Python standard library.  Tk's ``PhotoImage`` reads
  PGM/PPM natively, which is exactly the format the C code writes; PNG/JPEG
  inputs are converted by ``imgproc convert`` (stb_image in the C code), so
  no Python imaging library is needed.

Run with:  make gui        or       python3 gui/app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bin" / "imgproc"

# operation -> (CLI command, [(label, flag, [choices])])
OPERATIONS: dict[str, dict] = {
    "Greyscale": {
        "cmd": "grayscale",
        "ext": ".pgm",
        "choice": ("--method", ["bt601", "bt709", "average", "luminance"]),
        "blurb": "RGB to luma using the broadcast standards. Elementwise: "
                 "memory bound, except 'luminance' which does sRGB gamma.",
    },
    "Normalize": {
        "cmd": "normalize",
        "ext": ".ppm",
        "choice": ("--method", ["minmax", "zscore", "equalize"]),
        "blurb": "Global statistic, then a point transform. Uses OpenMP "
                 "reductions and per-thread private histograms.",
    },
    "Resize": {
        "cmd": "resize",
        "ext": ".ppm",
        "choice": ("--method", ["nearest", "bilinear", "bicubic"]),
        "blurb": "Separable resampling with anti-aliasing. Two parallel "
                 "passes with a barrier between them.",
        "size": True,
    },
    "Pad": {
        "cmd": "pad",
        "ext": ".ppm",
        "choice": ("--mode", ["zeros", "reflect", "replicate", "circular"]),
        "blurb": "CNN border handling. A pure gather: the most "
                 "bandwidth-bound kernel here.",
        "amount": True,
    },
    "Pool": {
        "cmd": "pool",
        "ext": ".ppm",
        "choice": ("--type", ["max", "avg"]),
        "blurb": "CNN downsampling. Windowed reduction, parallel over "
                 "output rows.",
        "window": True,
    },
    "Convolve": {
        "cmd": "conv",
        "ext": ".ppm",
        "choice": ("--kernel", ["box", "gauss", "sobelx", "sobely",
                                "laplacian", "sharpen", "emboss"]),
        "blurb": "Depthwise filtering. High arithmetic intensity, so this "
                 "scales best of all the kernels.",
    },
}


class Application(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=10)

        self.tmpdir = Path(tempfile.mkdtemp(prefix="imgproc_gui_"))
        self.source: Path | None = None
        self.result_path: Path | None = None
        self._photo_in: tk.PhotoImage | None = None
        self._photo_out: tk.PhotoImage | None = None
        self._queue: queue.Queue = queue.Queue()

        self.max_threads = self._detect_threads()
        self._build_widgets()
        self._generate_sample()
        self.after(100, self._drain_queue)

    # -- setup ----------------------------------------------------------
    def _detect_threads(self) -> int:
        try:
            out = subprocess.run([str(BINARY), "info", "--json"],
                                 capture_output=True, text=True, timeout=30)
            return int(json.loads(out.stdout)["max_threads"])
        except Exception:
            return 4

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # --- controls ---------------------------------------------------
        controls = ttk.LabelFrame(self, text="Operation", padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        for c in range(8):
            controls.columnconfigure(c, weight=0)

        ttk.Button(controls, text="Open image...",
                   command=self.open_image).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(controls, text="Sample image",
                   command=self._generate_sample).grid(row=0, column=1, padx=4)

        ttk.Label(controls, text="Operation:").grid(row=0, column=2, padx=(12, 2))
        self.op_var = tk.StringVar(value="Convolve")
        op_box = ttk.Combobox(controls, textvariable=self.op_var, width=12,
                              state="readonly", values=list(OPERATIONS))
        op_box.grid(row=0, column=3)
        op_box.bind("<<ComboboxSelected>>", lambda _e: self._on_operation())

        ttk.Label(controls, text="Variant:").grid(row=0, column=4, padx=(12, 2))
        self.variant_var = tk.StringVar()
        self.variant_box = ttk.Combobox(controls, textvariable=self.variant_var,
                                        width=12, state="readonly")
        self.variant_box.grid(row=0, column=5)

        # second row: parallel settings
        ttk.Label(controls, text="Threads:").grid(row=1, column=0,
                                                  pady=(8, 0), sticky="e")
        self.threads_var = tk.IntVar(value=self.max_threads)
        thread_box = ttk.Combobox(
            controls, textvariable=self.threads_var, width=5, state="readonly",
            values=[str(t) for t in range(1, self.max_threads + 1)])
        thread_box.grid(row=1, column=1, pady=(8, 0), sticky="w")

        ttk.Label(controls, text="Schedule:").grid(row=1, column=2,
                                                   pady=(8, 0), sticky="e")
        self.sched_var = tk.StringVar(value="static")
        ttk.Combobox(controls, textvariable=self.sched_var, width=10,
                     state="readonly",
                     values=["static", "dynamic", "guided"]).grid(
            row=1, column=3, pady=(8, 0), sticky="w")

        ttk.Label(controls, text="Param:").grid(row=1, column=4,
                                                pady=(8, 0), sticky="e")
        self.param_var = tk.StringVar(value="")
        self.param_entry = ttk.Entry(controls, textvariable=self.param_var,
                                     width=12)
        self.param_entry.grid(row=1, column=5, pady=(8, 0), sticky="w")

        self.run_button = ttk.Button(controls, text="Run",
                                     command=self.run_operation)
        self.run_button.grid(row=1, column=6, padx=(12, 0), pady=(8, 0))
        ttk.Button(controls, text="Save result...",
                   command=self.save_result).grid(row=1, column=7, padx=4,
                                                  pady=(8, 0))

        # --- results ----------------------------------------------------
        self.metrics = ttk.LabelFrame(self, text="Measurement", padding=8)
        self.metrics.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.metric_label = ttk.Label(
            self.metrics, font=("TkFixedFont", 10),
            text="Load an image and press Run.")
        self.metric_label.grid(row=0, column=0, sticky="w")
        self.blurb_label = ttk.Label(self.metrics, foreground="#444",
                                     wraplength=760, justify="left", text="")
        self.blurb_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        images = ttk.Frame(self)
        images.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=1)
        images.rowconfigure(1, weight=1)

        ttk.Label(images, text="Input").grid(row=0, column=0)
        ttk.Label(images, text="Output").grid(row=0, column=1)
        self.canvas_in = tk.Canvas(images, background="#202020",
                                   width=380, height=380, highlightthickness=0)
        self.canvas_in.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.canvas_out = tk.Canvas(images, background="#202020",
                                    width=380, height=380, highlightthickness=0)
        self.canvas_out.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        self.status = ttk.Label(self, text="", foreground="#333")
        self.status.grid(row=3, column=0, sticky="w", pady=(6, 0))

        self._on_operation()

    # -- helpers --------------------------------------------------------
    def _on_operation(self) -> None:
        spec = OPERATIONS[self.op_var.get()]
        _flag, choices = spec["choice"]
        self.variant_box["values"] = choices
        self.variant_var.set(choices[0])
        self.blurb_label.configure(text=spec["blurb"])

        if spec.get("size"):
            self.param_var.set("256 256")
            self.param_entry.configure(state="normal")
        elif spec.get("amount"):
            self.param_var.set("48")
            self.param_entry.configure(state="normal")
        elif spec.get("window"):
            self.param_var.set("4")
            self.param_entry.configure(state="normal")
        else:
            self.param_var.set("")
            self.param_entry.configure(state="disabled")

    def _run_cli(self, args: list[str]) -> dict:
        cmd = [str(BINARY), *args, "--json", "--quiet"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "the operation failed")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def _generate_sample(self) -> None:
        target = self.tmpdir / "sample.ppm"
        try:
            self._run_cli(["gen", str(target), "--size", "512"])
        except Exception as exc:
            messagebox.showerror("imgproc", f"Could not generate a sample "
                                            f"image:\n{exc}")
            return
        self.source = target
        self._show(self.canvas_in, target, which="in")
        self.canvas_out.delete("all")
        self.status.configure(text=f"Sample image generated  ({target.name})")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.ppm *.pgm *.png *.jpg *.jpeg *.bmp *.tga"),
                       ("Netpbm", "*.ppm *.pgm"), ("All files", "*.*")])
        if not path:
            return
        source = Path(path)
        if source.suffix.lower() not in (".ppm", ".pgm"):
            source = self._convert_to_netpbm(source)
            if source is None:
                return
        self.source = source
        self._show(self.canvas_in, source, which="in")
        self.canvas_out.delete("all")
        self.status.configure(text=f"Loaded {Path(path).name}")

    def _convert_to_netpbm(self, path: Path) -> Path | None:
        """PNG/JPEG/BMP/TGA -> PPM via the C binary (stb_image); no Pillow."""
        target = self.tmpdir / (path.stem + ".ppm")
        try:
            self._run_cli(["convert", str(path), str(target)])
        except Exception as exc:
            messagebox.showerror("imgproc", f"Could not read the image:\n{exc}")
            return None
        return target

    def _show(self, canvas: tk.Canvas, path: Path, which: str) -> None:
        """Display a PGM/PPM file.  Tk reads these natively."""
        try:
            photo = tk.PhotoImage(file=str(path))
        except tk.TclError as exc:
            self.status.configure(text=f"Cannot display {path.name}: {exc}")
            return
        # Fit the preview to the canvas.  Tk scales only by integer factors,
        # which is enough for a preview and keeps this dependency-free.
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 380)
        ch = max(canvas.winfo_height(), 380)
        iw, ih = photo.width(), photo.height()
        shrink = max(1, -(-iw // cw), -(-ih // ch))
        if shrink > 1:
            photo = photo.subsample(shrink, shrink)
        else:
            grow = max(1, min(cw // iw, ch // ih))
            if grow > 1:
                photo = photo.zoom(grow, grow)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo)
        if which == "in":
            self._photo_in = photo
        else:
            self._photo_out = photo

    # -- running --------------------------------------------------------
    def run_operation(self) -> None:
        if not self.source:
            messagebox.showinfo("imgproc", "Load or generate an image first.")
            return

        name = self.op_var.get()
        spec = OPERATIONS[name]
        out = self.tmpdir / f"result{spec['ext']}"
        flag, _choices = spec["choice"]

        args = [spec["cmd"], str(self.source), str(out), flag,
                self.variant_var.get(),
                "--threads", str(self.threads_var.get()),
                "--schedule", self.sched_var.get(),
                "--repeat", "3"]

        try:
            if spec.get("size"):
                h, _, w = self.param_var.get().partition(" ")
                args += ["--size", h.strip(), (w.strip() or h.strip())]
            elif spec.get("amount"):
                args += ["--amount", self.param_var.get().strip()]
            elif spec.get("window"):
                window = self.param_var.get().strip()
                args += ["--size", window, "--stride", window]
        except Exception:
            messagebox.showerror("imgproc", "Could not read the parameter.")
            return

        self.run_button.configure(state="disabled")
        self.status.configure(text="Running...")
        # Run off the Tk thread so the window stays responsive on big images.
        threading.Thread(target=self._worker, args=(args, out), daemon=True
                         ).start()

    def _worker(self, args: list[str], out: Path) -> None:
        try:
            info = self._run_cli(args)
            self._queue.put(("ok", info, out))
        except Exception as exc:
            self._queue.put(("error", str(exc), None))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload, out = self._queue.get_nowait()
                if kind == "ok":
                    self._on_success(payload, out)
                else:
                    self.run_button.configure(state="normal")
                    self.status.configure(text="Failed.")
                    messagebox.showerror("imgproc", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _on_success(self, info: dict, out: Path) -> None:
        self.run_button.configure(state="normal")
        self.result_path = out
        self._show(self.canvas_out, out, which="out")

        threads = info.get("threads", 1)
        serial = info.get("serial_ms", 0.0)
        omp = info.get("omp_ms", 0.0)
        speedup = info.get("speedup", 0.0)
        efficiency = info.get("efficiency", 0.0) * 100.0
        self.metric_label.configure(
            text=(f"serial {serial:9.3f} ms     "
                  f"openmp {omp:9.3f} ms  on {threads} thread(s)     "
                  f"speedup {speedup:5.2f}x     efficiency {efficiency:5.1f}%"))
        self.status.configure(
            text=f"{info.get('operation')} ({info.get('detail')})  ->  "
                 f"{info.get('out_height')} x {info.get('out_width')} x "
                 f"{info.get('out_channels')}")

    def save_result(self) -> None:
        if not self.result_path or not self.result_path.is_file():
            messagebox.showinfo("imgproc", "Run an operation first.")
            return
        target = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("Netpbm", "*.ppm *.pgm"), ("All files", "*.*")])
        if not target:
            return
        try:
            self._run_cli(["convert", str(self.result_path), target])
        except Exception as exc:
            messagebox.showerror("imgproc", f"Could not save:\n{exc}")
            return
        self.status.configure(text=f"Saved to {target}")


# ---------------------------------------------------------------------------
# Test-suite tab
# ---------------------------------------------------------------------------
SWEEP_OPS: dict[str, list[str]] = {
    "conv sobelx 3x3":     ["conv", "--kernel", "sobelx"],
    "conv gauss 5x5":      ["conv", "--kernel", "gauss"],
    "greyscale luminance": ["grayscale", "--method", "luminance"],
    "normalize equalize":  ["normalize", "--method", "equalize"],
    "resize bicubic":      ["resize", "--size", "512", "512", "--method", "bicubic"],
    "pool max 2x2":        ["pool", "--type", "max", "--size", "2", "--stride", "2"],
    "pad reflect 64":      ["pad", "--amount", "64", "--mode", "reflect"],
}


class TestSuite(ttk.Frame):
    """Runs each test stage of the project and streams its output."""

    STAGES = [
        ("correctness", "C correctness suite",
         "41k assertions: every kernel vs. hand-checked cases, serial == OpenMP, "
         "1/2/3/4 threads identical, all schedules identical."),
        ("cli", "CLI & image-format tests",
         "The shipped binary end to end: PNG/BMP/TGA lossless, JPEG decode, "
         "PNG input == PPM input, thread/schedule invariance."),
        ("verify", "Cross-validate vs NumPy",
         "Compares the C output with an independent NumPy implementation "
         "(needs NumPy)."),
        ("sweep", "Thread-count sweep",
         "Runs the selected operation at 1..N threads: speedup table, and a "
         "check that every thread count gives bit-identical output."),
        ("bench", "Quick benchmark",
         "Short strong-scaling run of every kernel (bin/benchmark --quick)."),
    ]

    def __init__(self, master: tk.Misc, max_threads: int) -> None:
        super().__init__(master, padding=10)
        self.max_threads = max_threads
        self.tmpdir = Path(tempfile.mkdtemp(prefix="imgproc_tests_"))
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self.badges: dict[str, ttk.Label] = {}
        self.buttons: list[ttk.Button] = []
        self._build()
        self.after(80, self._drain)

    # -- layout -----------------------------------------------------------
    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        stages = ttk.LabelFrame(self, text="Test stages", padding=8)
        stages.grid(row=0, column=0, sticky="ew")
        stages.columnconfigure(2, weight=1)

        for row, (key, title, blurb) in enumerate(self.STAGES):
            btn = ttk.Button(stages, text=title, width=26,
                             command=lambda k=key: self.run([k]))
            btn.grid(row=row, column=0, sticky="w", pady=2)
            self.buttons.append(btn)
            badge = ttk.Label(stages, text="  --  ", width=8, anchor="center",
                              style="Idle.TLabel")
            badge.grid(row=row, column=1, padx=8)
            self.badges[key] = badge
            ttk.Label(stages, text=blurb, foreground="#555", wraplength=520,
                      justify="left").grid(row=row, column=2, sticky="w")

        opts = ttk.Frame(stages)
        opts.grid(row=len(self.STAGES), column=0, columnspan=3, sticky="w",
                  pady=(8, 0))
        ttk.Label(opts, text="Sweep operation:").grid(row=0, column=0)
        self.sweep_var = tk.StringVar(value="conv gauss 5x5")
        ttk.Combobox(opts, textvariable=self.sweep_var, state="readonly",
                     width=20, values=list(SWEEP_OPS)).grid(row=0, column=1,
                                                            padx=4)
        ttk.Label(opts, text="Image size:").grid(row=0, column=2, padx=(12, 2))
        self.size_var = tk.StringVar(value="1024")
        ttk.Combobox(opts, textvariable=self.size_var, state="readonly",
                     width=6, values=["256", "512", "1024", "2048"]).grid(
            row=0, column=3)
        run_all = ttk.Button(opts, text="Run all stages", style="Accent.TButton",
                             command=lambda: self.run([k for k, *_ in self.STAGES]))
        run_all.grid(row=0, column=4, padx=(20, 0))
        self.buttons.append(run_all)
        clear = ttk.Button(opts, text="Clear log",
                           command=lambda: self.log.delete("1.0", "end"))
        clear.grid(row=0, column=5, padx=4)

        self.log = scrolledtext.ScrolledText(
            self, height=18, font=("TkFixedFont", 10), background="#14161a",
            foreground="#d8dee9", insertbackground="#d8dee9", relief="flat",
            padx=8, pady=6)
        self.log.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log.tag_configure("head", foreground="#88c0d0",
                               font=("TkFixedFont", 10, "bold"))
        self.log.tag_configure("pass", foreground="#a3be8c",
                               font=("TkFixedFont", 10, "bold"))
        self.log.tag_configure("fail", foreground="#bf616a",
                               font=("TkFixedFont", 10, "bold"))

        self.summary = ttk.Label(self, text="Pick a stage, or Run all stages.")
        self.summary.grid(row=2, column=0, sticky="w", pady=(6, 0))

    # -- running ----------------------------------------------------------
    def run(self, keys: list[str]) -> None:
        if self._busy:
            return
        self._busy = True
        for b in self.buttons:
            b.configure(state="disabled")
        for k in keys:
            self._badge(k, "queued", "Idle.TLabel")
        threading.Thread(target=self._worker, args=(keys,), daemon=True).start()

    def _worker(self, keys: list[str]) -> None:
        results = []
        for key in keys:
            self._queue.put(("badge", key, "running"))
            title = next(t for k, t, _ in self.STAGES if k == key)
            self._queue.put(("head", f"\n=== {title} ===\n"))
            try:
                ok = getattr(self, f"_stage_{key}")()
            except Exception as exc:  # report, never crash the GUI
                self._queue.put(("text", f"error: {exc}\n"))
                ok = False
            results.append(ok)
            self._queue.put(("badge", key, "PASS" if ok else "FAIL"))
        self._queue.put(("done", sum(results), len(results)))

    def _stream(self, cmd: list[str]) -> bool:
        shown = [Path(c).relative_to(ROOT).as_posix()
                 if c.startswith(str(ROOT)) else Path(c).name if c == sys.executable
                 else c for c in cmd]
        self._queue.put(("text", "$ " + " ".join(shown) + "\n"))
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            self._queue.put(("text", line))
        return proc.wait() == 0

    def _stage_correctness(self) -> bool:
        return self._stream([str(ROOT / "bin" / "test_correctness")])

    def _stage_cli(self) -> bool:
        return self._stream(["sh", str(ROOT / "tests" / "test_cli.sh")])

    def _stage_verify(self) -> bool:
        return self._stream([sys.executable, "-u",
                             str(ROOT / "tools" / "verify_against_reference.py")])

    def _stage_bench(self) -> bool:
        return self._stream([str(ROOT / "bin" / "benchmark"), "--quick"])

    def _stage_sweep(self) -> bool:
        size = self.size_var.get()
        label = self.sweep_var.get()
        op = SWEEP_OPS[label]
        src = self.tmpdir / f"sweep_{size}.ppm"
        if not src.exists():
            subprocess.run([str(BINARY), "gen", str(src), "--size", size,
                            "--quiet"], check=True)
        self._queue.put(("text", f"operation: {label}   image: {size} x {size} x 3"
                                 f"   threads: 1..{self.max_threads}\n\n"))
        self._queue.put(("text", f"{'threads':>8} {'serial ms':>11} "
                                 f"{'openmp ms':>11} {'speedup':>9} "
                                 f"{'efficiency':>11}  output sha1\n"))
        digests = set()
        threads = sorted({1, *[t for t in (2, 4, 8, 16, 32, 64)
                               if t <= self.max_threads], self.max_threads})
        for t in threads:
            out = self.tmpdir / f"sweep_out_{t}.ppm"
            cmd = [str(BINARY), op[0], str(src), str(out), *op[1:],
                   "--threads", str(t), "--repeat", "3", "--json", "--quiet"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                self._queue.put(("text", proc.stderr))
                return False
            info = json.loads(proc.stdout.strip().splitlines()[-1])
            digest = hashlib.sha1(out.read_bytes()).hexdigest()[:12]
            digests.add(digest)
            self._queue.put(("text",
                f"{t:>8} {info['serial_ms']:>11.3f} {info['omp_ms']:>11.3f} "
                f"{info['speedup']:>8.2f}x {info['efficiency'] * 100:>10.1f}%"
                f"  {digest}\n"))
        same = len(digests) == 1
        self._queue.put(("text", "\noutput identical at every thread count: "
                                 f"{'yes' if same else 'NO -- possible data race'}\n"))
        return same

    # -- UI updates (Tk thread only) ----------------------------------------
    def _badge(self, key: str, text: str, style: str) -> None:
        self.badges[key].configure(text=text, style=style)

    def _drain(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "text":
                    self.log.insert("end", item[1])
                    self.log.see("end")
                elif kind == "head":
                    self.log.insert("end", item[1], "head")
                    self.log.see("end")
                elif kind == "badge":
                    _, key, state = item
                    style = {"PASS": "Pass.TLabel", "FAIL": "Fail.TLabel"}.get(
                        state, "Run.TLabel")
                    self._badge(key, state, style)
                    if state in ("PASS", "FAIL"):
                        self.log.insert("end", f"[{state}]\n",
                                        "pass" if state == "PASS" else "fail")
                elif kind == "done":
                    _, passed, total = item
                    self._busy = False
                    for b in self.buttons:
                        b.configure(state="normal")
                    self.summary.configure(
                        text=f"{passed}/{total} stage(s) passed.")
        except queue.Empty:
            pass
        self.after(80, self._drain)


def _styles() -> None:
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Idle.TLabel", background="#e5e7eb", foreground="#374151")
    style.configure("Run.TLabel", background="#fde68a", foreground="#78350f")
    style.configure("Pass.TLabel", background="#bbf7d0", foreground="#14532d",
                    font=("TkDefaultFont", 9, "bold"))
    style.configure("Fail.TLabel", background="#fecaca", foreground="#7f1d1d",
                    font=("TkDefaultFont", 9, "bold"))
    style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))


def main() -> int:
    if not BINARY.is_file():
        print(f"error: {BINARY} not found.  Build it first with `make`.")
        return 1
    root = tk.Tk()
    root.title("imgproc -- parallel image processing (OpenMP)  |  MTCS-103P")
    root.geometry("980x780")
    _styles()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    tabs = ttk.Notebook(root)
    tabs.grid(row=0, column=0, sticky="nsew")
    ops = Application(tabs)
    tabs.add(ops, text="  Operations  ")
    tabs.add(TestSuite(tabs, ops.max_threads), text="  Test suite  ")
    if "--tab" in sys.argv:  # used to take the README screenshots
        tabs.select(int(sys.argv[sys.argv.index("--tab") + 1]))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
