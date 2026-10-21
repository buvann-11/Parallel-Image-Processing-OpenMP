#!/usr/bin/env python3
"""Render report/REPORT.md to report/REPORT.pdf.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Run this after editing the report (in particular, after filling in the team
table and re-running the benchmarks on your own machine):

    python3 tools/make_report_pdf.py

Needs `markdown` (pip install markdown) and one of `wkhtmltopdf` or `pandoc`.
If neither is available the script still writes report/REPORT.html, which any
browser can print to PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
MD_PATH = REPORT_DIR / "REPORT.md"
# The HTML is written NEXT TO the markdown, not in a temp directory, so the
# relative figures/*.png links resolve when the PDF renderer loads it.
HTML_PATH = REPORT_DIR / "REPORT.html"
PDF_PATH = REPORT_DIR / "REPORT.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'DejaVu Sans', Arial, Helvetica, sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #bbb;
     padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 16px; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 12px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
        font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left;
         vertical-align: top; }
th { background: #eee; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt;
       background: #f4f4f4; padding: 1px 3px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px;
      overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; font-size: 8.5pt; }
img { max-width: 100%; page-break-inside: avoid; display: block;
      margin: 12px auto; border: 1px solid #ddd; }
blockquote { border-left: 4px solid #d0a000; background: #fffbe8;
             margin: 10px 0; padding: 8px 12px; }
hr { border: none; border-top: 1px solid #ccc; margin: 18px 0; }
"""


def build_html() -> None:
    try:
        import markdown
    except ImportError:
        print("error: the `markdown` package is required.\n"
              "       pip install markdown")
        raise SystemExit(1)

    text = MD_PATH.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    HTML_PATH.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>MTCS-103P Mini Project Report</title>\n<style>{CSS}</style>"
        f"</head>\n<body>\n{body}\n</body></html>\n",
        encoding="utf-8")
    print(f"  wrote {HTML_PATH.relative_to(ROOT)}")


def build_pdf() -> bool:
    if shutil.which("wkhtmltopdf"):
        cmd = ["wkhtmltopdf", "--quiet", "--enable-local-file-access",
               "--page-size", "A4",
               "--margin-top", "16mm", "--margin-bottom", "16mm",
               "--margin-left", "14mm", "--margin-right", "14mm",
               HTML_PATH.name, PDF_PATH.name]
        # Run from the report directory so relative image paths resolve.
        result = subprocess.run(cmd, cwd=REPORT_DIR,
                                capture_output=True, text=True)
        if result.returncode == 0 and PDF_PATH.is_file():
            return True
        print(f"  wkhtmltopdf failed: {result.stderr.strip()[:200]}")

    if shutil.which("pandoc"):
        cmd = ["pandoc", MD_PATH.name, "-o", PDF_PATH.name,
               "--resource-path", ".", "-V", "geometry:margin=2cm"]
        result = subprocess.run(cmd, cwd=REPORT_DIR,
                                capture_output=True, text=True)
        if result.returncode == 0 and PDF_PATH.is_file():
            return True
        print(f"  pandoc failed: {result.stderr.strip()[:200]}")

    return False


def main() -> int:
    if not MD_PATH.is_file():
        print(f"error: {MD_PATH} not found")
        return 1

    missing = [f.name for f in [
        REPORT_DIR / "figures" / n for n in
        ("speedup.png", "efficiency.png", "schedules.png", "karp_flatt.png",
         "chunk_size.png", "results_part1.png", "results_part2.png",
         "gui_screenshot.png")] if not f.is_file()]
    if missing:
        print(f"  warning: {len(missing)} figure(s) missing: "
              f"{', '.join(missing)}")
        print("           run `make bench` and the plotting scripts first")

    print("Rendering the report...")
    build_html()
    if build_pdf():
        size = PDF_PATH.stat().st_size / 1024
        print(f"  wrote {PDF_PATH.relative_to(ROOT)} ({size:.0f} KB)")
    else:
        print("  no PDF renderer found (wkhtmltopdf or pandoc).")
        print(f"  Open {HTML_PATH.relative_to(ROOT)} in a browser and "
              "print to PDF instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
