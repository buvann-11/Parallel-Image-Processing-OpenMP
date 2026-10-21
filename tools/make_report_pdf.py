#!/usr/bin/env python3
"""Markdown -> PDF renderer used by make_individual_reports.py.

Run on its own it renders reports/REPORT.md to reports/REPORT.pdf.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Run this after editing the report (in particular, after filling in the team
table and re-running the benchmarks on your own machine):

    python3 tools/make_report_pdf.py

Needs `markdown` (pip install markdown) and one of `wkhtmltopdf` or `pandoc`.
If neither is available the script still writes reports/REPORT.html, which any
browser can print to PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
MD_PATH = REPORT_DIR / "REPORT.md"
# The HTML is written NEXT TO the markdown, not in a temp directory, so the
# relative figures/*.png links resolve when the PDF renderer loads it.
HTML_PATH = REPORT_DIR / "REPORT.html"
PDF_PATH = REPORT_DIR / "REPORT.pdf"

CSS = """
@page { size: A4; }
body { font-family: 'DejaVu Sans', Arial, Helvetica, sans-serif;
       font-size: 12pt; line-height: 1.4; color: #111; }
h1 { font-size: 21pt; border-bottom: 2px solid #333; padding-bottom: 6px;
     margin-bottom: 8px; }
h2 { font-size: 16pt; margin-top: 18px; break-after: avoid; border-bottom: 1px solid #bbb;
     padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 13.5pt; margin-top: 14px; page-break-after: avoid; }
p, li { margin: 4px 0; }
ul, ol { margin: 4px 0 6px 0; padding-left: 22px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0;
        font-size: 11pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left;
         vertical-align: top; }
th { background: #eee; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 10.5pt;
       background: #f4f4f4; padding: 0 3px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px;
      page-break-inside: avoid; }
pre code { background: none; font-size: 10.5pt; }
img { width: 100%; max-width: 100%; break-inside: avoid; page-break-inside: avoid; display: block;
      margin: 10px auto; border: 1px solid #ddd; }
hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
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


def build_pdf_chromium() -> bool:
    """Chromium (via Playwright) honours the print CSS exactly and keeps PNG
    figures lossless, so it is preferred when available."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(HTML_PATH.resolve().as_uri(), wait_until="networkidle")
            page.pdf(path=str(PDF_PATH), format="A4", print_background=True,
                     margin={"top": "13mm", "bottom": "13mm",
                             "left": "14mm", "right": "14mm"},
                     display_header_footer=True,
                     header_template="<span></span>",
                     footer_template=(
                         "<div style='font-size:8pt;width:100%;text-align:center;"
                         "color:#777'><span class='pageNumber'></span> / "
                         "<span class='totalPages'></span></div>"))
            browser.close()
        return PDF_PATH.is_file()
    except Exception as exc:  # fall back to the other renderers
        print(f"  chromium failed: {str(exc)[:200]}")
        return False


def build_pdf() -> bool:
    if build_pdf_chromium():
        return True
    if shutil.which("wkhtmltopdf"):
        cmd = ["wkhtmltopdf", "--quiet", "--enable-local-file-access",
               "--image-quality", "100", "--image-dpi", "600",
               "--title", "MTCS-103P Mini Project Report",
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
