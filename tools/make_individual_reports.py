#!/usr/bin/env python3
"""Render the two individual reports to PDF.

    reports/REPORT_26010303001.md  ->  reports/26010303001_Report.pdf
    reports/REPORT_26010303004.md  ->  reports/26010303004_Report.pdf

Uses the renderer in make_report_pdf.py (Chromium via Playwright preferred,
then wkhtmltopdf, then pandoc).  Run tools/make_part_figures.py first if the
figures need regenerating.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_report_pdf as renderer  # noqa: E402

REPORTS = ROOT / "reports"
STUDENTS = ["26010303001", "26010303004"]


def main() -> int:
    ok = True
    for reg in STUDENTS:
        renderer.MD_PATH = REPORTS / f"REPORT_{reg}.md"
        renderer.HTML_PATH = REPORTS / f"REPORT_{reg}.html"
        renderer.PDF_PATH = REPORTS / f"{reg}_Report.pdf"
        renderer.build_html()
        if renderer.build_pdf():
            print(f"wrote {renderer.PDF_PATH.relative_to(ROOT)}")
        else:
            print(f"could not render {renderer.PDF_PATH.name}")
            ok = False
        renderer.HTML_PATH.unlink(missing_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
