#!/usr/bin/env python3
"""Render one copy of the report per team member.

The content is identical; each copy names its submitter on the cover, as the
assignment asks each student to submit under their own registration number.
Writes reports/<regno>_Report.pdf and leaves reports/REPORT.pdf as the
team copy.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "reports" / "REPORT.md"
PDF = ROOT / "reports" / "REPORT.pdf"
TEAM = [("26010303001", "B. Sai Buvann"), ("26010303004", "M. Sai Praneeth")]
COVER = "**MTCS-103P — High Performance Computing Laboratory · Mini Project**"


def render() -> None:
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_report_pdf.py")],
                   check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    original = MD.read_text(encoding="utf-8")
    try:
        for reg, name in TEAM:
            MD.write_text(original.replace(
                COVER,
                f"{COVER}\n\n**Submitted by:** {name} — "
                f"Registration no. {reg}", 1),
                encoding="utf-8")
            render()
            target = ROOT / "reports" / f"{reg}_Report.pdf"
            shutil.copyfile(PDF, target)
            print(f"wrote {target.relative_to(ROOT)}")
    finally:
        MD.write_text(original, encoding="utf-8")
    render()
    (ROOT / "reports" / "REPORT.html").unlink(missing_ok=True)
    print(f"wrote {PDF.relative_to(ROOT)} (team copy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
