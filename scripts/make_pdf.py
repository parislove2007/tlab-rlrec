"""Renders README.md to a standalone HTML page and prints it to PDF."""
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import markdown

from rlrec.config import ROOT, REPORTS

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
:root { --ink:#1a1a1a; --mut:#5b6472; --rule:#dfe3e8; --accent:#0b5fff; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans","Liberation Sans",Arial,sans-serif;
       font-size: 9.6pt; line-height: 1.5; color: var(--ink); margin: 0;
       max-width: 185mm; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4mm; letter-spacing: -0.2pt; }
h2 { font-size: 13.5pt; margin: 9mm 0 3mm; padding-bottom: 1.5mm;
     border-bottom: 1.2px solid var(--rule); break-after: avoid; }
h3 { font-size: 11pt; margin: 6mm 0 2mm; break-after: avoid; }
h4 { font-size: 9.8pt; margin: 4mm 0 1.5mm; color: var(--mut); break-after: avoid; }
p { margin: 0 0 2.6mm; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
strong { font-weight: 700; }
code { font-family: "DejaVu Sans Mono","Liberation Mono",monospace; font-size: 8.4pt;
       background: #f4f6f8; padding: 0.3mm 1mm; border-radius: 2px; }
pre { background: #f7f9fb; border: 1px solid var(--rule); border-left: 2.5px solid var(--accent);
      padding: 2.5mm 3mm; overflow-x: auto; break-inside: avoid; margin: 0 0 3mm; }
pre code { background: none; padding: 0; font-size: 8pt; line-height: 1.42; }
table { border-collapse: collapse; width: 100%; font-size: 8.1pt; margin: 0 0 4mm;
        break-inside: avoid; }
th { background: #f1f4f7; text-align: left; font-weight: 700; }
th, td { border: 1px solid var(--rule); padding: 1.1mm 1.6mm; vertical-align: top; }
tbody tr:nth-child(even) { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: 2mm auto 1mm;
      break-inside: avoid; }
em { color: var(--mut); font-style: normal; font-size: 8.4pt; display: block;
     text-align: center; margin-bottom: 4mm; }
blockquote { margin: 0 0 3mm; padding-left: 3mm; border-left: 2px solid var(--rule);
             color: var(--mut); }
h2 + table, h3 + table, h2 + p, h3 + p { break-before: avoid; }
"""


def main():
    md = (ROOT / "README.md").read_text()
    # the HTML lives in reports/, so image paths lose their directory prefix
    md = md.replace("](reports/", "](")
    html = markdown.markdown(
        md, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )
    page = (f"<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
            f"<title>RL-дообучение рекомендательной модели</title>"
            f"<style>{CSS}</style></head><body>{html}</body></html>")
    out_html = REPORTS / "report.html"
    out_html.write_text(page)

    pdf = ROOT / "report.pdf"
    cmd = ["google-chrome-stable", "--headless=new", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--virtual-time-budget=20000",
           f"--print-to-pdf={pdf}", out_html.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf.exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("chrome did not produce a pdf")
    print(f"{out_html} -> {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
