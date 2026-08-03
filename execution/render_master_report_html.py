from __future__ import annotations

from pathlib import Path

import markdown


PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_MD = PROJECT_ROOT / "master_report.md"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_HTML = RESULTS_DIR / "master_report.html"


CSS = """
:root {
  color-scheme: light;
}

@page {
  size: A4;
  margin: 16mm 14mm 18mm;
}

html, body {
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #111;
  background: #fff;
}

body {
  max-width: 980px;
  margin: 0 auto;
  padding: 18px 20px 32px;
}

h1, h2, h3, h4, h5, h6 {
  line-height: 1.2;
  margin: 1.1em 0 0.45em;
  page-break-after: avoid;
}

h1 { font-size: 24pt; }
h2 { font-size: 17pt; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
h3 { font-size: 13pt; }
h4 { font-size: 11.5pt; }

p, ul, ol, blockquote, table, pre {
  margin: 0.45em 0 0.85em;
}

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.6em auto;
}

figure {
  margin: 0.75em 0 1em;
}

code, pre {
  font-family: Consolas, "Courier New", monospace;
  font-size: 9.5pt;
}

pre {
  white-space: pre-wrap;
  word-break: break-word;
  background: #f6f8fa;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 10px 12px;
  overflow-x: auto;
}

code {
  background: #f6f8fa;
  border-radius: 4px;
  padding: 0.1em 0.3em;
}

table {
  border-collapse: collapse;
  width: 100%;
  page-break-inside: auto;
}

thead {
  display: table-header-group;
}

tfoot {
  display: table-footer-group;
}

tr {
  page-break-inside: avoid;
  page-break-after: auto;
}

th, td {
  border: 1px solid #d0d7de;
  padding: 6px 8px;
  vertical-align: top;
}

th {
  background: #f3f4f6;
  font-weight: 700;
}

blockquote {
  border-left: 4px solid #cbd5e1;
  padding-left: 12px;
  color: #374151;
}

hr {
  border: 0;
  border-top: 1px solid #ddd;
  margin: 1.2em 0;
}

.mermaid {
  white-space: pre-wrap;
  background: #fcfcfc;
}

@media print {
  a {
    color: inherit;
    text-decoration: none;
  }
}
"""


def build_html() -> str:
    source = SOURCE_MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        source,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Master Report</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html()
    html = html.replace('<head>', '<head>\n  <base href="../" />', 1)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
