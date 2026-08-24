import sys
from pathlib import Path
import os
import re

qmd_path = Path("book/12_exhaustive_model_analysis.qmd")
html_out_path = Path("book/12_exhaustive_model_analysis.html")
qmd_text = qmd_path.read_text(encoding='utf-8')

# Very basic markdown to HTML for viewing
html_lines = []
for line in qmd_text.split('\n'):
    if line.startswith('## '):
        html_lines.append(f"<h2>{line[3:]}</h2>")
    elif line.startswith('### '):
        html_lines.append(f"<h3>{line[4:]}</h3>")
    elif line.startswith('# '):
        html_lines.append(f"<h1>{line[2:]}</h1>")
    elif line.startswith('- '):
        html_lines.append(f"<li>{line[2:]}</li>")
    elif line.strip() == '---':
        html_lines.append("<hr>")
    else:
        # bold
        line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
        # code
        line = re.sub(r'`(.*?)`', r'<code>\1</code>', line)
        if line.strip():
            html_lines.append(f"<p>{line}</p>")

html = f"""<html>
<head><style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; background: #0f172a; color: #f8fafc; }}
h1, h2, h3 {{ color: #38bdf8; }}
hr {{ border-color: #1e293b; margin: 20px 0; }}
code {{ background: #1e293b; padding: 2px 5px; border-radius: 4px; color: #818cf8; }}
</style></head>
<body>
{"".join(html_lines)}
</body></html>"""
html_out_path.write_text(html, encoding='utf-8')
print("Rendered book/12_exhaustive_model_analysis.html")
