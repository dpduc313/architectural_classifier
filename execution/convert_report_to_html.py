import re
from pathlib import Path
import markdown

PROJECT_ROOT = Path(__file__).parent.parent
MD_PATH = PROJECT_ROOT / "master_report.md"
HTML_PATH = PROJECT_ROOT / "master_report.html"

def build_html():
    if not MD_PATH.exists():
        print(f"Error: {MD_PATH} not found.")
        return

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process math syntax for MathJax compatibility
    # Replace $...$ with \(...\) and $$...$$ with \[...\]
    # Avoid replacing inline file links or image paths
    
    # Process fenced mermaid blocks to <pre class="mermaid">...</pre>
    def replace_mermaid(match):
        content = match.group(1).strip()
        return f'<div class="mermaid-container"><pre class="mermaid">\n{content}\n</pre></div>'

    md_processed = re.sub(r'```mermaid\s*\n(.*?)\n```', replace_mermaid, md_text, flags=re.DOTALL)

    # Convert Markdown to HTML
    extensions = [
        'fenced_code',
        'tables',
        'toc',
        'attr_list',
        'def_list',
        'sane_lists'
    ]
    
    html_body = markdown.markdown(md_processed, extensions=extensions)

    # Wrap in modern template with CSS & JS libraries (Mermaid.js & MathJax 3)
    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo Đồ án Thị giác Máy tính HK253 — Phân loại Di sản Kiến trúc TP.HCM</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Mermaid JS -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
            }});
        }});
    </script>

    <!-- MathJax 3 -->
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }},
            svg: {{
                fontCache: 'global'
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary: #0d9488;
            --accent: #f59e0b;
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --radius-md: 10px;
            --radius-lg: 16px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.7;
            font-size: 16px;
            padding: 0;
            margin: 0;
        }}

        /* Header Header Banner */
        .report-header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 60px 20px;
            text-align: center;
            border-bottom: 4px solid var(--primary);
            box-shadow: var(--shadow-lg);
        }}

        .report-header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 15px;
            background: linear-gradient(90deg, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .report-header p {{
            font-size: 1.1rem;
            color: #94a3b8;
            max-width: 800px;
            margin: 0 auto;
        }}

        .header-meta {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 25px;
            flex-wrap: wrap;
        }}

        .meta-tag {{
            background: rgba(255, 255, 255, 0.1);
            padding: 6px 16px;
            border-radius: 30px;
            font-size: 0.875rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }}

        /* Layout Container */
        .container {{
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 24px;
        }}

        .main-content {{
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 48px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
        }}

        /* Headings */
        h1, h2, h3, h4, h5 {{
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
            font-weight: 700;
            line-height: 1.3;
        }}

        h2 {{
            font-size: 1.75rem;
            margin-top: 48px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
            color: var(--primary-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        h3 {{
            font-size: 1.35rem;
            margin-top: 32px;
            margin-bottom: 16px;
            color: #1e293b;
        }}

        h4 {{
            font-size: 1.1rem;
            margin-top: 24px;
            margin-bottom: 12px;
            color: #334155;
        }}

        p {{
            margin-bottom: 16px;
            color: #334155;
        }}

        /* Blockquotes / Callouts */
        blockquote {{
            background: #f0f9ff;
            border-left: 4px solid var(--primary);
            padding: 16px 20px;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            margin: 24px 0;
            color: #1e40af;
            font-style: italic;
        }}

        blockquote p {{
            margin-bottom: 0;
        }}

        /* Lists */
        ul, ol {{
            margin-bottom: 20px;
            padding-left: 28px;
            color: #334155;
        }}

        li {{
            margin-bottom: 8px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 28px 0;
            font-size: 0.95rem;
            background: #ffffff;
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }}

        th {{
            background: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
            text-align: left;
            padding: 14px 16px;
            border-bottom: 2px solid var(--border-color);
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: #334155;
        }}

        tr:nth-child(even) td {{
            background-color: #f8fafc;
        }}

        tr:hover td {{
            background-color: #f1f5f9;
        }}

        /* Images & Visualizations */
        img {{
            max-width: 100%;
            height: auto;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-md);
            margin: 20px 0 10px 0;
            display: block;
            border: 1px solid var(--border-color);
        }}

        em {{
            display: block;
            text-align: center;
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-bottom: 24px;
            font-style: normal;
        }}

        /* Code & Pre */
        code {{
            font-family: 'Fira Code', monospace;
            background: #f1f5f9;
            color: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}

        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 20px;
            border-radius: var(--radius-md);
            overflow-x: auto;
            margin: 24px 0;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }}

        pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}

        /* Mermaid Container */
        .mermaid-container {{
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 24px;
            margin: 28px 0;
            box-shadow: var(--shadow-sm);
            display: flex;
            justify-content: center;
        }}

        .mermaid {{
            width: 100%;
            text-align: center;
        }}

        /* Links */
        a {{
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        /* Back to top button */
        .top-btn {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: var(--primary);
            color: #fff;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-lg);
            cursor: pointer;
            border: none;
            transition: transform 0.2s, background 0.2s;
            z-index: 1000;
        }}

        .top-btn:hover {{
            background: var(--primary-dark);
            transform: translateY(-3px);
        }}

        /* Footer */
        .report-footer {{
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color);
            margin-top: 60px;
        }}

        @media (max-width: 768px) {{
            .main-content {{
                padding: 24px;
            }}
            .report-header h1 {{
                font-size: 1.8rem;
            }}
        }}
    </style>
</head>
<body>

    <header class="report-header">
        <h1>BÁO CÁO TỔNG KẾT ĐỒ ÁN MÔN THỊ GIÁC MÁY TÍNH (HK253)</h1>
        <p>Phân loại Ảnh Các Công trình Di sản Kiến trúc tại TP. Hồ Chí Minh sử dụng CNNs & Vision Transformers</p>
        <div class="header-meta">
            <span class="meta-tag">Môn học: Thị giác Máy tính (HK253)</span>
            <span class="meta-tag">Repository: dpduc313/architectural_classifier</span>
            <span class="meta-tag">Ngày hoàn thành: 03/08/2026</span>
        </div>
    </header>

    <div class="container">
        <main class="main-content">
            {html_body}
        </main>

        <footer class="report-footer">
            <p>© 2026 Architectural Heritage Classifier Project. Antigravity Deep Learning Team.</p>
        </footer>
    </div>

    <button class="top-btn" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})" title="Lên đầu trang">
        ↑
    </button>

</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Successfully converted master_report.md -> {HTML_PATH}")

if __name__ == "__main__":
    build_html()
