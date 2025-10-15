import html
import re
from datetime import datetime
from typing import Any, Dict, List

from source_utils import format_display_filename


def generate_html_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """Generate an HTML report from evaluation results.

    Args:
        results: List of dictionaries containing evaluation results
        output_path: Path to save the HTML report
    """
    html_content = f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>דוח הערכת מערכת שאלות ותשובות</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metadata {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .result {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .question {{
            font-weight: bold;
            color: #1976D2;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        .section {{
            margin: 15px 0;
        }}
        .label {{
            font-weight: bold;
            color: #555;
            display: inline-block;
            min-width: 120px;
        }}
        .category {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            background-color: #2196F3;
            color: white;
            font-size: 0.9em;
        }}
        .expected {{
            background-color: #fff3cd;
            padding: 10px;
            border-right: 4px solid #ffc107;
            margin: 10px 0;
        }}
        .actual {{
            background-color: #d1ecf1;
            padding: 10px;
            border-right: 4px solid #17a2b8;
            margin: 10px 0;
        }}
        .citation {{
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }}
        .result-number {{
            background-color: #4CAF50;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 10px;
        }}
        .sources {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-right: 4px solid #6c757d;
        }}
        .source-item {{
            padding: 5px 0;
            font-size: 0.9em;
        }}
        .source-item.referenced {{
            background-color: #fff3cd;
            padding: 5px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .answer-text {{
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <h1>דוח הערכת מערכת שאלות ותשובות</h1>
    <div class="metadata">
        <p><strong>תאריך הרצה:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>מספר שאלות:</strong> {len(results)}</p>
    </div>
"""

    for i, result in enumerate(results, 1):
        # Extract referenced document numbers from the answer
        answer_text = result["actual_answer"]
        referenced_docs = set()
        for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer_text):
            nums = match.group(1).replace(" ", "").split(",")
            referenced_docs.update(int(n) for n in nums if n.isdigit())

        # Format sources
        sources_html = ""
        if "sources" in result and result["sources"]:
            sources_html = '<div class="sources"><div class="label">מקורות:</div>'
            for idx, source in enumerate(result["sources"], 1):
                source_file = source.get("source_file", "Unknown")
                page_number = source.get("page_number", "Unknown")

                # Use server-side helper to format a display filename
                filename = format_display_filename(source_file, max_parts=3)

                # Check if this source was referenced in the answer
                is_referenced = idx in referenced_docs
                ref_class = "source-item referenced" if is_referenced else "source-item"

                sources_html += f'<div class="{ref_class}">[{idx}] {html.escape(filename)}, עמוד {html.escape(str(page_number))}</div>'
            sources_html += "</div>"

        # Escape and preserve newlines in answers
        escaped_answer = html.escape(answer_text)
        escaped_expected = html.escape(result["expected_answer"])

        html_content += f"""
    <div class="result">
        <div class="result-number">{i}</div>
        <div class="question">{html.escape(result['question'])}</div>

        <div class="section">
            <span class="label">קטגוריה:</span>
            <span class="category">{html.escape(result['category'])}</span>
        </div>

        <div class="expected">
            <div class="label">תשובה צפויה:</div>
            <div class="answer-text">{escaped_expected}</div>
            <div class="citation">{html.escape(result['expected_citation'])}</div>
        </div>

        <div class="actual">
            <div class="label">תשובה בפועל:</div>
            <div class="answer-text">{escaped_answer}</div>
        </div>

        {sources_html}
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
