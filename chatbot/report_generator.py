from typing import List, Dict
from datetime import datetime


def generate_html_report(results: List[Dict[str, str]], output_path: str) -> None:
    """Generate an HTML report from evaluation results.

    Args:
        results: List of dictionaries containing evaluation results
        output_path: Path to save the HTML report
    """
    html = f"""<!DOCTYPE html>
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
        html += f"""
    <div class="result">
        <div class="result-number">{i}</div>
        <div class="question">{result['question']}</div>

        <div class="section">
            <span class="label">קטגוריה:</span>
            <span class="category">{result['category']}</span>
        </div>

        <div class="expected">
            <div class="label">תשובה צפויה:</div>
            <div>{result['expected_answer']}</div>
            <div class="citation">{result['expected_citation']}</div>
        </div>

        <div class="actual">
            <div class="label">תשובה בפועל:</div>
            <div>{result['actual_answer']}</div>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
