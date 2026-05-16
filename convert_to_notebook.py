"""Script chuyen run_chatbot.py thanh file .ipynb"""
import json, re

with open("run_chatbot.py", "r", encoding="utf-8") as f:
    content = f.read()

# Tach theo cac PHAN (section headers)
parts = re.split(r'(# ═+\n# PHẦN \d+:.*\n# ═+\n)', content)

cells = []

# Cell dau: docstring
if parts[0].strip():
    lines = parts[0].strip()
    # Markdown cell cho docstring
    doc = lines.replace('"""', '').strip()
    first_line = doc.split('\n')[0]
    rest = '\n'.join(doc.split('\n')[1:]).strip()
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {first_line}\n", f"{rest}\n"]
    })

# Ghep header + code
i = 1
while i < len(parts):
    if i < len(parts):
        header = parts[i].strip()
        # Lay ten phan lam markdown
        match = re.search(r'PHẦN \d+: (.+)', header)
        title = match.group(0) if match else header
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"]
        })
    
    if i + 1 < len(parts):
        code = parts[i + 1].strip()
        if code:
            source_lines = [line + "\n" for line in code.split("\n")]
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip("\n")
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            })
    i += 2

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "base",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.4"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = "Version5_Chatbot_RAG_Clean.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"[OK] Da tao: {output_path}")
print(f"     So cells: {len(cells)}")
