import json
nb = json.load(open('Version5_Chatbot_RAG_Clean.ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    first = c['source'][0][:55].strip() if c['source'] else '(empty)'
    # avoid unicode issues on Windows console
    safe = first.encode('ascii', 'replace').decode()
    print(f"  {i+1:2d}. [{ct:8s}] {safe}")
