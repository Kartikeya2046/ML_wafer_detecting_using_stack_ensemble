import json

notebook_path = r"d:\assignment college\VLSI_PROJECT\WMPC_Stacking_TF2\run_code\data_preprocessing.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Inject sys.path before the import
            if line.strip().startswith('from extract_manual_features import'):
                new_source.append("import sys\n")
                new_source.append("sys.path.append(r'd:/assignment college/VLSI_PROJECT/WMPC_Stacking_TF2/run_code')\n")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook import path fixed!")
