import csv
import glob
import json
import os

GLOTTO_PATH = 'data/glottolog-cldf-5.2.1/cldf/languages.csv'

name_dict = {}
with open(GLOTTO_PATH, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        glottocode = row.get('ID')
        name = row.get('Name')
        name_dict[glottocode] = name

files = glob.glob('bin/lexica/*/*.json')
file_set = {os.path.basename(f).removesuffix('.json') for f in files}
result_dict = {f: name_dict[f] for f in file_set}

with open('bin/languages.json', 'w') as f:
    json.dump(result_dict, f, indent=2)

print(f"Processed {len(result_dict)} files and saved to output.json")