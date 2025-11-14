#!/usr/bin/env python3
import csv
import pathlib
import xml.etree.ElementTree as ET
import json
from collections import defaultdict
import sys

INPUT_DIR = pathlib.Path('data/evosem_data_2025-02-24/EvoSem_data')
OUTPUT_DIR = pathlib.Path('bin/lexica/evosem')
GLOTTO_PATH = 'data/glottolog-cldf-5.2.1/cldf/languages.csv'
MIN_WORDS = 500

lang_lookup = {
    "burmese_(written)": "burmese",
    "lushai_[mizo]": "mizo",
    "tibetan_(written)": "tibetan",
    "armenian": "classical-middle_armenian",
    "indonesian": "standard_indonesian",
    "old_javanese": "kawi",
    "azerbaijani": "central_oghuz",
    "ancient_greek": "ionic-attic_ancient_greek",
    "malay": "standard_malay",
    "tongan": "tonga_(tonga_islands)",
    "serbo-croatian": "serbian-croatian-bosnian",
    "hanunóo": "hanunoo",
    "kapampangan": "pampanga",
    "karo_batak": "batak_karo",
    "casiguran_dumagat": "casiguran-nagtipunan_agta",
    "ilokano": "iloko",
    "manobo_(western_bukidnon)": "western_bukidnon_manobo",
    "isneg": "isnag",
    "makassarese": "makasar",
    "toba_batak": "batak_toba",
    "itbayaten": "itbayat",
    "malagasy": "plateau_malagasy",
    "bare'e": "pamona",
    "old_english": "old_english_(ca._450-1100)",
    "luxembourgish": "luxemburgish",
    "old_high_german": "old_high_german_(ca._750-1050)",
    "old_irish": "old_irish_(8-9th_century)",
    "tiddim": "tedim_chin"
}
def load_name_to_glotto_map(csv_path):
    name_to_glotto = {}

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            glottocode = row.get('ID')
            lang_name = row.get('Name').lower().replace(' ', '_')

            if lang_name in name_to_glotto.keys():
                print(f'Warning: {lang_name} contains multiple keys')
            else:
                name_to_glotto[lang_name] = glottocode

    return name_to_glotto
def process_xml_files():
    """
    Finds, parses, and processes all XML files to generate JSON outputs.
    """

    name_to_glotto = load_name_to_glotto_map(GLOTTO_PATH)

    # This is our main in-memory data structure.
    # e.g., {'tur': {'paska': {'hut'}}}
    language_data = defaultdict(lambda: defaultdict(set))

    print(f"Scanning for .xml files in: {INPUT_DIR}")

    if not INPUT_DIR.exists():
        print(f"Error: Input directory does not exist: {INPUT_DIR}", file=sys.stderr)
        return

    # Find all .xml files recursively
    xml_files = list(INPUT_DIR.glob('**/*.xml'))

    if not xml_files:
        print(f"Warning: No .xml files found in {INPUT_DIR}", file=sys.stderr)
        return

    print(f"Found {len(xml_files)} XML files. Starting processing...")

    file_count = 0
    reflex_count = 0

    for file_path in xml_files:
        file_count += 1
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find all <reflex> elements in the current file
            for reflex in root.findall('.//reflex'):

                # Extract language
                lang_elem = reflex.find('lang')
                if lang_elem is None or not lang_elem.text:
                    continue  # Skip reflex if lang is missing

                lang_name = lang_elem.text.strip()

                # Extract form
                form_elem = reflex.find('form')
                if form_elem is None or not form_elem.text:
                    continue  # Skip reflex if form is missing

                form_text = form_elem.text.strip()

                # Get the language code for the output filename
                lang_code = lang_name.lower().replace(' ', '_')

                # Extract concepts
                concepts_set = set()
                for c in reflex.findall('.//concepts/c'):
                    if c.text:
                        concepts_set.add(c.text.strip())

                # Add the concepts to the set for this form
                if concepts_set:
                    language_data[lang_code][form_text].update(concepts_set)
                    reflex_count += 1

        except ET.ParseError:
            print(f"Warning: Skipping malformed XML file: {file_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Unexpected error processing {file_path}: {e}", file=sys.stderr)

    print(f"\nProcessed {file_count} files and {reflex_count} reflexes.")
    print(f"Found data for {len(language_data)} unique language codes.")

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing JSON files to: {OUTPUT_DIR}")

    below_min_count = 0
    no_glotto_count = 0
    saved_count = 0

    saved_files = set()

    for lang_name, forms_map in language_data.items():

        if len(forms_map) < MIN_WORDS:
            below_min_count += 1
            continue

        # Convert some langs to
        if lang_name not in name_to_glotto.keys():
            lang_name = lang_lookup[lang_name]

        if lang_name not in name_to_glotto.keys():
            no_glotto_count += 1
            print(lang_name)
            continue

        lang_code = name_to_glotto[lang_name]

        output_file_path = OUTPUT_DIR / f"{lang_code}.json"
        assert output_file_path not in saved_files
        saved_files.add(output_file_path)

        # Convert the sets to lists for JSON serialization
        # e.g., {'paska': {'hut'}} -> {'paska': ['hut']}
        output_data = {form: list(concepts) for form, concepts in forms_map.items()}

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        saved_count += 1

    print(f"Successfully saved {saved_count} language ({below_min_count} with too few words; {no_glotto_count} failed glotto alignment).")


# --- Run the script ---
if __name__ == "__main__":
    process_xml_files()