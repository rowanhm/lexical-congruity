import csv
import xml.etree.ElementTree as ET
import json
import os
import collections

INPUT_FILE = '../cygnet/cygnet.xml'
OUTPUT_DIR = 'bin/lexica/cygnet'
CONCEPT_PREFIX_FILTER = 'cili'
GLOTTO_PATH = 'data/glottolog-cldf-5.2.1/cldf/languages.csv'

ISO_639_1_TO_3 = {
    'ab': 'abk', 'aa': 'aar', 'af': 'afr', 'ak': 'aka', 'sq': 'sqi', 'am': 'amh',
    'ar': 'ara', 'an': 'arg', 'hy': 'hye', 'as': 'asm', 'av': 'ava', 'ae': 'ave',
    'ay': 'aym', 'az': 'aze', 'bm': 'bam', 'ba': 'bak', 'eu': 'eus', 'be': 'bel',
    'bn': 'ben', 'bi': 'bis', 'bs': 'bos', 'br': 'bre', 'bg': 'bul', 'my': 'mya',
    'ca': 'cat', 'ch': 'cha', 'ce': 'che', 'ny': 'nya', 'zh': 'zho', 'cv': 'chv',
    'kw': 'cor', 'co': 'cos', 'cr': 'cre', 'hr': 'hrv', 'cs': 'ces', 'da': 'dan',
    'dv': 'div', 'nl': 'nld', 'dz': 'dzo', 'en': 'eng', 'eo': 'epo', 'et': 'est',
    'ee': 'ewe', 'fo': 'fao', 'fj': 'fij', 'fi': 'fin', 'fr': 'fra', 'ff': 'ful',
    'gl': 'glg', 'ka': 'kat', 'de': 'deu', 'el': 'ell', 'gn': 'grn', 'gu': 'guj',
    'ht': 'hat', 'ha': 'hau', 'he': 'heb', 'hz': 'her', 'hi': 'hin', 'ho': 'hmo',
    'hu': 'hun', 'ia': 'ina', 'id': 'ind', 'ie': 'ile', 'ga': 'gle', 'ig': 'ibo',
    'ik': 'ipk', 'io': 'ido', 'is': 'isl', 'it': 'ita', 'iu': 'iku', 'ja': 'jpn',
    'jv': 'jav', 'kl': 'kal', 'kn': 'kan', 'kr': 'kau', 'ks': 'kas', 'kk': 'kaz',
    'km': 'khm', 'ki': 'kik', 'rw': 'kin', 'ky': 'kir', 'kv': 'kom', 'kg': 'kon',
    'ko': 'kor', 'ku': 'kur', 'kj': 'kua', 'la': 'lat', 'lb': 'ltz', 'lg': 'lug',
    'li': 'lim', 'ln': 'lin', 'lo': 'lao', 'lt': 'lit', 'lu': 'lub', 'lv': 'lav',
    'gv': 'glv', 'mk': 'mkd', 'mg': 'mlg', 'ms': 'msa', 'ml': 'mal', 'mt': 'mlt',
    'mi': 'mri', 'mr': 'mar', 'mh': 'mah', 'mn': 'mon', 'na': 'nau', 'nv': 'nav',
    'nd': 'nde', 'ne': 'nep', 'ng': 'ndo', 'nb': 'nob', 'nn': 'nno', 'no': 'nor',
    'ii': 'iii', 'nr': 'nbl', 'oc': 'oci', 'oj': 'oji', 'cu': 'chu', 'om': 'orm',
    'or': 'ori', 'os': 'oss', 'pa': 'pan', 'pi': 'pli', 'fa': 'fas', 'pl': 'pol',
    'ps': 'pus', 'pt': 'por', 'qu': 'que', 'rm': 'roh', 'rn': 'run', 'ro': 'ron',
    'ru': 'rus', 'sa': 'san', 'sc': 'srd', 'sd': 'snd', 'se': 'sme', 'sm': 'smo',
    'sg': 'sag', 'sr': 'srp', 'gd': 'gla', 'sn': 'sna', 'si': 'sin', 'sk': 'slk',
    'sl': 'slv', 'so': 'som', 'st': 'sot', 'es': 'spa', 'su': 'sun', 'sw': 'swa',
    'ss': 'ssw', 'sv': 'swe', 'ta': 'tam', 'te': 'tel', 'tg': 'tgk', 'th': 'tha',
    'ti': 'tir', 'bo': 'bod', 'tk': 'tuk', 'tl': 'tgl', 'tn': 'tsn', 'to': 'ton',
    'tr': 'tur', 'ts': 'tso', 'tt': 'tat', 'tw': 'twi', 'ty': 'tah', 'ug': 'uig',
    'uk': 'ukr', 'ur': 'urd', 'uz': 'uzb', 've': 'ven', 'vi': 'vie', 'vo': 'vol',
    'wa': 'wln', 'cy': 'cym', 'wo': 'wol', 'fy': 'fry', 'xh': 'xho', 'yi': 'yid',
    'yo': 'yor', 'za': 'zha', 'zu': 'zul'
}

def load_iso_to_glotto_map(csv_path):
    iso_to_glotto = {}

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                glottocode = row.get('ID')
                iso_code = row.get('ISO639P3code')

                # Only add to map if the ISO code exists
                if iso_code:
                    iso_to_glotto[iso_code] = glottocode

    except FileNotFoundError:
        print(f"Error: Could not find file at '{csv_path}'.")
        print("Please check the path and try again.")
        return {}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

    print(f"Loaded {len(iso_to_glotto)} ISO-to-Glottolog mappings.")
    return iso_to_glotto


def get_glottocode(iso_code, iso_map):
    """Converts an ISO code to a Glottocode."""
    iso_code = iso_code.split('-')[0]

    if iso_code in ISO_639_1_TO_3:
        iso_code = ISO_639_1_TO_3[iso_code]

    return iso_map[iso_code]

def extract_wordform_concepts():

    iso_map = load_iso_to_glotto_map(GLOTTO_PATH)

    print(f"Starting extraction from {INPUT_FILE}...")

    try:
        tree = ET.parse(INPUT_FILE)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return

    # --- Step 1: Map Lexeme IDs to their primary wordform and language ---
    # We store {'lexeme_id': {'form': 'word', 'lang': 'en'}}
    lexeme_data = {}

    lexeme_layer = root.find('LexemeLayer')
    if lexeme_layer is None:
        print("Error: Could not find <LexemeLayer> in the XML.")
        return

    print("Processing LexemeLayer...")
    for lexeme in lexeme_layer.findall('Lexeme'):
        lexeme_id = lexeme.get('id')
        language = lexeme.get('language')
        wordform_elem = lexeme.find('Wordform')

        if lexeme_id and language and wordform_elem is not None:
            wordform = wordform_elem.get('form')
            if wordform:
                lexeme_data[lexeme_id] = {'form': wordform, 'lang': language}

    print(f"Found {len(lexeme_data)} valid lexemes.")

    # --- Step 2: Use Senses to map filtered wordforms to concepts by language ---
    # We store {'lang': {'wordform': {'concept_id_1', 'concept_id_2'}}}
    output_data = collections.defaultdict(lambda: collections.defaultdict(set))

    sense_layer = root.find('SenseLayer')
    if sense_layer is None:
        print("Error: Could not find <SenseLayer> in the XML.")
        return

    print(f"Processing SenseLayer with filter: concepts starting with '{CONCEPT_PREFIX_FILTER}'...")
    sense_count = 0
    filtered_count = 0
    for sense in sense_layer.findall('Sense'):
        sense_count += 1
        lexeme_id = sense.get('signifier')
        concept_id = sense.get('signified')

        if not lexeme_id or not concept_id:
            continue

        # APPLY FILTER: Only process concepts beginning with the specified prefix
        if not concept_id.startswith(CONCEPT_PREFIX_FILTER):
            continue

        filtered_count += 1
        data = lexeme_data.get(lexeme_id)

        if data:
            lang = data['lang']
            form = data['form']

            # Add the concept to the set for this wordform
            output_data[lang][form].add(concept_id)

    print(f"Processed {sense_count} senses. {filtered_count} matched the filter.")
    print(f"Data found for languages: {list(output_data.keys())}")

    # --- Step 3: Write the output JSON files ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR}. {e}")
        return

    for lang, data in output_data.items():
        output_filepath = os.path.join(OUTPUT_DIR, f"{get_glottocode(lang, iso_map)}.json")

        # Convert sets to sorted lists for stable, serializable JSON output
        serializable_data = {
            form: sorted(list(concepts))
            for form, concepts in data.items()
        }

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully wrote {len(serializable_data)} entries to {output_filepath}")
        except IOError as e:
            print(f"Error: Could not write to file {output_filepath}. {e}")

    print("\nExtraction complete.")


if __name__ == "__main__":
    extract_wordform_concepts()