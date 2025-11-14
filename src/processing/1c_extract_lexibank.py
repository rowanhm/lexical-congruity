import json
import collections
from pathlib import Path
from pycldf.dataset import Dataset

# --- Configuration ---
DATA_REPO_PATH = Path('data/lexibank-lexibank-analysed-e05c0f8')
METADATA_FILE = DATA_REPO_PATH / 'cldf' / 'wordlist-metadata.json'
OUTPUT_DIR = Path('bin/lexica/lexibank')
MIN_WORDS = 1000

def main():
    """
    Loads a Lexibank CLDF dataset and generates a JSON file for each language,
    mapping each form to a set of concepts it represents.
    """

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {METADATA_FILE}")

    try:
        # 1. Load the CLDF dataset
        ds = Dataset.from_metadata(METADATA_FILE)

        # 2. Load reference tables into dicts for fast lookup

        # Get all languages (we'll use this to create the files)
        languages = list(ds['LanguageTable'])
        lang_to_glot = {l['ID']: l['Glottocode'] for l in languages}

        # Create a mapping from Parameter_ID (concept) to its Name
        print("Loading concepts (parameters)...")
        concepts = {
            param['ID']: param['Name']
            for param in ds['ParameterTable']
        }

        # 3. Build the main data structure:
        # { lang_id: { form_string: set(concept_names) } }
        print("Processing all forms...")

        # Use defaultdict for easy and clean creation of nested structures
        # The outer dict's values are dicts, whose values are sets.
        all_language_data = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )

        for form in ds['FormTable']:
            lang_id = lang_to_glot[form['Language_ID']]
            form_value = form['Form']
            param_id = form['Parameter_ID']

            # Look up the concept name
            concept_name = concepts[param_id]

            # Add the concept to the set for that form in that language
            all_language_data[lang_id][form_value].add(concept_name)

        print(f"Finished processing forms.")

        # 4. Save each language's data to its own JSON file
        print(f"Saving {len(languages)} language files to {OUTPUT_DIR}...")

        saved_count = 0
        skipped_count = 0
        for glot_code in lang_to_glot.values():

            # Get the data for this language
            # Use .get() in case a language has no forms
            lang_data = all_language_data.get(glot_code, {})

            if not lang_data:
                # Skip languages with no forms if we don't want empty files
                continue

            if glot_code == 'song1313' or glot_code == 'lait1239':
                # Retired codes
                continue

            # Convert sets to lists for JSON serialization
            serializable_data = {
                form: list(concept_set) for form, concept_set in lang_data.items()
            }

            if len(serializable_data) >= MIN_WORDS:

                # Define the output file path
                output_file = OUTPUT_DIR / f"{glot_code}.json"

                # Write the JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)

                saved_count += 1

            else:
                skipped_count += 1

        print(f"Successfully saved {saved_count} languages ({skipped_count} with <{MIN_WORDS} words skipped).")

    except FileNotFoundError:
        print(f"Error: Could not find metadata file at '{METADATA_FILE}'")
        print("Please check that your 'DATA_REPO_PATH' is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()