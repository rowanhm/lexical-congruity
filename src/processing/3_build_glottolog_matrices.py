import json
import math
from pathlib import Path
from pycldf.dataset import Dataset
from tqdm import tqdm
import pandas as pd

with open('bin/languages.json', 'r') as file:
    data = json.load(file)
    PREDEFINED_LANGUAGES = list(data.keys())

# --- SCRIPT CONSTANTS ---
GLOTTOLOG_DIR = Path('data/glottolog-cldf-5.2.1')
# Note: The CLDF data is often in a 'cldf' subdirectory within the package
CLDF_METADATA_FILE = GLOTTOLOG_DIR / 'cldf' / 'cldf-metadata.json'
OUTPUT_DIR = Path('bin/matrices')

EARTH_RADIUS_KM = 6371  # Radius of Earth for Haversine calculation


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees).
    """
    # Handle missing coordinates
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return math.nan

    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_RADIUS_KM * c

    return distance


def calculate_patristic_distance(lineage1, lineage2):
    """
    Computes the patristic distance between two languages based on their
    Glottolog lineage.

    This is the sum of the steps from each language up to their
    most recent common ancestor (MRCA).

    A lower number means more closely related (e.g., sisters = 2).
    """
    if not lineage1 or not lineage2:
        return math.nan  # Cannot compute distance

    l1_parts = lineage1.split('/')
    l2_parts = lineage2.split('/')

    depth_l1 = len(l1_parts)
    depth_l2 = len(l2_parts)

    # Find depth of Most Recent Common Ancestor (MRCA)
    depth_mrca = 0
    for i in range(min(depth_l1, depth_l2)):
        if l1_parts[i] == l2_parts[i]:
            depth_mrca += 1
        else:
            break  # Diverged

    if depth_mrca == 0:
        # No common ancestor found in lineages (e.g., different top-level families)
        # Return NaN or a suitably large number, depending on preference.
        # Using NaN as it's undefined.
        return math.nan

    # Patristic distance = (L1 -> MRCA) + (L2 -> MRCA)
    dist_l1_to_mrca = depth_l1 - depth_mrca
    dist_l2_to_mrca = depth_l2 - depth_mrca

    return dist_l1_to_mrca + dist_l2_to_mrca


def main():
    """
    Main function to load data, compute matrices, and save files.
    """
    if not PREDEFINED_LANGUAGES:
        print("Error: The 'PREDEFINED_LANGUAGES' list is empty.")
        print("Please add Glottocodes to the list at the top of the script.")
        return

    if not CLDF_METADATA_FILE.exists():
        print(f"Error: Could not find CLDF metadata file at:")
        print(f"{CLDF_METADATA_FILE.resolve()}")
        print("Please check the 'GLOTTOLOG_DIR' path.")
        return

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Glottolog Data ---
    print(f"Loading Glottolog data from {CLDF_METADATA_FILE}...")
    try:
        ds = Dataset.from_metadata(CLDF_METADATA_FILE)
        languages_table = list(ds.iter_rows('LanguageTable'))
    except Exception as e:
        print(f"Error loading CLDF dataset: {e}")
        print("Please ensure 'pycldf' is installed (`pip install pycldf`).")
        return

    # Create a lookup dictionary for the languages we care about
    lang_data = {}
    predefined_set = set(PREDEFINED_LANGUAGES)

    for lang in languages_table:
        gcode = lang.get('Glottocode')
        if gcode in predefined_set:
            lang_data[gcode] = {
                'Latitude': lang.get('Latitude'),
                'Longitude': lang.get('Longitude'),
                'Lineage': lang.get('Lineage')  # Assumes 'Lineage' column exists
            }

    # Check for missing languages
    found_glottocodes = sorted(list(lang_data.keys()))
    missing = predefined_set - set(found_glottocodes)
    if missing:
        print(f"Warning: Could not find data for {len(missing)} language(s):")
        for gcode in missing:
            print(f"  - {gcode}")

    if not found_glottocodes:
        print("No valid languages found. Exiting.")
        return

    print(f"Found data for {len(found_glottocodes)} languages.")

    # --- 2. Initialize Matrices (as Pandas DataFrames) ---
    df_distance = pd.DataFrame(
        index=found_glottocodes,
        columns=found_glottocodes,
        dtype=float
    )
    df_patristic_dist = pd.DataFrame(
        index=found_glottocodes,
        columns=found_glottocodes,
        dtype=float  # Changed to float to accommodate NaN
    )

    # --- 3. Compute Metrics ---
    print("Computing distance and relatedness matrices...")
    for i, gcode1 in enumerate(tqdm(found_glottocodes, desc="Computing matrices")):
        for j, gcode2 in enumerate(found_glottocodes):
            # Optimization: matrix is symmetric, only compute half
            if i > j:
                continue

            lang1 = lang_data[gcode1]
            lang2 = lang_data[gcode2]

            # (1) Physical Distance
            dist = haversine_distance(
                lang1['Latitude'], lang1['Longitude'],
                lang2['Latitude'], lang2['Longitude']
            )
            df_distance.loc[gcode1, gcode2] = dist
            df_distance.loc[gcode2, gcode1] = dist

            # (2) Relatedness (Patristic Distance)
            related = calculate_patristic_distance(
                lang1['Lineage'],
                lang2['Lineage']
            )
            df_patristic_dist.loc[gcode1, gcode2] = related
            df_patristic_dist.loc[gcode2, gcode1] = related

    # --- 4. Save to Files ---
    dist_file = OUTPUT_DIR / 'physical_distance.csv'
    rel_file = OUTPUT_DIR / 'patristic_distance.csv'  # Changed filename

    print(f"Saving physical distance matrix to: {dist_file}")
    df_distance.to_csv(dist_file)

    print(f"Saving patristic distance matrix to: {rel_file}")  # Updated text
    df_patristic_dist.to_csv(rel_file)  # Changed dataframe

    print("\nDone.")


if __name__ == "__main__":
    main()