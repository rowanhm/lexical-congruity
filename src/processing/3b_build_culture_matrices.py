import csv

import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = 'data/hofstede_country_scores.csv'
OUTPUT_DIR = 'bin/matrices'
METRICS = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
HOFSTEDE_RANGE = 100.0

# TODO Correct this and apply it to the matrix
countries_dict = {
    "Albania": "alba1267",
    "Algeria": "alge1239",
    "Angola": "port1283",
    "Argentina": "stan1288",
    "Armenia": "nucl1235",
    "Australia": "stan1293",
    "Austria": "stan1295",
    "Azerbaijan": "nort2697",
    "Bangladesh": "beng1280",
    "Belarus": "bela1254",
    "Belgium": "dutc1256",
    "Bhutan": "dzon1239",
    "Bolivia": "stan1288",
    "Bosnia and herzegovina": "bosn1245",
    "Brazil": "port1283",
    "Bulgaria": "bulg1262",
    "Burkina faso": "moss1236",
    "Canada": "stan1293",
    "Cape verde": "kabu1256",
    "Chile": "stan1288",
    "China": "mand1415",
    "Colombia": "stan1288",
    "Costa rica": "stan1288",
    "Croatia": "croa1245",
    "Czech republic": "czec1258",
    "Denmark": "dani1285",
    "Dominican republic": "stan1288",
    "Ecuador": "stan1288",
    "Egypt": "egyp1253",
    "El salvador": "stan1288",
    "Estonia": "esto1258",
    "Ethiopia": "amha1245",
    "Fiji": "fiji1243",
    "Finland": "finn1318",
    "France": "stan1290",
    "Georgia": "nucl1302",
    "Germany": "stan1295",
    "Ghana": "akan1250",
    "Greece": "mode1248",
    "Guatemala": "stan1288",
    "Honduras": "stan1288",
    "Hong kong": "cant1236",
    "Hungary": "hung1274",
    "Iceland": "icel1247",
    "India": "hind1269",
    "Indonesia": "indo1316",
    "Iran": "west2369",
    "Iraq": "gili1239",
    "Ireland": "iris1253",
    "Israel": "mode1271",
    "Italy": "ital1282",
    "Jamaica": "jama1262",
    "Japan": "nucl1643",
    "Jordan": "sout3123",
    "Kazakhstan": "kaza1248",
    "Kenya": "swah1253",
    "Kuwait": "gulf1241",
    "Latvia": "latv1249",
    "Lebanon": "nort3139",
    "Libya": "liby1240",
    "Lithuania": "lith1251",
    "Luxembourg": "luxe1243",
    "Malawi": "nyan1308",
    "Malaysia": "stan1306",
    "Malta": "malt1254",
    "Mexico": "stan1288",
    "Moldova": "roma1327",
    "Mongolia": "halh1238",
    "Montenegro": "mont1282",
    "Morocco": "moro1292",
    "Mozambique": "port1283",
    "Namibia": "stan1293",
    "Nepal": "nepa1254",
    "Netherlands": "dutc1256",
    "New zealand": "stan1293",
    "Nigeria": "nige1257",
    "North macedonia": "mace1251",
    "Norway": "norw1258",
    "Pakistan": "urdu1245",
    "Panama": "stan1288",
    "Paraguay": "guar1248",
    "Peru": "stan1288",
    "Philippines": "taga1270",
    "Poland": "poli1260",
    "Portugal": "port1283",
    "Puerto rico": "stan1288",
    "Qatar": "gulf1241",
    "Romania": "roma1327",
    "Russia": "russ1263",
    "São tomé and príncipe": "saot1240",
    "Saudi arabia": "najd1235",
    "Senegal": "wolo1247",
    "Serbia": "serb1264",
    "Sierra leone": "krio1253",
    "Singapore": "stan1293",
    "Slovakia": "slov1268",
    "Slovenia": "slov1269",
    "South africa": "zulu1248",
    "South korea": "kore1280",
    "Spain": "stan1288",
    "Sri lanka": "sinl1252",
    "Suriname": "dutc1256",
    "Sweden": "swed1254",
    "Switzerland": "stan1295",
    "Syria": "nort3139",
    "Taiwan": "mand1415",
    "Tanzania": "swah1253",
    "Thailand": "thai1261",
    "Trinidad and tobago": "trin1278",
    "Tunisia": "tuni1253",
    "Turkey": "turk1311",
    "Ukraine": "ukra1253",
    "United arab emirates": "gulf1241",
    "United kingdom": "stan1293",
    "United states": "stan1288",
    "Uruguay": "stan1288",
    "Venezuela": "stan1288",
    "Vietnam": "viet1252",
    "Zambia": "stan1293"
}

# --- Main Script ---
def main():
    # 1. Setup: Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_FILE, index_col='country', na_values=['', 'null', 'NULL'])
    except Exception as e:
        print(f"Error reading CSV file '{INPUT_FILE}': {e}")
        return

    # Ensure all metric columns are numeric, coercing errors (blanks -> NaN)
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Metric column '{col}' not found in CSV. Skipping.")

    countries = df.index.tolist()

    # 3. Create Normalized DataFrame (Scores from 0.0 to 1.0)
    df_normalized = df[METRICS] / HOFSTEDE_RANGE

    # 4. Generate Per-Metric Difference Matrices (Vectorized)
    print("Generating per-metric difference matrices...")
    for metric in METRICS:
        if metric not in df_normalized.columns:
            continue

        # Extract column as a (N, 1) numpy array
        col_data = df_normalized[[metric]].values

        # Use numpy broadcasting (N, 1) - (1, N) to get an (N, N) matrix
        # np.abs(NaN - 0.5) correctly results in NaN.
        diff_matrix_np = np.abs(col_data - col_data.T)

        # Convert back to DataFrame
        diff_matrix = pd.DataFrame(diff_matrix_np, index=countries, columns=countries)

        # Save the matrix to CSV
        output_path = os.path.join(OUTPUT_DIR, f'hofstede_{metric}_difference.csv')
        diff_matrix.to_csv(output_path)

    print("Per-metric matrices saved.")

    # 5. Generate Overall Euclidean Distance Matrix (Vectorized)
    print("Generating overall Euclidean distance matrix...")

    # Get the (N, 6) numpy array of normalized data
    data_norm = df_normalized.values

    # Use broadcasting to create an (N, N, 6) array of differences
    # (N, 1, 6) - (1, N, 6) = (N, N, 6)
    diffs = data_norm[:, np.newaxis, :] - data_norm[np.newaxis, :, :]

    # Square, sum, and sqrt. np.sum propagates NaNs, which is the correct logic
    # (if any metric is missing, the total distance is NaN)
    sum_sq_diffs = np.sum(diffs ** 2, axis=2)
    euc_dist_np = np.sqrt(sum_sq_diffs)

    # Convert to DataFrame
    euclidean_matrix = pd.DataFrame(euc_dist_np, index=countries, columns=countries)

    # Save the final matrix
    output_path = os.path.join(OUTPUT_DIR, 'hofstede_euclidean_difference.csv')
    euclidean_matrix.to_csv(output_path)

    print(f"Euclidean distance matrix saved to {output_path}")
    print("All tasks complete.")


if __name__ == "__main__":
    main()