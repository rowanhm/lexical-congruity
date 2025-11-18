import csv

import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = 'data/hofstede_country_scores.csv'
OUTPUT_DIR = 'bin/matrices'
METRICS = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
HOFSTEDE_RANGE = 100.0

COUNTRIES_TO_LANGS = {
    "Albania": "gheg1238",
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
    "Iraq": "jude1266",
    "Ireland": "iris1253",
    "Israel": "hebr1245",
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
    "North macedonia": "mace1250",
    "Norway": "norw1258",
    "Pakistan": "urdu1245",
    "Panama": "stan1288",
    "Paraguay": "stan1288",
    "Peru": "stan1288",
    "Philippines": "taga1270",
    "Poland": "poli1260",
    "Portugal": "port1283",
    "Puerto rico": "stan1288",
    "Qatar": "gulf1241",
    "Romania": "roma1327",
    "Russia": "russ1263",
    "São tomé and príncipe": "port1283",
    "Saudi arabia": "najd1235",
    "Senegal": "wolo1247",
    "Serbia": "serb1264",
    "Sierra leone": "krio1253",
    "Singapore": "stan1293",
    "Slovakia": "slov1269",
    "Slovenia": "slov1268",
    "South africa": "zulu1248",
    "South korea": "kore1280",
    "Spain": "stan1288",
    "Sri lanka": "sinh1246",
    "Suriname": "dutc1256",
    "Sweden": "swed1254",
    "Switzerland": "swis1247",
    "Syria": "nort3139",
    "Taiwan": "mand1415",
    "Tanzania": "swah1253",
    "Thailand": "thai1261",
    "Trinidad and tobago": "trin1274",
    "Tunisia": "tuni1259",
    "Turkey": "nucl1301",
    "Ukraine": "ukra1253",
    "United arab emirates": "gulf1241",
    "United kingdom": "stan1293",
    "United states": "stan1293",
    "Uruguay": "stan1288",
    "Venezuela": "stan1288",
    "Vietnam": "viet1252",
    "Zambia": "stan1293"
}

COUNTRIES_TO_EXCLUDE = {
    "Argentina", "Bolivia", "Chile", "Colombia", "Costa rica",
    "Dominican republic", "Ecuador", "El salvador", "Guatemala", "Honduras",
    "Mexico", "Panama", "Paraguay", "Peru", "Puerto rico", "Uruguay", "Venezuela",
    "Angola", "Brazil", "Mozambique", "São tomé and príncipe", "Cape verde",
    "Australia", "Canada", "Fiji", "Ghana", "Jamaica", "Kenya", "Malawi",
    "Namibia", "New zealand", "Nigeria", "Sierra leone", "Singapore",
    "South africa", "Tanzania", "Trinidad and tobago", "Zambia", "United states", "Belgium",
    "Switzerland", "Suriname", "Burkina faso", "Senegal"
}


# --- Main Script ---
def main():
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_FILE, index_col='country', na_values=['', 'null', 'NULL'])
    except Exception as e:
        print(f"Error reading CSV file '{INPUT_FILE}': {e}")
        return

    # 3. --- (Req 1): Filter out colonial-context countries ---
    original_count = len(df)
    df = df[~df.index.isin(COUNTRIES_TO_EXCLUDE)]
    dropped_count = original_count - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} specified colonial-context countries.")

    # 4. Process Data
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Metric column '{col}' not found in CSV. Skipping.")

    # 5. --- (Req 2): Aggregation Step ---
    grouping_keys = [COUNTRIES_TO_LANGS[country] for country in df.index]
    df['lang_code'] = grouping_keys

    # Calculate mean per metric (respects Req 2)
    df_agg = df.groupby('lang_code')[METRICS].mean()

    # 6. Normalize
    # This DataFrame will contain NaNs for languages with missing metrics
    df_normalized = df_agg / HOFSTEDE_RANGE
    print(f"Aggregated data into {len(df_normalized)} language groups.")

    # 7. --- (Req 3 Corrected): Generate Per-Metric Difference Matrices ---
    print("Generating per-metric difference matrices...")
    for metric in METRICS:
        if metric not in df_normalized.columns:
            continue

        # Select the column for this metric AND drop NaNs *for this metric only*
        metric_data_series = df_normalized[metric].dropna()

        # Get the list of languages that have data *for this metric*
        metric_countries = metric_data_series.index.tolist()

        if not metric_countries:
            print(f"  Skipping '{metric}' matrix: No language groups have data.")
            continue

        # Get data as a (N, 1) numpy array
        col_data = metric_data_series.to_frame().values

        # Calculate difference matrix
        diff_matrix_np = np.abs(col_data - col_data.T)
        diff_matrix = pd.DataFrame(diff_matrix_np, index=metric_countries, columns=metric_countries)

        # Save
        output_path = os.path.join(OUTPUT_DIR, f'hofstede_{metric}.csv')
        diff_matrix.to_csv(output_path)
        print(f"  Saved '{metric}' matrix with {len(metric_countries)} languages.")

    print("Per-metric matrices saved.")

    # 8. --- (Req 3 Corrected): Generate Overall Euclidean Distance Matrix ---
    print("Generating overall Euclidean distance matrix...")

    # *Now*, create a new DataFrame by dropping rows with *any* missing metrics
    df_euclidean = df_normalized.dropna(subset=METRICS, how='any')

    euclidean_countries = df_euclidean.index.tolist()

    if not euclidean_countries:
        print("  Skipping Euclidean matrix: No language groups have complete (6/6) data.")
    else:
        # Get the (N, 6) numpy array (guaranteed to be free of NaNs)
        data_norm = df_euclidean.values

        # Calculate Euclidean distance
        diffs = data_norm[:, np.newaxis, :] - data_norm[np.newaxis, :, :]
        sum_sq_diffs = np.sum(diffs ** 2, axis=2)
        euc_dist_np = np.sqrt(sum_sq_diffs)

        euclidean_matrix = pd.DataFrame(euc_dist_np, index=euclidean_countries, columns=euclidean_countries)

        # Save
        output_path = os.path.join(OUTPUT_DIR, 'hofstede_avg.csv')
        euclidean_matrix.to_csv(output_path)
        print(f"  Euclidean distance matrix saved with {len(euclidean_countries)} languages.")

    print("All tasks complete.")


if __name__ == "__main__":
    main()