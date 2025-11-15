import json
import math
import os
import glob
import itertools
import csv
import concurrent.futures

import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm

# --- Configuration ---
LEXICA_DIR = "bin/lexica"
OUTPUT_DIR = "bin/matrices"
METRICS = ["omega", "nmi", "f1", "jaccard"]


# --- Optimized Metric Computation (Vectorized) ---
# (These functions, _xlogx, _entropy, _compute_nmi, are unchanged)
def _xlogx(v):
    if v == 0: return 0.0
    return v * np.log2(v)


def _entropy(p):
    return -_xlogx(p) - _xlogx(1 - p)


def _compute_nmi(cm, sizes1, sizes2, n_items):
    if n_items == 0 or len(sizes1) == 0 or len(sizes2) == 0:
        return 0.0
    p1 = sizes1 / n_items
    p2 = sizes2 / n_items
    h_x_vec = np.vectorize(_entropy)(p1)
    h_y_vec = np.vectorize(_entropy)(p2)
    H_X = h_x_vec.sum()
    H_Y = h_y_vec.sum()
    if H_X == 0 or H_Y == 0:
        return 0.0
    sizes1_col = sizes1.reshape(-1, 1)
    sizes2_row = sizes2.reshape(1, -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_x_given_y = np.nan_to_num(cm / sizes2_row)
        h_x_given_y_matrix = np.vectorize(_entropy)(p_x_given_y)
        p_y_given_x = np.nan_to_num(cm / sizes1_col)
        h_y_given_x_matrix = np.vectorize(_entropy)(p_y_given_x)
    H_X_given_Y = np.sum((p2 * np.sum(h_x_given_y_matrix, axis=0)))
    H_Y_given_X = np.sum((p1 * np.sum(h_y_given_x_matrix, axis=1)))
    I_XY = (H_X - H_X_given_Y + H_Y - H_Y_given_X) / 2.0
    nmi = I_XY / max(np.sqrt(H_X * H_Y), 1e-15)
    return nmi


def _build_membership_matrix(sets_list, item_map):
    """
    Builds a sparse binary membership matrix (CSC format).
    Rows = items, Cols = clusters
    """
    n_items = len(item_map)
    n_clusters = len(sets_list)
    if n_clusters == 0:
        return csc_matrix((n_items, 0), dtype=np.int8)
    data, row_indices, col_indices = [], [], []
    for j, cluster in enumerate(sets_list):
        for item in cluster:
            i = item_map.get(item)  # Get index from GLOBAL map
            if i is not None:
                data.append(1)
                row_indices.append(i)
                col_indices.append(j)
    return csc_matrix((data, (row_indices, col_indices)),
                      shape=(n_items, n_clusters), dtype=np.int8)


# --- NEW: Refactored function to accept pre-built matrices ---
def compute_all_metrics_from_matrices(M1, M2, n_items):
    """
    Computes all metrics from two PRE-COMPUTED sparse matrices.
    """
    results = {}

    # === 1. Compute Pair-Based Metrics (Omega, F1, Jaccard) ===
    C1_co = M1.dot(M1.T).tocoo()
    c1_pairs = set((i, j) for i, j in zip(C1_co.row, C1_co.col) if i < j)

    C2_co = M2.dot(M2.T).tocoo()
    c2_pairs = set((i, j) for i, j in zip(C2_co.row, C2_co.col) if i < j)

    a = len(c1_pairs.intersection(c2_pairs))
    a_plus_b = len(c1_pairs)
    a_plus_c = len(c2_pairs)
    b = a_plus_b - a
    c = a_plus_c - a
    union = a + b + c

    denominator_f1 = (2 * a) + b + c
    results["f1"] = (2 * a) / denominator_f1 if denominator_f1 > 0 else 1.0

    jaccard = a / union if union > 0 else 1.0
    results["jaccard"] = jaccard
    results["omega"] = jaccard

    # === 2. Compute NMI (from Confusion Matrix) ===
    cm = M1.T.dot(M2).toarray()
    sizes1 = M1.sum(axis=0).A1
    sizes2 = M2.sum(axis=0).A1

    results["nmi"] = _compute_nmi(cm, sizes1, sizes2, n_items)
    return results


# --- Helper Functions for Parallelism ---

def _load_language_file(lang_file):
    """
    Worker function to load a single JSON file (for parallel I/O).
    """
    lang_code = os.path.basename(lang_file).split(".")[0]
    try:
        with open(lang_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert to sets immediately
            sets_list = [set(v) for v in data.values()]
            return lang_code, sets_list
    except Exception as e:
        print(f"  Error loading {lang_file}: {e}")
        return lang_code, None


def _compute_pair(task_data):
    """
    Worker function to compute all metrics for a single pair of languages.
    NOW receives pre-computed matrices.
    """
    lang1, lang2, M1, M2, n_items = task_data

    try:
        # This one call does all the math!
        results = compute_all_metrics_from_matrices(M1, M2, n_items)

    except Exception as e:
        print(f"Error computing metrics for {lang1}-{lang2}: {e}")
        results = {metric: math.nan for metric in METRICS}

    return (lang1, lang2, results)


# --- Main Script ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resource_paths = glob.glob(os.path.join(LEXICA_DIR, "*"))
    for resource_path in resource_paths:
        if not os.path.isdir(resource_path):
            continue

        if 'lexibank' in resource_path:
            continue

        resource_name = os.path.basename(resource_path)
        print(f"Processing resource: {resource_name}...")

        # --- 2. Load all language data (Now in parallel) ---
        language_data = {}  # Still {lang: list[set]}
        lang_files = glob.glob(os.path.join(resource_path, "*.json"))

        print(f"  Loading {len(lang_files)} language files...")
        # Use ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_load_language_file, f) for f in lang_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="  Loading data"):
                lang_code, sets_list = future.result()
                if sets_list is not None:
                    language_data[lang_code] = sets_list

        if len(language_data) < 2:
            print(f"  Not enough language data to compare for {resource_name}. Skipping.")
            continue

        # --- NEW: 3. Pre-computation Step ---
        print("  Building global item universe...")
        global_universe = set()
        for sets_list in language_data.values():
            for s in sets_list:
                global_universe.update(s)

        n_items = len(global_universe)
        if n_items == 0:
            print("  No items found in resource. Skipping.")
            continue

        global_item_map = {item: i for i, item in enumerate(global_universe)}

        print("  Pre-computing all language matrices...")
        language_matrices = {}  # Will hold {lang: csc_matrix}
        for lang_code, sets_list in tqdm(language_data.items(), desc="  Building matrices"):
            language_matrices[lang_code] = _build_membership_matrix(sets_list, global_item_map)

        # Clear large intermediate data
        del language_data
        del global_item_map

        # --- 4. Prepare result matrices (as before) ---
        languages = sorted(language_matrices.keys())
        all_results = {}
        for metric_name in METRICS:
            all_results[metric_name] = {
                lang: {inner_lang: 0.0 for inner_lang in languages}
                for lang in languages
            }

        # --- 5. Compute metrics (PARALLELIZED) ---
        num_pairs = math.comb(len(languages), 2)
        print(f"  Preparing {num_pairs} comparison tasks...")
        tasks = []
        for lang1, lang2 in itertools.combinations(languages, 2):
            # Pass the PRE-COMPUTED matrices to the worker
            tasks.append((lang1, lang2,
                          language_matrices[lang1],
                          language_matrices[lang2],
                          n_items))

        if not tasks:
            print("  No language pairs to compute.")
            continue

        # Use ProcessPoolExecutor for CPU-bound tasks (as before)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_task = {executor.submit(_compute_pair, task): task for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task),
                               total=len(tasks),
                               desc="  Processing metrics"):

                lang1, lang2, pair_results = future.result()

                for metric_name, value in pair_results.items():
                    if metric_name in all_results:
                        all_results[metric_name][lang1][lang2] = value
                        all_results[metric_name][lang2][lang1] = value

        # --- 6. Set diagonal (as before) ---
        for metric_name in METRICS:
            for lang in languages:
                all_results[metric_name][lang][lang] = 1.0

        # --- 7. Save matrices (as before) ---
        print("  Saving matrices...")
        for metric_name, results_matrix in all_results.items():
            output_filename = os.path.join(OUTPUT_DIR, f"{resource_name}_{metric_name}.csv")
            with open(output_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([""] + languages)
                for lang_row in languages:
                    row_data = [results_matrix[lang_row][lang_col] for lang_col in languages]
                    writer.writerow([lang_row] + row_data)

    print("Processing complete.")


if __name__ == "__main__":
    main()