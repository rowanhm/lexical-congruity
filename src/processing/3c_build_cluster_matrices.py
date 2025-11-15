import json
import math
import os
import glob
import itertools
import csv
import concurrent.futures
import sys  # For flushing output

import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm

# --- Configuration ---
LEXICA_DIR = "bin/lexica"
OUTPUT_DIR = "bin/matrices"
METRICS = ["omega", "nmi", "f1", "jaccard"]


# --- Optimized Metric Computation (Vectorized) ---
# (Helpers _xlogx, _entropy, _compute_nmi, _build_membership_matrix are unchanged)
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
    n_items = len(item_map)
    n_clusters = len(sets_list)
    if n_clusters == 0:
        return csc_matrix((n_items, 0), dtype=np.int8)
    data, row_indices, col_indices = [], [], []
    for j, cluster in enumerate(sets_list):
        for item in cluster:
            i = item_map.get(item)
            if i is not None:
                data.append(1)
                row_indices.append(i)
                col_indices.append(j)
    return csc_matrix((data, (row_indices, col_indices)),
                      shape=(n_items, n_clusters), dtype=np.int8)


# --- REVISED: Now uses fast sparse intersection ---
def compute_all_metrics_from_matrices(M1, M2, n_items):
    """
    Computes all metrics from two PRE-COMPUTED sparse matrices
    using optimized sparse-only operations.
    """
    results = {}

    # === 1. Compute Pair-Based Metrics (Omega, F1, Jaccard) ===

    # C1_co = (items x items) sparse co-occurrence matrix for M1
    C1_co = M1.dot(M1.T)
    # C2_co = (items x items) sparse co-occurrence matrix for M2
    C2_co = M2.dot(M2.T)

    # 'a' = pairs in both (intersection)
    # C_common = element-wise product
    C_common = C1_co.multiply(C2_co)

    # .sum() sums all elements.
    # We must subtract the diagonal, then divide by 2 to get upper triangle.
    diag_sum_common = C_common.diagonal().sum()
    a = (C_common.sum() - diag_sum_common) / 2.0

    # 'a+b' = pairs in C1
    diag_sum_c1 = C1_co.diagonal().sum()
    a_plus_b = (C1_co.sum() - diag_sum_c1) / 2.0

    # 'a+c' = pairs in C2
    diag_sum_c2 = C2_co.diagonal().sum()
    a_plus_c = (C2_co.sum() - diag_sum_c2) / 2.0

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


# --- Helper Functions for Parallelism (Unchanged) ---
def _load_language_file(lang_file):
    lang_code = os.path.basename(lang_file).split(".")[0]
    try:
        with open(lang_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            sets_list = [set(v) for v in data.values()]
            return lang_code, sets_list
    except Exception as e:
        print(f"  Error loading {lang_file}: {e}")
        return lang_code, None


def _compute_pair(task_data):
    lang1, lang2, M1, M2, n_items = task_data
    try:
        results = compute_all_metrics_from_matrices(M1, M2, n_items)
    except Exception as e:
        print(f"Error computing metrics for {lang1}-{lang2}: {e}")
        results = {metric: math.nan for metric in METRICS}  # Use NaN for errors
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

        # --- 2. Load all language data (Parallel I/O) ---
        language_data = {}
        lang_files = glob.glob(os.path.join(resource_path, "*.json"))

        print(f"  Loading {len(lang_files)} language files...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_load_language_file, f) for f in lang_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="  Loading data"):
                lang_code, sets_list = future.result()
                if sets_list is not None:
                    language_data[lang_code] = sets_list

        if len(language_data) < 2:
            print(f"  Not enough language data to compare for {resource_name}. Skipping.")
            continue

        # --- 3. Pre-computation Step (Unchanged) ---
        print("  Building global item universe...")
        global_universe = set()
        for sets_list in language_data.values():
            for s in sets_list: global_universe.update(s)

        n_items = len(global_universe)
        if n_items == 0:
            print("  No items found in resource. Skipping.")
            continue

        global_item_map = {item: i for i, item in enumerate(global_universe)}

        print("  Pre-computing all language matrices...")
        language_matrices = {}
        for lang_code, sets_list in tqdm(language_data.items(), desc="  Building matrices"):
            language_matrices[lang_code] = _build_membership_matrix(sets_list, global_item_map)

        del language_data, global_item_map  # Free memory

        # --- 4. Prepare result matrices ---
        languages = sorted(language_matrices.keys())
        all_results = {}
        for metric_name in METRICS:
            # --- MODIFIED: Initialize with None for "uncomputed" ---
            all_results[metric_name] = {
                lang: {inner_lang: None for inner_lang in languages}
                for lang in languages
            }

        # --- NEW: 5. Load Existing Results (Checkpointing) ---
        print("  Loading existing results for checkpointing...")
        num_existing = 0
        for metric_name in METRICS:
            output_filename = os.path.join(OUTPUT_DIR, f"{resource_name}_{metric_name}.csv")
            if os.path.exists(output_filename):
                try:
                    with open(output_filename, "r", newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        header = next(reader)[1:]  # Get lang codes from header
                        lang_map = {lang: idx for idx, lang in enumerate(header)}

                        for row in reader:
                            lang_row = row[0]
                            if lang_row not in languages: continue

                            for lang_col, idx in lang_map.items():
                                if lang_col not in languages: continue

                                value_str = row[idx + 1]
                                if value_str:  # Not empty string
                                    try:
                                        all_results[metric_name][lang_row][lang_col] = float(value_str)
                                        if lang_row != lang_col:
                                            num_existing += 1
                                    except ValueError:
                                        pass  # Keep as None if parse fails
                except Exception as e:
                    print(f"  Warning: Could not load {output_filename}. Recomputing. Error: {e}")

        # --- 6. Set diagonal ---
        for metric_name in METRICS:
            for lang in languages:
                all_results[metric_name][lang][lang] = 1.0

        # --- 7. Compute metrics (PARALLELIZED) ---
        total_pairs = math.comb(len(languages), 2)
        tasks = []

        # --- MODIFIED: Only add tasks that are uncomputed ---
        for lang1, lang2 in itertools.combinations(languages, 2):
            if all_results["omega"][lang1][lang2] is None:  # Check one metric
                tasks.append((lang1, lang2,
                              language_matrices[lang1],
                              language_matrices[lang2],
                              n_items))

        num_found = total_pairs - len(tasks)
        print(f"  Found {num_found} / {total_pairs} existing results.")

        if not tasks:
            print("  All comparisons are already computed. Skipping.")
        else:
            print(f"  Preparing {len(tasks)} new comparison tasks...")
            sys.stdout.flush()  # Force print before tqdm
            # Use ProcessPoolExecutor for CPU-bound tasks
            max_cores_to_use = max(1, os.cpu_count() - 2)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores_to_use) as executor:
                future_to_task = {executor.submit(_compute_pair, task): task for task in tasks}

                for future in tqdm(concurrent.futures.as_completed(future_to_task),
                                   total=len(tasks),
                                   desc="  Processing metrics"):

                    lang1, lang2, pair_results = future.result()

                    for metric_name, value in pair_results.items():
                        if metric_name in all_results:
                            all_results[metric_name][lang1][lang2] = value
                            all_results[metric_name][lang2][lang1] = value

        # --- 8. Save matrices (as before) ---
        print("  Saving matrices...")
        for metric_name, results_matrix in all_results.items():
            output_filename = os.path.join(OUTPUT_DIR, f"{resource_name}_{metric_name}.csv")
            with open(output_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([""] + languages)
                for lang_row in languages:
                    row_data = [
                        results_matrix[lang_row][lang_col]
                        for lang_col in languages
                    ]
                    # Write row, saving None as an empty string
                    writer.writerow([lang_row] + ["" if v is None else v for v in row_data])

    print("Processing complete.")


if __name__ == "__main__":
    main()