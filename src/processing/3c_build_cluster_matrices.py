import json
import math
import os
import glob
import itertools
import csv
import concurrent.futures
import sqlite3  # For the database
import multiprocessing  # For the Manager Queue
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


def compute_all_metrics_from_matrices(M1, M2, n_items):
    # This function is unchanged and uses the fast sparse intersection
    results = {}
    C1_co = M1.dot(M1.T)
    C2_co = M2.dot(M2.T)
    C_common = C1_co.multiply(C2_co)
    diag_sum_common = C_common.diagonal().sum()
    a = (C_common.sum() - diag_sum_common) / 2.0
    diag_sum_c1 = C1_co.diagonal().sum()
    a_plus_b = (C1_co.sum() - diag_sum_c1) / 2.0
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
    cm = M1.T.dot(M2).toarray()
    sizes1 = M1.sum(axis=0).A1
    sizes2 = M2.sum(axis=0).A1
    results["nmi"] = _compute_nmi(cm, sizes1, sizes2, n_items)
    return results


# --- Helper Functions for Parallelism ---

def _load_language_file(lang_file):
    # This is the (fixed) file loader, unchanged
    lang_code = os.path.basename(lang_file).split(".")[0]
    try:
        with open(lang_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            sets_list = [set(v) for v in data.values()]
            return lang_code, sets_list
    except Exception as e:
        print(f"  Error loading {lang_file}: {e}")
        return lang_code, None


# --- MODIFIED: Worker now sends results to a Queue ---
def _compute_pair(task_data):
    """
    Worker function: Computes metrics and PUTS the result into the shared queue.
    """
    lang1, lang2, M1, M2, n_items, queue = task_data
    try:
        results = compute_all_metrics_from_matrices(M1, M2, n_items)
    except Exception as e:
        print(f"Error computing metrics for {lang1}-{lang2}: {e}")
        results = {metric: math.nan for metric in METRICS}

    # Put the result on the queue instead of returning it
    queue.put((lang1, lang2, results))
    return True  # Return a simple success signal


# --- NEW: Database Helper Functions ---

def init_db(conn, languages):
    """
    Initializes the SQLite database, creating tables if they don't exist.
    """
    with conn:  # Creates a transaction
        cursor = conn.cursor()
        # Create table to hold all results
        metric_cols = ", ".join([f"{name} REAL" for name in METRICS])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS results (
                lang1 TEXT,
                lang2 TEXT,
                {metric_cols},
                PRIMARY KEY (lang1, lang2)
            )
        """)
        # Create table to hold the master list of languages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS languages (
                lang TEXT PRIMARY KEY
            )
        """)
        # Add all languages to the languages table
        cursor.executemany(
            "INSERT OR IGNORE INTO languages (lang) VALUES (?)",
            [(lang,) for lang in languages]
        )


def load_done_pairs(conn):
    """
    Reads the DB and returns a set of all (lang1, lang2) pairs already computed.
    """
    done_pairs = set()
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT lang1, lang2 FROM results")
        for row in cursor.fetchall():
            # Sort to handle (A, B) vs (B, A)
            done_pairs.add(tuple(sorted(row)))
    return done_pairs


def save_result_to_db(conn, lang1, lang2, results):
    """
    Saves a single computed result to the database.
    This is crash-safe.
    """
    with conn:  # Creates a transaction
        metric_names = ", ".join(METRICS)
        placeholders = ", ".join(["?"] * len(METRICS))
        values = [results[name] for name in METRICS]

        conn.execute(
            f"INSERT OR REPLACE INTO results (lang1, lang2, {metric_names}) VALUES (?, ?, {placeholders})",
            (lang1, lang2, *values)
        )


def export_to_csv(conn, resource_name, output_dir):
    """
    Reads the complete database and writes all final CSV matrices.
    """
    languages = []
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT lang FROM languages ORDER BY lang")
        languages = [row[0] for row in cursor.fetchall()]

    if not languages:
        print("  No languages found in DB, skipping CSV export.")
        return

    # 1. Initialize empty results dictionaries
    all_results = {}
    for metric_name in METRICS:
        all_results[metric_name] = {
            lang: {inner_lang: None for inner_lang in languages}
            for lang in languages
        }
        # Set diagonal
        for lang in languages:
            all_results[metric_name][lang][lang] = 1.0

    # 2. Load all results from DB
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results")
        for row in cursor.fetchall():
            lang1, lang2 = row[0], row[1]
            for i, metric_name in enumerate(METRICS):
                value = row[i + 2]
                if lang1 in all_results[metric_name] and lang2 in all_results[metric_name][lang1]:
                    all_results[metric_name][lang1][lang2] = value
                    all_results[metric_name][lang2][lang1] = value

    # 3. Write all CSV files (same logic as before)
    for metric_name, results_matrix in all_results.items():
        output_filename = os.path.join(output_dir, f"{resource_name}_{metric_name}.csv")
        with open(output_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + languages)
            for lang_row in languages:
                row_data = [results_matrix[lang_row][lang_col] for lang_col in languages]
                writer.writerow([lang_row] + ["" if v is None else v for v in row_data])


# --- Main Script (Refactored for DB) ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resource_paths = glob.glob(os.path.join(LEXICA_DIR, "*"))
    for resource_path in resource_paths:
        if not os.path.isdir(resource_path):
            continue

        # TODO remove
        if 'lexibank' in resource_path:
            continue

        resource_name = os.path.basename(resource_path)
        print(f"Processing resource: {resource_name}...")

        # --- 2. Load all language data (FIXED: Uses ThreadPoolExecutor) ---
        language_data = {}
        lang_files = glob.glob(os.path.join(resource_path, "*.json"))

        print(f"  Loading {len(lang_files)} language files...")
        # Use ThreadPoolExecutor for I/O-bound tasks (this is the correct pool)
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

        # --- NEW: 4. Database & Task Setup ---
        languages = sorted(language_matrices.keys())
        db_path = os.path.join(OUTPUT_DIR, f"{resource_name}_cache.db")
        print(f"  Opening database for checkpointing at {db_path}...")
        conn = sqlite3.connect(db_path)

        # Create tables and add language list
        init_db(conn, languages)

        # Load pairs we've already finished
        done_pairs = load_done_pairs(conn)
        total_pairs = math.comb(len(languages), 2)
        print(f"  Found {len(done_pairs)} / {total_pairs} existing results in DB.")

        # --- 5. Compute metrics (PARALLELIZED with Queue) ---

        # We need a Manager for a queue that can be shared by processes
        with multiprocessing.Manager() as manager:
            queue = manager.Queue()
            tasks = []

            # Build task list, skipping completed pairs
            for lang1, lang2 in itertools.combinations(languages, 2):
                if tuple(sorted((lang1, lang2))) not in done_pairs:
                    tasks.append((lang1, lang2,
                                  language_matrices[lang1],
                                  language_matrices[lang2],
                                  n_items,
                                  queue))  # Pass the queue to the worker

            if not tasks:
                print("  All comparisons are already computed.")
            else:
                print(f"  Preparing {len(tasks)} new comparison tasks...")
                sys.stdout.flush()

                # Limit cores to leave 2 free, but always use at least 1
                max_cores_to_use = max(1, os.cpu_count() - 2)
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores_to_use) as executor:

                    # Submit all tasks
                    futures = [executor.submit(_compute_pair, task) for task in tasks]

                    # Start the listener loop
                    # This loop pulls results from the queue as they come in
                    # and saves them to the DB one by one.
                    for _ in tqdm(range(len(tasks)), desc="  Processing metrics"):
                        # Get next completed result from ANY worker
                        lang1, lang2, results = queue.get()

                        # Save this one result to the DB.
                        # This is crash-safe.
                        save_result_to_db(conn, lang1, lang2, results)

            print("Computation complete.")

        # --- NEW: 6. Final Export ---
        print(f"  Exporting all matrices from DB to CSV files...")
        export_to_csv(conn, resource_name, OUTPUT_DIR)

        # Close the database connection
        conn.close()

    print("Processing complete.")


if __name__ == "__main__":
    main()