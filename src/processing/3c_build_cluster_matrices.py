import json
import math
import os
import glob
import itertools
import csv
import concurrent.futures
import sqlite3
import multiprocessing
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm

# --- Configuration ---
LEXICA_DIR = "bin/lexica"
OUTPUT_DIR = "bin/matrices"
METRICS = [
    "omega",  # Omega Index
    "nmi",  # Overlapping NMI
    "f1",  # F1 Score
    "jaccard",  # Jaccard Similarity
    "overlap",  # Simpson's Overlap (Symmetric Robust)
    "prec_max",  # Highest Precision (Robustness)
    "rec_min",  # Lowest Recall (Missing Data Penalty)
    "homo_max",  # Highest Homogeneity (Robustness)
    "comp_min"  # Lowest Completeness (Missing Data Penalty)
]

# --- Global Storage for Workers ---
_worker_matrices = None
_worker_stats = None


def _init_worker(matrices, stats):
    """
    Initializes worker processes with shared read-only data.
    """
    global _worker_matrices, _worker_stats
    _worker_matrices = matrices
    _worker_stats = stats


def _binary_entropy_vectorized(p):
    """
    Computes H(p) = -p*log2(p) - (1-p)*log2(1-p) for a vector p.
    Handles 0 and 1 safely to avoid log(0).
    """
    # Initialize result with zeros (covers p=0 and p=1 cases implicitly)
    res = np.zeros_like(p)

    # Mask for p in (0, 1)
    mask = (p > 0) & (p < 1)

    if np.any(mask):
        pm = p[mask]
        res[mask] = -pm * np.log2(pm) - (1 - pm) * np.log2(1 - pm)

    return res


def _compute_asymmetric_entropy_sparse(cm_sparse, sizes1, sizes2, n_items, h1_marginal, h2_marginal):
    """
    Computes conditional entropies and NMI using sparse matrix operations.
    Avoids densifying the matrix to prevent OOM errors.
    """
    if n_items == 0:
        return 0.0, 0.0, 0.0

    if h1_marginal == 0 and h2_marginal == 0:
        return 1.0, 1.0, 1.0
    if h1_marginal == 0:
        return 0.0, 1.0, 0.0
    if h2_marginal == 0:
        return 1.0, 0.0, 0.0

    # Extract sparse data for vectorization
    # cm_sparse is M1.T @ M2 (Rows: Lang1 Clusters, Cols: Lang2 Clusters)
    data = cm_sparse.data
    rows, cols = cm_sparse.nonzero()

    p1 = sizes1 / n_items
    p2 = sizes2 / n_items

    # --- H(X|Y): Uncertainty of Lang1 given Lang2 ---
    # P(x|y) = count(x,y) / size(y)
    denoms_y = sizes2[cols]
    p_x_given_y = data / denoms_y

    # Calculate binary entropy for each intersection
    h_vals_xy = _binary_entropy_vectorized(p_x_given_y)

    # Aggregate entropies per column (Lang2 cluster)
    # Sum_x H(X=x|Y=y)
    h_xy_sums = np.zeros(len(sizes2))
    np.add.at(h_xy_sums, cols, h_vals_xy)

    # Weighted average by P(Y)
    H_X_given_Y = np.sum(p2 * h_xy_sums)

    # --- H(Y|X): Uncertainty of Lang2 given Lang1 ---
    # P(y|x) = count(x,y) / size(x)
    denoms_x = sizes1[rows]
    p_y_given_x = data / denoms_x

    # Calculate binary entropy for each intersection
    h_vals_yx = _binary_entropy_vectorized(p_y_given_x)

    # Aggregate entropies per row (Lang1 cluster)
    h_yx_sums = np.zeros(len(sizes1))
    np.add.at(h_yx_sums, rows, h_vals_yx)

    # Weighted average by P(X)
    H_Y_given_X = np.sum(p1 * h_yx_sums)

    # Results
    # If H(X|Y) is 0, knowing Y tells us X perfectly.
    homog_1_given_2 = 1.0 - (H_X_given_Y / h1_marginal)
    homog_2_given_1 = 1.0 - (H_Y_given_X / h2_marginal)

    # NMI (Symmetric)
    I_XY = (h1_marginal - H_X_given_Y + h2_marginal - H_Y_given_X) / 2.0
    nmi = I_XY / max(np.sqrt(h1_marginal * h2_marginal), 1e-15)

    return homog_1_given_2, homog_2_given_1, nmi


def precompute_language_stats(matrix, n_items):
    """
    Calculates invariant statistics for a language matrix.
    Run once per language to save compute time during pairwise comparisons.
    """
    # 1. Cluster Sizes and Marginal Entropy
    sizes = np.array(matrix.sum(axis=0)).flatten()
    p_vec = sizes / n_items
    marginal_entropy = np.sum(_binary_entropy_vectorized(p_vec))

    # 2. Self-Pairs (for Omega/F1)
    # Faster to compute using CSR dot product
    matrix_csr = matrix.tocsr()
    c_co = matrix_csr.dot(matrix_csr.T)
    diag_sum = c_co.diagonal().sum()
    pairs_count = (c_co.sum() - diag_sum) / 2.0

    return {
        "sizes": sizes,
        "marginal_entropy": marginal_entropy,
        "pairs_count": pairs_count
    }


def _compute_pair(task_data):
    """
    Worker function to compute metrics between two languages.
    """
    lang1, lang2, n_items, queue = task_data

    # Retrieve shared data
    M1 = _worker_matrices[lang1]
    M2 = _worker_matrices[lang2]
    stats1 = _worker_stats[lang1]
    stats2 = _worker_stats[lang2]

    results = {}

    try:
        # --- Pairwise Counts (Intersection) ---
        # M is (Items x Clusters). Dot product gives (Items x Items) co-occurrence.
        # We only compute the intersection of the two co-occurrence matrices.

        # Note: M1.dot(M1.T) is heavy. We rely on sparse multiplication efficiency.
        # In a highly optimized setting, we would avoid the full M*M.T if items > 50k,
        # but strictly following the logic required for Omega/F1 on clusters, we proceed.
        C1_co = M1.dot(M1.T)
        C2_co = M2.dot(M2.T)

        # Element-wise multiply to find common pairs
        C_common = C1_co.multiply(C2_co)

        diag_sum_common = C_common.diagonal().sum()
        a = (C_common.sum() - diag_sum_common) / 2.0  # Intersection

        pairs_in_1 = stats1["pairs_count"]
        pairs_in_2 = stats2["pairs_count"]

        b = pairs_in_1 - a
        c = pairs_in_2 - a

        # --- Basic Metrics ---
        union = a + b + c
        denom_f1 = (2 * a) + b + c

        results["f1"] = (2 * a) / denom_f1 if denom_f1 > 0 else 1.0
        results["jaccard"] = a / union if union > 0 else 1.0

        # --- Omega Index ---
        n_pairs_total = (n_items * (n_items - 1)) / 2.0
        if n_pairs_total > 0:
            expected_a = (pairs_in_1 * pairs_in_2) / n_pairs_total
            max_a = (pairs_in_1 + pairs_in_2) / 2.0
            denominator = max_a - expected_a
            if denominator != 0:
                results["omega"] = (a - expected_a) / denominator
            else:
                results["omega"] = 1.0
        else:
            results["omega"] = 0.0

        # --- Simpson / Precision / Recall ---
        min_size = min(pairs_in_1, pairs_in_2)
        results["overlap"] = a / min_size if min_size > 0 else 1.0

        val1 = a / pairs_in_1 if pairs_in_1 > 0 else 0.0
        val2 = a / pairs_in_2 if pairs_in_2 > 0 else 0.0

        results["prec_max"] = max(val1, val2)
        results["rec_min"] = min(val1, val2)

        # --- Entropy Metrics (Sparse) ---
        # Contingency Matrix: Rows=Clusters1, Cols=Clusters2
        cm_sparse = M1.T.dot(M2)

        h1_given_2, h2_given_1, nmi = _compute_asymmetric_entropy_sparse(
            cm_sparse,
            stats1["sizes"],
            stats2["sizes"],
            n_items,
            stats1["marginal_entropy"],
            stats2["marginal_entropy"]
        )

        results["nmi"] = nmi
        results["homo_max"] = max(h1_given_2, h2_given_1)
        results["comp_min"] = min(h1_given_2, h2_given_1)

    except Exception as e:
        print(f"Error computing {lang1}-{lang2}: {e}")
        results = {metric: math.nan for metric in METRICS}

    queue.put((lang1, lang2, results))


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

    # Create CSC first for efficient construction
    return csc_matrix((data, (row_indices, col_indices)),
                      shape=(n_items, n_clusters), dtype=np.int8)


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


# --- Database Helpers ---

def init_db(conn, languages):
    with conn:
        cursor = conn.cursor()
        metric_cols = ", ".join([f"{name} REAL" for name in METRICS])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS results (
                lang1 TEXT,
                lang2 TEXT,
                {metric_cols},
                PRIMARY KEY (lang1, lang2)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS languages (
                lang TEXT PRIMARY KEY
            )
        """)
        cursor.executemany(
            "INSERT OR IGNORE INTO languages (lang) VALUES (?)",
            [(lang,) for lang in languages]
        )


def load_done_pairs(conn):
    done_pairs = set()
    with conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT lang1, lang2 FROM results")
            for row in cursor.fetchall():
                done_pairs.add(tuple(sorted(row)))
        except sqlite3.OperationalError:
            pass  # Table might not exist yet
    return done_pairs


def save_batch_results(conn, batch):
    if not batch:
        return
    with conn:
        metric_names = ", ".join(METRICS)
        placeholders = ", ".join(["?"] * len(METRICS))

        db_rows = []
        for lang1, lang2, results in batch:
            values = [results.get(name, math.nan) for name in METRICS]
            db_rows.append((lang1, lang2, *values))

        conn.executemany(
            f"INSERT OR REPLACE INTO results (lang1, lang2, {metric_names}) VALUES (?, ?, {placeholders})",
            db_rows
        )


def export_to_csv(conn, resource_name, output_dir):
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT lang FROM languages ORDER BY lang")
        languages = [row[0] for row in cursor.fetchall()]

    if not languages:
        return

    all_results = {}
    for metric_name in METRICS:
        all_results[metric_name] = {
            lang: {inner_lang: None for inner_lang in languages}
            for lang in languages
        }
        for lang in languages:
            all_results[metric_name][lang][lang] = 1.0

    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()

        # Map column indices
        # Schema: lang1, lang2, metric1, metric2...
        for row in rows:
            lang1, lang2 = row[0], row[1]
            for i, metric_name in enumerate(METRICS):
                value = row[i + 2]
                if lang1 in all_results[metric_name] and lang2 in all_results[metric_name][lang1]:
                    all_results[metric_name][lang1][lang2] = value
                    all_results[metric_name][lang2][lang1] = value

    for metric_name, results_matrix in all_results.items():
        output_filename = os.path.join(output_dir, f"{resource_name[:3]}_{metric_name}.csv")
        with open(output_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + languages)
            for lang_row in languages:
                row_data = [results_matrix[lang_row][lang_col] for lang_col in languages]
                writer.writerow([lang_row] + ["" if v is None else v for v in row_data])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    resource_paths = glob.glob(os.path.join(LEXICA_DIR, "*"))

    for resource_path in resource_paths:
        if not os.path.isdir(resource_path):
            continue

        resource_name = os.path.basename(resource_path)
        print(f"Processing resource: {resource_name}...")

        # 1. Load Data
        language_data = {}
        lang_files = glob.glob(os.path.join(resource_path, "*.json"))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_load_language_file, f) for f in lang_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="  Loading data"):
                lang_code, sets_list = future.result()
                if sets_list is not None:
                    language_data[lang_code] = sets_list

        if len(language_data) < 2:
            continue

        # 2. Build Universe
        global_universe = set()
        for sets_list in language_data.values():
            for s in sets_list: global_universe.update(s)
        n_items = len(global_universe)
        if n_items == 0: continue
        global_item_map = {item: i for i, item in enumerate(global_universe)}

        # 3. Build Matrices & Precompute Stats
        print("  Building matrices and pre-computing stats...")
        language_matrices = {}
        language_stats = {}

        for lang_code, sets_list in tqdm(language_data.items(), desc="  Matrix construction"):
            # Build basic matrix
            mat = _build_membership_matrix(sets_list, global_item_map)

            # Convert to CSR immediately for efficient math operations later
            mat_csr = mat.tocsr()
            language_matrices[lang_code] = mat_csr

            # Precompute invariants
            language_stats[lang_code] = precompute_language_stats(mat_csr, n_items)

        del language_data, global_item_map

        # 4. Prepare DB
        languages = sorted(language_matrices.keys())
        db_path = os.path.join(OUTPUT_DIR, f"{resource_name}_cache.db")
        conn = sqlite3.connect(db_path)
        init_db(conn, languages)
        done_pairs = load_done_pairs(conn)

        # 5. Multiprocessing Execution
        with multiprocessing.Manager() as manager:
            queue = manager.Queue()
            tasks = []

            for lang1, lang2 in itertools.combinations(languages, 2):
                if tuple(sorted((lang1, lang2))) not in done_pairs:
                    tasks.append((lang1, lang2, n_items, queue))

            if tasks:
                max_cores = max(1, os.cpu_count() - 2)
                print(f"  Processing {len(tasks)} pairs with {max_cores} cores...")

                batch_buffer = []
                BATCH_SIZE = 200

                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_cores,
                        initializer=_init_worker,
                        initargs=(language_matrices, language_stats)
                ) as executor:
                    futures = [executor.submit(_compute_pair, task) for task in tasks]

                    # Consume queue
                    for _ in tqdm(range(len(tasks)), desc="  Computing metrics"):
                        lang1, lang2, results = queue.get()
                        batch_buffer.append((lang1, lang2, results))

                        if len(batch_buffer) >= BATCH_SIZE:
                            save_batch_results(conn, batch_buffer)
                            batch_buffer = []

                    # Save remaining
                    if batch_buffer:
                        save_batch_results(conn, batch_buffer)

        print(f"  Exporting to CSV...")
        export_to_csv(conn, resource_name, OUTPUT_DIR)
        conn.close()

    print("Processing complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()