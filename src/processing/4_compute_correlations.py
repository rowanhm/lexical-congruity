import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
from tqdm import tqdm  # Added import


def process_matrices(input_glob, output_file):
    """
    Computes Spearman correlation for all pairs of matrices from an input glob
    and saves the result as a correlation matrix.
    """
    # Find all file paths matching the pattern
    filepaths = sorted(glob.glob(input_glob))
    if not filepaths:
        print(f"No files found matching {input_glob}")
        return

    # Get just the filenames to use as keys in the output
    filenames = [os.path.basename(fp) for fp in filepaths]

    # Load all matrices into memory
    print("Loading matrices...")
    matrices = {}
    for fp, name in tqdm(zip(filepaths, filenames), total=len(filepaths), desc="Loading files"):
        try:
            matrices[name] = pd.read_csv(fp, index_col=0)
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    # Initialize the output DataFrame
    output_df = pd.DataFrame(index=filenames, columns=filenames, dtype=float)
    output_df.index.name = "matrix"
    output_df.columns.name = "matrix"

    # Calculate total number of pairs for tqdm
    num_pairs = len(filenames) * (len(filenames) - 1) // 2

    print("Processing matrix pairs...")
    # Iterate over all unique pairs of matrix names
    for file_a_name, file_b_name in tqdm(combinations(filenames, 2), total=num_pairs, desc="Processing pairs"):
        df_a = matrices[file_a_name]
        df_b = matrices[file_b_name]

        # Find shared keys (index/columns)
        shared_keys = df_a.index.intersection(df_b.index)

        if len(shared_keys) < 2:
            print(f"Skipping {file_a_name} and {file_b_name}: Not enough shared keys.")
            continue

        # Filter both matrices to only include shared keys
        df_a_shared = df_a.loc[shared_keys, shared_keys]
        df_b_shared = df_b.loc[shared_keys, shared_keys]

        # Get the upper triangle values (excluding diagonal, k=1)
        indices = np.triu_indices_from(df_a_shared, k=1)
        vals_a = df_a_shared.values[indices]
        vals_b = df_b_shared.values[indices]

        correlation = np.nan
        # Ensure there are at least 2 values to correlate
        if vals_a.size > 1:
            try:
                # Calculate Spearman correlation
                corr, _ = spearmanr(vals_a, vals_b)
                if not np.isnan(corr):
                    correlation = corr
            except ValueError as e:
                print(f"Could not calculate correlation for {file_a_name} and {file_b_name}: {e}")

        # Store the result in the output DataFrame (symmetrically)
        output_df.loc[file_a_name, file_b_name] = correlation
        output_df.loc[file_b_name, file_a_name] = correlation

    # Set the diagonal to 1.0
    np.fill_diagonal(output_df.values, 1.0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the output
    output_df.to_csv(output_file)
    print(f"Correlation matrix saved to {output_file}")


if __name__ == "__main__":
    # Define file paths
    input_pattern = 'bin/matrices/*.csv'
    output_path = 'bin/correlations.csv'

    # Run the processing
    process_matrices(input_pattern, output_path)