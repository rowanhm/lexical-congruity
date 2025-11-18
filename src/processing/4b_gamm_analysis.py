import pandas as pd
import numpy as np
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import conversion, default_converter

CORES = 4
ITERATIONS = 4000
CHAINS = 4

INDEPENDENT_VARIABLES = [
    'patristic_distance',  # Independent Variable (X1)
    'physical_distance',  # Independent Variable (X2)
    'lexibank_omega'
]

DEPENDENT_VAR = 'hofstede_euclidean_difference'

datasets = []

print("Loading and aligning matrices...")

for var_name in INDEPENDENT_VARIABLES + [DEPENDENT_VAR]:
    filename = f"{var_name}.csv"
    filepath = os.path.join('bin', 'matrices', filename)

    # Load matrix (assuming TSV format)
    matrix = pd.read_csv(filepath, index_col=0)

    # Flatten matrix to [lang_1, lang_2, value]
    flat = matrix.stack().reset_index()
    flat.columns = ['l1', 'l2', var_name]

    # Remove self-comparisons (e.g. English-English)
    flat = flat[flat['l1'] != flat['l2']]

    # Create a sorted array of the language pairs
    sorted_langs = np.sort(flat[['l1', 'l2']].values, axis=1)

    # Assign the sorted values BACK to the DataFrame columns
    flat['l1'] = sorted_langs[:, 0]
    flat['l2'] = sorted_langs[:, 1]

    # Now drop duplicates.
    # Since 'l1' and 'l2' are now alphabetically consistent (always English-French),
    # the second occurrence (French-English) becomes a duplicate and is dropped.
    flat = flat.drop_duplicates(subset=['l1', 'l2'])

    flat = flat.set_index(['l1', 'l2'])[[var_name]]

    datasets.append(flat)

# Inner Join: Finds the intersection of all matrices automatically
# Pairs missing in ANY matrix are dropped
combined = pd.concat(datasets, axis=1, join='inner').dropna()
combined = combined.reset_index()

# We must ensure l1 and l2 are categories with the same set of allowed values
all_langs = pd.unique(combined[['l1', 'l2']].values.ravel('K'))
# Convert to categorical type in Pandas with shared categories
combined['l1'] = pd.Categorical(combined['l1'], categories=all_langs)
combined['l2'] = pd.Categorical(combined['l2'], categories=all_langs)

print(f"Data Loaded. Found {len(combined)} overlapping pairs.")
print(f"Total observations for model: {len(combined)}")

print("Standardizing variables (Z-score)...")
# Combine lists to scale both dependent and independent variables
vars_to_scale = INDEPENDENT_VARIABLES + [DEPENDENT_VAR]

for var in vars_to_scale:
    if var in combined.columns:
        combined[var] = (combined[var] - combined[var].mean()) / combined[var].std()

# Define formula string for R
formula = f"{DEPENDENT_VAR} ~ {' + '.join(['s(' + p + ')' for p in INDEPENDENT_VARIABLES])} + (1 | mm(l1, l2))"
print(f"Fitting Model: {formula}")

model_code = f"""
model <- brm(
    formula = {formula},
    data = df_r,
    family = gaussian(),
    chains = {CHAINS},    # Number of Markov Chains
    iter = {ITERATIONS},   # Number of iterations
    cores = {CORES},      # Parallel processing
    backend = "cmdstanr",
    control = list(adapt_delta = 0.99)
)
summary(model)
"""

# Execute in R context
print("Converting data to R object...")

brms = importr('brms')

# Create a conversion context that includes pandas conversion
with (default_converter + pandas2ri.converter).context():
    # Convert the pandas dataframe to an R dataframe explicitly
    r_data = ro.conversion.get_conversion().py2rpy(combined)
    # Inject into R global environment
    ro.globalenv['df_r'] = r_data

# Run the model
print("Compiling and running Bayesian model (this may take a minute)...")
output = ro.r(model_code)
print(output)

# ==========================================
# ANALYSIS & INTERPRETATION CODE (FINAL)
# ==========================================

print("Extracting model results and statistics...")

# Define R extraction code
extraction_code = """
# 1. Fixed Effects (Linear Trends)
fixed_effects <- fixef(model)

# 2. Smooth Terms (Curves/Wiggliness)
model_summary <- summary(model)
smooth_terms <- model_summary$splines

# 3. Bayesian R2
r2_val <- bayes_R2(model)
r2_mean <- mean(r2_val[,1])
"""

# Run R extraction
ro.r(extraction_code)

# Bring data back to Python
with (default_converter + pandas2ri.converter).context():
    # Fixed Effects
    fixed_data = ro.globalenv['fixed_effects']
    fixed_index = list(ro.r("rownames(fixed_effects)"))
    fixed_cols = list(ro.r("colnames(fixed_effects)"))
    fixed_df = pd.DataFrame(fixed_data, index=fixed_index, columns=fixed_cols)

    # Smooth Terms
    smooth_data = ro.globalenv['smooth_terms']
    smooth_index = list(ro.r("rownames(smooth_terms)"))
    smooth_cols = list(ro.r("colnames(smooth_terms)"))
    smooth_df = pd.DataFrame(smooth_data, index=smooth_index, columns=smooth_cols)

    # R2
    r2_score = ro.globalenv['r2_mean'][0]

print("\n" + "=" * 60)
print(f"  MODEL REPORT: {DEPENDENT_VAR}")
print("=" * 60)

print(f"\n1. OVERALL MODEL FIT")
print(f"   Bayesian R-squared: {r2_score:.4f}")
print(f"   (Model explains {r2_score * 100:.1f}% of the variance)")

print("\n2. LINEAR TRENDS (FIXED EFFECTS)")
print("   (Directional effects: Positive = Increases difference, Negative = Decreases)")
print("-" * 60)

for index, row in fixed_df.iterrows():
    # Clean name
    clean_name = index.replace('_', ' ').title()
    if clean_name == "Intercept": continue  # Skip intercept for clarity in this list

    est = row['Estimate']
    lower = row['Q2.5']
    upper = row['Q97.5']

    # Significance: Does the interval cross zero?
    is_sig = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)
    sig_tag = "[SIGNIFICANT]" if is_sig else "[Not Significant]"

    print(f"   * {clean_name}:")
    print(f"       Slope:      {est:.4f}")
    print(f"       95% CI:     [{lower:.4f}, {upper:.4f}]")
    print(f"       Conclusion: {sig_tag}")

print("\n3. NON-LINEAR PATTERNS (SMOOTH TERMS)")
print("   (Magnitude of 'wiggliness' or complexity. Higher = More complex curve)")
print("-" * 60)

for index, row in smooth_df.iterrows():
    # Clean name
    clean_name = index.replace('sds(', '').replace('s(', '').replace(')', '').replace('_', ' ').title()

    est = row['Estimate']
    lower = row['l-95% CI']
    upper = row['u-95% CI']

    # Significance: Is the lower bound comfortably away from zero?
    # We use > 0.01 as a threshold for "numerical significance"
    is_sig = lower > 0.01
    sig_tag = "[SIGNIFICANT]" if is_sig else "[Negligible]"

    # Add nuance for borderline cases (like Omega might be)
    if is_sig and lower < 0.05:
        sig_tag += " (Borderline/Weak)"

    print(f"   * {clean_name}:")
    print(f"       Variation (SD): {est:.4f}")
    print(f"       95% CI:         [{lower:.4f}, {upper:.4f}]")
    print(f"       Conclusion:     {sig_tag}")

print("\n" + "=" * 60)