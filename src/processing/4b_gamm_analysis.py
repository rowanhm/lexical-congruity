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
ALPHA = 0.05

INDEPENDENT_VARIABLES = [
    'dist_patristic',  # Independent Variable (X1)
    'dist_physical',  # Independent Variable (X2)
    'lex_overlap'
]

DEPENDENT_VAR = 'hofstede_avg'

# Calculate the probabilities for R
prob_mass = 1 - ALPHA          # e.g., 0.95
p_lower = ALPHA / 2            # e.g., 0.025
p_upper = 1 - (ALPHA / 2)      # e.g., 0.975

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

print("Standardizing variables (Z-score)...")
# Combine lists to scale both dependent and independent variables
vars_to_scale = INDEPENDENT_VARIABLES + [DEPENDENT_VAR]

for var in vars_to_scale:
    if var in combined.columns:
        combined[var] = (combined[var] - combined[var].mean()) / combined[var].std()

print(f"Data Loaded. Found {len(combined)} overlapping pairs.")
print(f"Total observations for model: {len(combined)}")

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

# 1. Fixed Effects
# We pass the specific quantiles based on your Python ALPHA
fixed_effects <- fixef(model, probs = c({p_lower}, {p_upper}))

# 2. Smooth Terms
# We pass the probability mass (e.g., 0.95)
model_summary <- summary(model, prob = {prob_mass})
smooth_terms <- model_summary$splines

# 3. Bayesian R2
r2_val <- bayes_R2(model)
r2_mean <- mean(r2_val[,1])
"""

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

print(f"\n2. LINEAR TRENDS (FIXED EFFECTS) | Alpha: {ALPHA}")
print("-" * 60)

for index, row in fixed_df.iterrows():
    clean_name = index.replace('_', ' ').title()
    if clean_name == "Intercept": continue

    # Use integer location (.iloc) to avoid column name errors
    # Col 0: Estimate, Col 1: Error, Col 2: Lower Bound, Col 3: Upper Bound
    est = row.iloc[0]
    lower = row.iloc[2]
    upper = row.iloc[3]

    # Check significance
    is_sig = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)
    sig_tag = "[SIGNIFICANT]" if is_sig else "[Not Significant]"

    print(f"   * {clean_name}:")
    print(f"       Slope:      {est:.4f}")
    print(f"       {(1-ALPHA)*100:.0f}% CI:     [{lower:.4f}, {upper:.4f}]")
    print(f"       Conclusion: {sig_tag}")

print(f"\n3. NON-LINEAR PATTERNS (SMOOTH TERMS) | Alpha: {ALPHA}")
print("-" * 60)

for index, row in smooth_df.iterrows():
    clean_name = index.replace('sds(', '').replace('s(', '').replace(')', '').replace('_', ' ').title()

    # Use integer location (.iloc) here as well
    est = row.iloc[0]
    lower = row.iloc[2]
    upper = row.iloc[3]

    # Significance check
    is_sig = lower > 0.01
    sig_tag = "[SIGNIFICANT]" if is_sig else "[Negligible]"

    print(f"   * {clean_name}:")
    print(f"       Variation (SD): {est:.4f}")
    print(f"       {(1-ALPHA)*100:.0f}% CI:     [{lower:.4f}, {upper:.4f}]")
    print(f"       Conclusion:     {sig_tag}")