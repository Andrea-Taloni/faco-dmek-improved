# GLOBAL BASELINE CALCULATION - CONSISTENT ACROSS ALL METHODS
# ===========================================================
# PURPOSE: Calculate baseline performance ONCE for fair comparison
# This ensures all methods use the same baseline for improvement calculations

print("=" * 80)
print("CALCULATING GLOBAL BASELINE FOR FAIR COMPARISON")
print("=" * 80)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

print("\n📊 CALCULATING BASELINE ACROSS ALL SEEDS:")
print("-" * 50)

# Store baseline MAEs for each seed
global_baseline_maes = []

for seed_idx, SEED in enumerate(SEEDS, 1):
    # Split data with this seed
    X_train_base, X_test_base = train_test_split(df, test_size=0.25, random_state=SEED)
    
    # Calculate K_avg
    X_test_base['K_avg'] = (X_test_base['Bio-Ks'] + X_test_base['Bio-Kf']) / 2
    
    # Calculate baseline predictions
    X_test_base['SRKT2_Baseline'] = X_test_base.apply(
        lambda row: calculate_SRKT2(
            AL=row['Bio-AL'],
            K_avg=row['K_avg'],
            IOL_power=row['IOL Power'],
            A_constant=row['A-Constant']
        ), axis=1
    )
    
    # Calculate MAE for this seed
    baseline_mae = mean_absolute_error(
        X_test_base['PostOP Spherical Equivalent'], 
        X_test_base['SRKT2_Baseline']
    )
    
    global_baseline_maes.append(baseline_mae)
    print(f"  Seed {SEED}: Baseline MAE = {baseline_mae:.4f} D")

# Calculate global baseline (average across all seeds)
GLOBAL_BASELINE_MAE = np.mean(global_baseline_maes)
GLOBAL_BASELINE_STD = np.std(global_baseline_maes)

print("\n" + "="*80)
print("GLOBAL BASELINE ESTABLISHED:")
print("="*80)
print(f"\n📌 BASELINE MAE: {GLOBAL_BASELINE_MAE:.4f} ± {GLOBAL_BASELINE_STD:.4f} D")
print(f"   Min: {np.min(global_baseline_maes):.4f} D")
print(f"   Max: {np.max(global_baseline_maes):.4f} D")

print("\n⚠️ IMPORTANT:")
print("-" * 50)
print("All methods will now use this baseline for improvement calculations.")
print("This ensures fair comparison between methods.")
print(f"Improvement = (({GLOBAL_BASELINE_MAE:.4f} - Method_MAE) / {GLOBAL_BASELINE_MAE:.4f}) × 100%")

# Store for use by all subsequent methods
print(f"\n✅ Global baseline stored in variable: GLOBAL_BASELINE_MAE = {GLOBAL_BASELINE_MAE:.4f}")
print("All methods should now calculate improvement as:")
print("   improvement = ((GLOBAL_BASELINE_MAE - test_mae) / GLOBAL_BASELINE_MAE) * 100")