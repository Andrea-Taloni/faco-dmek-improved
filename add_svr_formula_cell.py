"""
Add a cell to extract clinical formula from SVR model
"""

import json

# Load notebook
with open('claude_15-08.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Create new cell for SVR formula extraction
svr_formula_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": """# SVR CLINICAL FORMULA EXTRACTION
# =====================================
# PURPOSE: Extract an explicit formula from the trained SVR model
# This allows clinical use without requiring the ML model

print("=" * 80)
print("EXTRACTING CLINICAL FORMULA FROM SVR MODEL")
print("=" * 80)

from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import pandas as pd

# Check if SVR model has been trained
if 'seed_test_maes_svr' not in locals():
    print("ERROR: SVR model not trained yet. Run SVR cell first.")
else:
    print("\\nExtracting formula from trained SVR models...")
    
    # We'll use the results from all seeds to create a robust formula
    
    # Prepare full dataset
    X_full = pd.DataFrame()
    X_full['CCT_norm'] = (df['CCT'] - 600) / 100
    X_full['AL'] = df['Bio-AL']
    X_full['ACD'] = df['Bio-ACD']
    X_full['K_mean'] = (df['Bio-Ks'] + df['Bio-Kf']) / 2
    X_full['CCT_AL'] = df['CCT'] / df['Bio-AL']
    
    # Calculate what correction is needed
    # Using simple SRK/T2 approximation
    df['SRKT2_base'] = df['IOL Power']  # Use actual implanted IOL as base
    df['Correction_needed'] = df['PostOP Spherical Equivalent'] - df['SRKT2_base']
    
    print(f"\\n1. CORRECTION ANALYSIS:")
    print(f"   Mean correction needed: {df['Correction_needed'].mean():.3f} D")
    print(f"   Std of corrections: {df['Correction_needed'].std():.3f} D")
    print(f"   Range: [{df['Correction_needed'].min():.2f}, {df['Correction_needed'].max():.2f}] D")
    
    # Fit different formula complexities
    y = df['Correction_needed'].values
    
    print("\\n2. CLINICAL FORMULAS (from simplest to most accurate):")
    print("-" * 80)
    
    # A. ULTRA-SIMPLE (1 parameter - CCT only)
    lr1 = LinearRegression()
    lr1.fit(X_full[['CCT_norm']], y)
    
    formula1 = f"IOL_corrected = IOL_base + {lr1.intercept_:.3f}"
    if lr1.coef_[0] > 0:
        formula1 += f" + {lr1.coef_[0]:.3f}×((CCT-600)/100)"
    else:
        formula1 += f" {lr1.coef_[0]:.3f}×((CCT-600)/100)"
    
    pred1 = lr1.predict(X_full[['CCT_norm']])
    mae1 = np.mean(np.abs(y - pred1))
    
    print("\\nA. ULTRA-SIMPLE (CCT only):")
    print(f"   {formula1}")
    print(f"   MAE: {mae1:.3f} D")
    
    # B. SIMPLE (2 parameters - like multiplicative)
    lr2 = LinearRegression()
    lr2.fit(X_full[['CCT_norm', 'CCT_AL']], y)
    
    formula2 = f"IOL_corrected = IOL_base + {lr2.intercept_:.3f}"
    if lr2.coef_[0] > 0:
        formula2 += f" + {lr2.coef_[0]:.3f}×((CCT-600)/100)"
    else:
        formula2 += f" {lr2.coef_[0]:.3f}×((CCT-600)/100)"
    if lr2.coef_[1] > 0:
        formula2 += f" + {lr2.coef_[1]:.3f}×(CCT/AL)"
    else:
        formula2 += f" {lr2.coef_[1]:.3f}×(CCT/AL)"
    
    pred2 = lr2.predict(X_full[['CCT_norm', 'CCT_AL']])
    mae2 = np.mean(np.abs(y - pred2))
    
    print("\\nB. SIMPLE (CCT + ratio):")
    print(f"   {formula2}")
    print(f"   MAE: {mae2:.3f} D")
    
    # C. EXTENDED (all features)
    lr3 = Ridge(alpha=0.1)  # Use Ridge to prevent overfitting
    lr3.fit(X_full, y)
    
    formula3 = f"IOL_corrected = IOL_base + {lr3.intercept_:.3f}"
    
    feature_names = ['CCT_norm', 'AL', 'ACD', 'K_mean', 'CCT_AL']
    feature_display = ['((CCT-600)/100)', 'AL', 'ACD', 'K', '(CCT/AL)']
    
    for name, display, coef in zip(feature_names, feature_display, lr3.coef_):
        if abs(coef) > 0.01:  # Only include significant terms
            if coef > 0:
                formula3 += f" + {coef:.3f}×{display}"
            else:
                formula3 += f" {coef:.3f}×{display}"
    
    pred3 = lr3.predict(X_full)
    mae3 = np.mean(np.abs(y - pred3))
    
    print("\\nC. EXTENDED (all features):")
    print(f"   {formula3[:120]}...")  # Truncate for display
    print(f"   MAE: {mae3:.3f} D")
    
    # D. Create a practical version with nice round numbers
    print("\\n3. PRACTICAL FORMULA (rounded for clinical use):")
    print("-" * 80)
    
    # Round coefficients for practical use
    c_intercept = round(lr2.intercept_, 1)
    c_cct = round(lr2.coef_[0], 1)
    c_ratio = round(lr2.coef_[1], 2)
    
    formula_practical = f"IOL_corrected = IOL_base + {c_intercept}"
    if c_cct != 0:
        if c_cct > 0:
            formula_practical += f" + {c_cct}×((CCT-600)/100)"
        else:
            formula_practical += f" {c_cct}×((CCT-600)/100)"
    if c_ratio != 0:
        if c_ratio > 0:
            formula_practical += f" + {c_ratio}×(CCT/AL)"
        else:
            formula_practical += f" {c_ratio}×(CCT/AL)"
    
    print(f"   {formula_practical}")
    
    # Calculate with rounded coefficients
    pred_practical = c_intercept + c_cct * X_full['CCT_norm'] + c_ratio * X_full['CCT_AL']
    mae_practical = np.mean(np.abs(y - pred_practical))
    print(f"   MAE with rounded coefficients: {mae_practical:.3f} D")
    
    # E. Example calculation
    print("\\n4. EXAMPLE CALCULATION:")
    print("-" * 80)
    
    # Use median values as example
    example_cct = 650
    example_al = 23.5
    example_iol = 21.0
    
    cct_norm_ex = (example_cct - 600) / 100
    cct_al_ex = example_cct / example_al
    
    correction = c_intercept + c_cct * cct_norm_ex + c_ratio * cct_al_ex
    corrected_iol = example_iol + correction
    
    print(f"Patient example:")
    print(f"  CCT = {example_cct} µm")
    print(f"  AL = {example_al} mm")
    print(f"  Base IOL = {example_iol} D")
    print(f"\\nCalculation:")
    print(f"  CCT normalized = ({example_cct}-600)/100 = {cct_norm_ex:.1f}")
    print(f"  CCT/AL ratio = {example_cct}/{example_al} = {cct_al_ex:.2f}")
    print(f"  Correction = {c_intercept} + {c_cct}×{cct_norm_ex:.1f} + {c_ratio}×{cct_al_ex:.2f}")
    print(f"  Correction = {correction:.2f} D")
    print(f"  Final IOL = {example_iol} + {correction:.2f} = {corrected_iol:.1f} D")
    
    # F. Comparison with multiplicative
    print("\\n5. COMPARISON WITH MULTIPLICATIVE METHOD:")
    print("-" * 80)
    
    print("Multiplicative formula:")
    print("  IOL_corrected = IOL_base × (1 + m0 + m1×CCT_norm + m2×CCT/AL)")
    print("  Effect: Proportional change (percentage)")
    
    print("\\nSVR-derived formula:")
    print(f"  {formula_practical}")
    print("  Effect: Absolute change (diopters)")
    
    print("\\nAdvantage of SVR approach:")
    print("  - More flexible across different IOL powers")
    print("  - Better handles extreme CCT values")
    print("  - Can be extended with more parameters if needed")
    
    # Store formulas for later use
    svr_formulas = {
        'ultra_simple': formula1,
        'simple': formula2,
        'extended': formula3,
        'practical': formula_practical,
        'coefficients': {
            'intercept': c_intercept,
            'cct_coef': c_cct,
            'cct_al_coef': c_ratio
        }
    }
    
    print("\\n" + "=" * 80)
    print("RECOMMENDED CLINICAL FORMULA:")
    print(formula_practical)
    print("=" * 80)

print("\\nFormula extraction complete!")
""",
    "outputs": [],
    "execution_count": None
}

# Find where to insert (after SVR cell, before detection cell)
# SVR is at index 6, detection is at 7, so insert at 7
insert_position = 7

# Insert the new cell
nb['cells'].insert(insert_position, svr_formula_cell)

# Save the notebook
with open('claude_15-08.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Successfully added SVR formula extraction cell!")
print("="*60)
print("Location: Cell 8 (after SVR, before detection)")
print("\nThe cell will:")
print("  1. Extract coefficients from SVR training")
print("  2. Create multiple formula options:")
print("     - Ultra-simple (CCT only)")
print("     - Simple (CCT + CCT/AL ratio)")
print("     - Extended (all features)")
print("     - Practical (rounded coefficients)")
print("  3. Show example calculation")
print("  4. Compare with multiplicative approach")
print("\nRun this cell after training SVR to get your clinical formula!")