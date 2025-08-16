"""
Create a clinically usable formula from SVR model
Uses residual correction approach
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

def create_svr_clinical_formula(df):
    """
    Create a clinical formula that approximates SVR correction
    """
    print("="*80)
    print("CREATING CLINICAL FORMULA FROM SVR APPROACH")
    print("="*80)
    
    # Calculate base SRKT2 predictions
    df['SRKT2_base'] = 118.4 - 2.5 * df['Bio-AL'] - 0.9 * ((df['Bio-Ks'] + df['Bio-Kf'])/2)
    
    # Calculate residuals (what needs to be corrected)
    df['Residual'] = df['PostOP Spherical Equivalent'] - df['SRKT2_base']
    
    print("\n1. ANALYZING CORRECTION PATTERNS:")
    print(f"   Mean correction needed: {df['Residual'].mean():.3f} D")
    print(f"   Std of corrections: {df['Residual'].std():.3f} D")
    
    # Create features for correction
    X = pd.DataFrame()
    X['CCT_norm'] = (df['CCT'] - 600) / 100
    X['AL'] = df['Bio-AL']
    X['ACD'] = df['Bio-ACD']
    X['K_mean'] = (df['Bio-Ks'] + df['Bio-Kf']) / 2
    X['CCT_AL_ratio'] = df['CCT'] / df['Bio-AL']
    
    y = df['Residual'].values
    
    # First train full SVR for comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_scaled, y)
    svr_pred = svr.predict(X_scaled)
    svr_mae = mean_absolute_error(y, svr_pred)
    
    print(f"\n2. SVR MODEL PERFORMANCE:")
    print(f"   MAE on residuals: {svr_mae:.3f} D")
    
    # Now create simplified formulas
    print("\n3. SIMPLIFIED CLINICAL FORMULAS:")
    print("-"*60)
    
    # A. Simple linear formula
    lr = LinearRegression()
    lr.fit(X[['CCT_norm', 'CCT_AL_ratio']], y)
    
    intercept = lr.intercept_
    coef_cct = lr.coef_[0]
    coef_ratio = lr.coef_[1]
    
    print("\nA. SIMPLE LINEAR (2 parameters - like multiplicative):")
    formula_simple = f"SRKT2_corrected = SRKT2_base + {intercept:.3f} "
    if coef_cct > 0:
        formula_simple += f"+ {coef_cct:.3f}×((CCT-600)/100) "
    else:
        formula_simple += f"{coef_cct:.3f}×((CCT-600)/100) "
    if coef_ratio > 0:
        formula_simple += f"+ {coef_ratio:.3f}×(CCT/AL)"
    else:
        formula_simple += f"{coef_ratio:.3f}×(CCT/AL)"
    
    print(f"   {formula_simple}")
    
    simple_pred = lr.predict(X[['CCT_norm', 'CCT_AL_ratio']])
    simple_mae = mean_absolute_error(y, simple_pred)
    print(f"   MAE: {simple_mae:.3f} D")
    
    # B. Extended linear formula
    lr_full = Ridge(alpha=0.1)  # Use Ridge to prevent overfitting
    lr_full.fit(X, y)
    
    print("\nB. EXTENDED LINEAR (5 parameters):")
    formula_extended = f"SRKT2_corrected = SRKT2_base + {lr_full.intercept_:.3f}"
    
    feature_names = ['CCT_norm', 'AL', 'ACD', 'K_mean', 'CCT_AL_ratio']
    for name, coef in zip(feature_names, lr_full.coef_):
        if abs(coef) > 0.01:  # Only include significant terms
            if name == 'CCT_norm':
                term = f"{coef:.3f}×((CCT-600)/100)"
            elif name == 'AL':
                term = f"{coef:.3f}×AL"
            elif name == 'ACD':
                term = f"{coef:.3f}×ACD"
            elif name == 'K_mean':
                term = f"{coef:.3f}×K"
            elif name == 'CCT_AL_ratio':
                term = f"{coef:.3f}×(CCT/AL)"
            else:
                continue
                
            if coef > 0:
                formula_extended += f" + {term}"
            else:
                formula_extended += f" {term}"
    
    print(f"   {formula_extended}")
    
    extended_pred = lr_full.predict(X)
    extended_mae = mean_absolute_error(y, extended_pred)
    print(f"   MAE: {extended_mae:.3f} D")
    
    # C. Compare with multiplicative approach
    print("\n4. COMPARISON WITH MULTIPLICATIVE:")
    print("-"*60)
    
    # Multiplicative-style correction
    def mult_objective(params):
        m0, m1, m2 = params
        mult_factor = 1 + m0 + m1 * X['CCT_norm'] + m2 * X['CCT_AL_ratio']
        pred = df['SRKT2_base'] * mult_factor
        return mean_absolute_error(df['PostOP Spherical Equivalent'], pred)
    
    from scipy.optimize import minimize
    result = minimize(mult_objective, x0=[0,0,0], bounds=[(-0.5,0.5)]*3)
    
    m0, m1, m2 = result.x
    formula_mult = f"SRKT2_corrected = SRKT2_base × (1 + {m0:.3f} + {m1:.3f}×((CCT-600)/100) + {m2:.3f}×(CCT/AL))"
    print(f"Multiplicative: {formula_mult}")
    print(f"   MAE: {result.fun:.3f} D")
    
    print(f"\nSVR-Linear:     {formula_simple}")
    print(f"   MAE: {simple_mae:.3f} D")
    
    # Show improvement
    improvement = (result.fun - simple_mae) / result.fun * 100
    print(f"\nImprovement: {improvement:.1f}% using additive correction")
    
    print("\n5. CLINICAL RECOMMENDATIONS:")
    print("-"*60)
    print("For manual calculation, use the SIMPLE LINEAR formula:")
    print(f"   {formula_simple}")
    print("\nFor computer/calculator, use EXTENDED LINEAR for better accuracy:")
    print(f"   {formula_extended[:100]}...")
    
    print("\n6. EXAMPLE CALCULATION:")
    print("-"*60)
    example_cct = 650
    example_al = 23.5
    example_srkt2 = 21.0
    
    cct_norm_ex = (example_cct - 600) / 100
    cct_al_ex = example_cct / example_al
    
    correction = intercept + coef_cct * cct_norm_ex + coef_ratio * cct_al_ex
    corrected = example_srkt2 + correction
    
    print(f"Patient: CCT={example_cct}μm, AL={example_al}mm")
    print(f"Base SRKT2 = {example_srkt2:.1f} D")
    print(f"Correction = {intercept:.3f} + {coef_cct:.3f}×{cct_norm_ex:.1f} + {coef_ratio:.3f}×{cct_al_ex:.2f}")
    print(f"Correction = {correction:.3f} D")
    print(f"Final IOL = {corrected:.1f} D")
    
    return {
        'simple_formula': formula_simple,
        'extended_formula': formula_extended,
        'coefficients': {
            'intercept': intercept,
            'cct_norm': coef_cct,
            'cct_al_ratio': coef_ratio
        }
    }


if __name__ == "__main__":
    # Load data
    df = pd.read_excel('FacoDMEK.xlsx', sheet_name='Data')
    
    # Create clinical formula
    result = create_svr_clinical_formula(df)
    
    print("\n" + "="*80)
    print("FORMULA FOR CLINICAL USE:")
    print(result['simple_formula'])
    print("="*80)