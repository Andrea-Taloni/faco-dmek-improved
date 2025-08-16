# COMBINED APPROACH WITH K-FOLD CROSS-VALIDATION - MULTI-SEED
# ========================================================
# PURPOSE: Combine all three methods with nested K-fold CV and multi-seed validation
# NOW USES THE BEST POLYNOMIAL DEGREE FROM ADDITIVE ANALYSIS

print("=" * 80)
print("COMBINED FORMULA WITH K-FOLD CV - MULTI-SEED ANALYSIS")
print("=" * 80)

# Determine which polynomial degree to use from additive cell results
if 'best_degree' in locals():
    print(f"\n📐 Using {best_degree.upper()} polynomial degree (determined optimal in additive cell)")
else:
    print("\n⚠️ No polynomial analysis found, defaulting to LINEAR")
    best_degree = 'linear'

print("\n🎯 MULTI-SEED NESTED CV FOR COMBINED APPROACH:")
print("-" * 50)
print(f"• Testing {len(SEEDS)} different random seeds: {SEEDS}")
print("• Each seed: 75/25 train/test split")
print("• Inner: 5-fold CV for each method")
print(f"• Additive correction using: {best_degree} polynomial")

from sklearn.model_selection import train_test_split, KFold
from scipy.optimize import minimize, differential_evolution
import numpy as np

# Store results for each seed
seed_results_combined = []
seed_test_maes_combined = []
seed_train_maes_combined = []
seed_baseline_maes_combined = []
seed_improvements_combined = []
seed_overfit_ratios_combined = []

# Store individual method results
seed_param_results = []
seed_mult_results = []
seed_add_results = []

print("\n" + "="*80)
print("RUNNING MULTI-SEED ANALYSIS")
print("="*80)

for seed_idx, SEED in enumerate(SEEDS, 1):
    print(f"\n{'='*40}")
    print(f"SEED {seed_idx}/{len(SEEDS)}: {SEED}")
    print(f"{'='*40}")
    
    # OUTER SPLIT - consistent across all methods
    X_train_comb, X_test_comb = train_test_split(df, test_size=0.25, random_state=SEED)
    X_train_comb['K_avg'] = (X_train_comb['Bio-Ks'] + X_train_comb['Bio-Kf']) / 2
    X_test_comb['K_avg'] = (X_test_comb['Bio-Ks'] + X_test_comb['Bio-Kf']) / 2
    
    print(f"📊 Split: {len(X_train_comb)} train, {len(X_test_comb)} test")
    
    # Calculate baseline for all
    for dataset in [X_train_comb, X_test_comb]:
        dataset['SRKT2_Baseline'] = dataset.apply(
            lambda row: calculate_SRKT2(
                AL=row['Bio-AL'],
                K_avg=row['K_avg'],
                IOL_power=row['IOL Power'],
                A_constant=row['A-Constant']
            ), axis=1
        )
    
    print("\n📁 K-FOLD CV FOR EACH METHOD:")
    print("-" * 40)
    
    # Setup K-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Store fold results for each method
    param_fold_results = []
    mult_fold_results = []
    add_fold_results = []
    combined_fold_maes = []
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_comb), 1):
        print(f"  Fold {fold_num}/5: ", end="")
        
        fold_train = X_train_comb.iloc[train_idx]
        fold_val = X_train_comb.iloc[val_idx]
        
        # 1. PARAMETER METHOD
        def param_obj(params, df_data):
            nc_base, nc_cct, k_base, k_cct, acd_base, acd_cct = params
            predictions = []
            for _, row in df_data.iterrows():
                cct_norm = (row['CCT'] - 600) / 100
                nc = nc_base + nc_cct * cct_norm
                k_index = k_base + k_cct * cct_norm
                acd_offset = acd_base + acd_cct * cct_norm
                pred = calculate_SRKT2(
                    AL=row['Bio-AL'], K_avg=row['K_avg'],
                    IOL_power=row['IOL Power'],
                    A_constant=row['A-Constant'] + acd_offset,
                    nc=nc, k_index=k_index
                )
                predictions.append(pred)
            return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
        
        bounds_p = [(1.20, 1.50), (-0.20, 0.20), (1.20, 1.60), (-0.30, 0.30), (-3.0, 3.0), (-3.0, 3.0)]
        result_p = differential_evolution(lambda p: param_obj(p, fold_train), bounds_p, 
                                         maxiter=20, seed=SEED+fold_num, disp=False)
        param_fold_results.append(result_p.x)
        
        # 2. MULTIPLICATIVE METHOD
        def mult_obj(params, df_data):
            m0, m1, m2 = params
            predictions = []
            for _, row in df_data.iterrows():
                base_pred = row['SRKT2_Baseline']
                cct_norm = (row['CCT'] - 600) / 100
                cct_ratio = row['CCT'] / row['Bio-AL']
                correction = 1 + m0 + m1 * cct_norm + m2 * cct_ratio
                predictions.append(base_pred * correction)
            return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
        
        result_m = minimize(lambda p: mult_obj(p, fold_train), [0,0,0], 
                           method='L-BFGS-B', bounds=[(-0.5,0.5)]*3)
        mult_fold_results.append(result_m.x)
        
        # 3. ADDITIVE METHOD - WITH BEST POLYNOMIAL DEGREE
        if best_degree == 'linear':
            def add_obj(params, df_data):
                a0, a1, a2, a3 = params
                predictions = []
                for _, row in df_data.iterrows():
                    base_pred = row['SRKT2_Baseline']
                    cct_norm = (row['CCT'] - 600) / 100
                    cct_ratio = row['CCT'] / row['Bio-AL']
                    correction = a0 + a1 * cct_norm + a2 * cct_ratio + a3 * row['K_avg']
                    predictions.append(base_pred + correction)
                return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
            
            add_bounds = [(-2,2),(-2,2),(-2,2),(-0.1,0.1)]
            add_initial = [0,0,0,0]
            
        elif best_degree == 'quadratic':
            def add_obj(params, df_data):
                a0, a1, a2, a3, a4 = params
                predictions = []
                for _, row in df_data.iterrows():
                    base_pred = row['SRKT2_Baseline']
                    cct_norm = (row['CCT'] - 600) / 100
                    cct_ratio = row['CCT'] / row['Bio-AL']
                    correction = (a0 + a1 * cct_norm + a2 * cct_ratio + 
                                a3 * row['K_avg'] + a4 * cct_norm**2)
                    predictions.append(base_pred + correction)
                return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
            
            add_bounds = [(-2,2),(-2,2),(-2,2),(-0.1,0.1),(-1,1)]
            add_initial = [0,0,0,0,0]
            
        else:  # cubic
            def add_obj(params, df_data):
                a0, a1, a2, a3, a4, a5 = params
                predictions = []
                for _, row in df_data.iterrows():
                    base_pred = row['SRKT2_Baseline']
                    cct_norm = (row['CCT'] - 600) / 100
                    cct_ratio = row['CCT'] / row['Bio-AL']
                    correction = (a0 + a1 * cct_norm + a2 * cct_ratio + 
                                a3 * row['K_avg'] + a4 * cct_norm**2 + 
                                a5 * cct_norm**3)
                    predictions.append(base_pred + correction)
                return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
            
            add_bounds = [(-2,2),(-2,2),(-2,2),(-0.1,0.1),(-1,1),(-0.5,0.5)]
            add_initial = [0,0,0,0,0,0]
        
        result_a = minimize(lambda p: add_obj(p, fold_train), add_initial,
                           method='L-BFGS-B', bounds=add_bounds)
        add_fold_results.append(result_a.x)
        
        # VALIDATE COMBINED on fold validation set
        nc_b, nc_c, k_b, k_c, acd_b, acd_c = result_p.x
        m0, m1, m2 = result_m.x
        
        combined_preds = []
        for _, row in fold_val.iterrows():
            cct_norm = (row['CCT'] - 600) / 100
            cct_ratio = row['CCT'] / row['Bio-AL']
            
            # Modified SRK/T2
            nc = nc_b + nc_c * cct_norm
            k_index = k_b + k_c * cct_norm
            acd_offset = acd_b + acd_c * cct_norm
            modified = calculate_SRKT2(
                AL=row['Bio-AL'], K_avg=row['K_avg'],
                IOL_power=row['IOL Power'],
                A_constant=row['A-Constant'] + acd_offset,
                nc=nc, k_index=k_index
            )
            
            # Apply multiplicative
            mult_factor = 1 + m0 + m1 * cct_norm + m2 * cct_ratio
            after_mult = modified * mult_factor
            
            # Apply additive with appropriate polynomial
            if best_degree == 'linear':
                a0, a1, a2, a3 = result_a.x
                add_correction = a0 + a1 * cct_norm + a2 * cct_ratio + a3 * row['K_avg']
            elif best_degree == 'quadratic':
                a0, a1, a2, a3, a4 = result_a.x
                add_correction = (a0 + a1 * cct_norm + a2 * cct_ratio + 
                                a3 * row['K_avg'] + a4 * cct_norm**2)
            else:  # cubic
                a0, a1, a2, a3, a4, a5 = result_a.x
                add_correction = (a0 + a1 * cct_norm + a2 * cct_ratio + 
                                a3 * row['K_avg'] + a4 * cct_norm**2 + 
                                a5 * cct_norm**3)
            
            final = after_mult + add_correction
            combined_preds.append(final)
        
        fold_mae = mean_absolute_error(fold_val['PostOP Spherical Equivalent'], combined_preds)
        combined_fold_maes.append(fold_mae)
        print(f"MAE={fold_mae:.4f} ", end="")
    
    print()  # New line after folds
    
    # Average parameters across folds
    avg_param = np.mean(param_fold_results, axis=0)
    avg_mult = np.mean(mult_fold_results, axis=0)
    avg_add = np.mean(add_fold_results, axis=0)
    avg_combined_mae = np.mean(combined_fold_maes)
    std_combined_mae = np.std(combined_fold_maes)
    
    print(f"  CV MAE: {avg_combined_mae:.4f} ± {std_combined_mae:.4f} D")
    
    # FINAL RETRAINING on full training set
    print("  Final optimization on full training set...")
    
    result_p_final = differential_evolution(lambda p: param_obj(p, X_train_comb), bounds_p, 
                                           maxiter=50, seed=SEED, disp=False)
    nc_base_c, nc_cct_c, k_base_c, k_cct_c, acd_base_c, acd_cct_c = result_p_final.x
    
    result_m_final = minimize(lambda p: mult_obj(p, X_train_comb), [0,0,0], 
                             method='L-BFGS-B', bounds=[(-0.5,0.5)]*3)
    m0_c, m1_c, m2_c = result_m_final.x
    
    result_a_final = minimize(lambda p: add_obj(p, X_train_comb), add_initial,
                             method='L-BFGS-B', bounds=add_bounds)
    
    # EVALUATE ON TRAINING SET (for overfitting check)
    predictions_combined_train = []
    for _, row in X_train_comb.iterrows():
        cct_norm = (row['CCT'] - 600) / 100
        cct_ratio = row['CCT'] / row['Bio-AL']
        k_avg = row['K_avg']
        
        # Modified SRK/T2 with optimized parameters
        nc = nc_base_c + nc_cct_c * cct_norm
        k_index = k_base_c + k_cct_c * cct_norm
        acd_offset = acd_base_c + acd_cct_c * cct_norm
        modified = calculate_SRKT2(
            AL=row['Bio-AL'], K_avg=k_avg,
            IOL_power=row['IOL Power'],
            A_constant=row['A-Constant'] + acd_offset,
            nc=nc, k_index=k_index
        )
        
        # Apply multiplicative correction
        mult_factor = 1 + m0_c + m1_c * cct_norm + m2_c * cct_ratio
        after_mult = modified * mult_factor
        
        # Apply additive correction with polynomial
        if best_degree == 'linear':
            a0_c, a1_c, a2_c, a3_c = result_a_final.x
            add_correction = a0_c + a1_c * cct_norm + a2_c * cct_ratio + a3_c * k_avg
        elif best_degree == 'quadratic':
            a0_c, a1_c, a2_c, a3_c, a4_c = result_a_final.x
            add_correction = (a0_c + a1_c * cct_norm + a2_c * cct_ratio + 
                            a3_c * k_avg + a4_c * cct_norm**2)
        else:  # cubic
            a0_c, a1_c, a2_c, a3_c, a4_c, a5_c = result_a_final.x
            add_correction = (a0_c + a1_c * cct_norm + a2_c * cct_ratio + 
                            a3_c * k_avg + a4_c * cct_norm**2 + a5_c * cct_norm**3)
        
        final = after_mult + add_correction
        predictions_combined_train.append(final)
    
    train_mae_combined = mean_absolute_error(X_train_comb['PostOP Spherical Equivalent'], 
                                            predictions_combined_train)
    
    # EVALUATE ON TEST SET
    predictions_combined_test = []
    for _, row in X_test_comb.iterrows():
        cct_norm = (row['CCT'] - 600) / 100
        cct_ratio = row['CCT'] / row['Bio-AL']
        k_avg = row['K_avg']
        
        # Modified SRK/T2 with optimized parameters
        nc = nc_base_c + nc_cct_c * cct_norm
        k_index = k_base_c + k_cct_c * cct_norm
        acd_offset = acd_base_c + acd_cct_c * cct_norm
        modified = calculate_SRKT2(
            AL=row['Bio-AL'], K_avg=k_avg,
            IOL_power=row['IOL Power'],
            A_constant=row['A-Constant'] + acd_offset,
            nc=nc, k_index=k_index
        )
        
        # Apply multiplicative correction
        mult_factor = 1 + m0_c + m1_c * cct_norm + m2_c * cct_ratio
        after_mult = modified * mult_factor
        
        # Apply additive correction with polynomial
        if best_degree == 'linear':
            add_correction = a0_c + a1_c * cct_norm + a2_c * cct_ratio + a3_c * k_avg
        elif best_degree == 'quadratic':
            add_correction = (a0_c + a1_c * cct_norm + a2_c * cct_ratio + 
                            a3_c * k_avg + a4_c * cct_norm**2)
        else:  # cubic
            add_correction = (a0_c + a1_c * cct_norm + a2_c * cct_ratio + 
                            a3_c * k_avg + a4_c * cct_norm**2 + a5_c * cct_norm**3)
        
        final = after_mult + add_correction
        predictions_combined_test.append(final)
    
    test_mae_combined = mean_absolute_error(X_test_comb['PostOP Spherical Equivalent'], 
                                           predictions_combined_test)
    baseline_mae_combined = mean_absolute_error(X_test_comb['PostOP Spherical Equivalent'], 
                                                X_test_comb['SRKT2_Baseline'])
    
    improvement_combined = ((baseline_mae_combined - test_mae_combined) / baseline_mae_combined) * 100
    overfit_ratio = test_mae_combined / train_mae_combined if train_mae_combined > 0 else float('inf')
    
    print("\n📈 RESULTS:")
    print("-" * 40)
    print(f"  Train MAE: {train_mae_combined:.4f} D")
    print(f"  Test MAE:  {test_mae_combined:.4f} D")
    print(f"  Baseline:  {baseline_mae_combined:.4f} D")
    print(f"  Improvement: {improvement_combined:.1f}%")
    print(f"  Overfit ratio: {overfit_ratio:.3f}")
    
    # Store results
    seed_results_combined.append({
        'seed': SEED,
        'param_values': result_p_final.x,
        'mult_values': result_m_final.x,
        'add_values': result_a_final.x,
        'train_mae': train_mae_combined,
        'test_mae': test_mae_combined,
        'baseline_mae': baseline_mae_combined,
        'improvement': improvement_combined,
        'overfit_ratio': overfit_ratio
    })
    
    seed_test_maes_combined.append(test_mae_combined)
    seed_train_maes_combined.append(train_mae_combined)
    seed_baseline_maes_combined.append(baseline_mae_combined)
    seed_improvements_combined.append(improvement_combined)
    seed_overfit_ratios_combined.append(overfit_ratio)
    
    seed_param_results.append(result_p_final.x)
    seed_mult_results.append(result_m_final.x)
    seed_add_results.append(result_a_final.x)

# SUMMARY STATISTICS
print("\n" + "="*80)
print(f"MULTI-SEED SUMMARY - COMBINED APPROACH WITH {best_degree.upper()} ADDITIVE")
print("="*80)

print(f"\n📊 PERFORMANCE ACROSS {len(SEEDS)} SEEDS:")
print("-" * 50)
print(f"Test MAE:     {np.mean(seed_test_maes_combined):.4f} ± {np.std(seed_test_maes_combined):.4f} D")
print(f"Train MAE:    {np.mean(seed_train_maes_combined):.4f} ± {np.std(seed_train_maes_combined):.4f} D")
print(f"Baseline MAE: {np.mean(seed_baseline_maes_combined):.4f} ± {np.std(seed_baseline_maes_combined):.4f} D")
print(f"Improvement:  {np.mean(seed_improvements_combined):.1f}% ± {np.std(seed_improvements_combined):.1f}%")
print(f"Overfit ratio: {np.mean(seed_overfit_ratios_combined):.3f} ± {np.std(seed_overfit_ratios_combined):.3f}")

# Parameter consistency analysis
print("\n🔬 PARAMETER CONSISTENCY:")
print("-" * 50)

param_names = ['nc_base', 'nc_cct', 'k_base', 'k_cct', 'acd_base', 'acd_cct']
param_array = np.array(seed_param_results)
print("\nParameter optimization values:")
for i, name in enumerate(param_names):
    values = param_array[:, i]
    print(f"  {name:10s}: {np.mean(values):7.4f} ± {np.std(values):.4f}")

mult_names = ['m0', 'm1_cct', 'm2_ratio']
mult_array = np.array(seed_mult_results)
print("\nMultiplicative correction values:")
for i, name in enumerate(mult_names):
    values = mult_array[:, i]
    print(f"  {name:10s}: {np.mean(values):7.4f} ± {np.std(values):.4f}")

add_array = np.array(seed_add_results)
print(f"\nAdditive correction values ({best_degree}):")
if best_degree == 'linear':
    add_names = ['a0', 'a1_cct', 'a2_ratio', 'a3_K']
elif best_degree == 'quadratic':
    add_names = ['a0', 'a1_cct', 'a2_ratio', 'a3_K', 'a4_cct2']
else:  # cubic
    add_names = ['a0', 'a1_cct', 'a2_ratio', 'a3_K', 'a4_cct2', 'a5_cct3']

for i, name in enumerate(add_names):
    values = add_array[:, i]
    print(f"  {name:10s}: {np.mean(values):7.4f} ± {np.std(values):.4f}")

# Clinical significance
mae_mean = np.mean(seed_test_maes_combined)
mae_std = np.std(seed_test_maes_combined)
print("\n" + "="*80)
print("CLINICAL INTERPRETATION")
print("="*80)
print(f"✅ Combined approach with {best_degree} additive achieves:")
print(f"   • Mean absolute error: {mae_mean:.3f} ± {mae_std:.3f} D")
print(f"   • {np.mean(seed_improvements_combined):.0f}% improvement over standard SRK/T2")

if mae_mean < 0.5:
    print("   • EXCELLENT: Within ±0.50 D target for most patients")
elif mae_mean < 0.75:
    print("   • GOOD: Within ±0.75 D for most patients")
else:
    print("   • MODERATE: Further optimization may be beneficial")