# FULL COMBINATION - ALL METHODS INCLUDING MULTIPLICATIVE AND SVR
# ================================================================
# PURPOSE: Test if combining ALL correction methods provides additional benefit
# This includes: Parameter + Multiplicative + SVR + Additive (Quadratic)

print("=" * 80)
print("FULL COMBINATION WITH ALL METHODS - EXPERIMENTAL")
print("=" * 80)

print("\n⚠️ WARNING: This combines redundant corrections (Multiplicative AND SVR)")
print("Both address CCT error - risk of overcorrection")
print("\n🔬 TESTING HYPOTHESIS: Can redundant corrections complement each other?")

from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import numpy as np
import pandas as pd

# Check if previous methods have been run
required_vars = ['seed_test_maes_param', 'seed_test_maes_mult', 'seed_test_maes_additive']
missing_vars = [var for var in required_vars if var not in locals()]
if missing_vars:
    print(f"\n❌ ERROR: Missing variables: {missing_vars}")
    print("Please run Parameter, Multiplicative, SVR, and Additive cells first!")
else:
    print("\n✅ All prerequisite methods detected. Proceeding with full combination...")
    
    # Store results
    seed_test_maes_all = []
    seed_train_maes_all = []
    seed_baseline_maes_all = []
    seed_improvements_all = []
    seed_overfit_ratios_all = []
    
    print("\n" + "="*80)
    print("RUNNING MULTI-SEED ANALYSIS WITH ALL METHODS")
    print("="*80)
    
    for seed_idx, SEED in enumerate(SEEDS, 1):
        print(f"\n{'='*40}")
        print(f"SEED {seed_idx}/{len(SEEDS)}: {SEED}")
        print(f"{'='*40}")
        
        # Split data
        X_train_all, X_test_all = train_test_split(df, test_size=0.25, random_state=SEED)
        X_train_all['K_avg'] = (X_train_all['Bio-Ks'] + X_train_all['Bio-Kf']) / 2
        X_test_all['K_avg'] = (X_test_all['Bio-Ks'] + X_test_all['Bio-Kf']) / 2
        
        print(f"📊 Split: {len(X_train_all)} train, {len(X_test_all)} test")
        
        # Calculate baseline
        for dataset in [X_train_all, X_test_all]:
            dataset['SRKT2_Baseline'] = dataset.apply(
                lambda row: calculate_SRKT2(
                    AL=row['Bio-AL'],
                    K_avg=row['K_avg'],
                    IOL_power=row['IOL Power'],
                    A_constant=row['A-Constant']
                ), axis=1
            )
        
        baseline_mae = mean_absolute_error(X_test_all['PostOP Spherical Equivalent'], 
                                           X_test_all['SRKT2_Baseline'])
        
        # Setup K-fold
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        
        # Store fold results
        param_results = []
        mult_results = []
        svr_models = []
        svr_scalers = []
        add_results = []
        
        print("\n📐 Training each method with 5-fold CV:")
        print("-" * 40)
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_all), 1):
            print(f"  Fold {fold_num}/5: ", end="")
            
            fold_train = X_train_all.iloc[train_idx]
            fold_val = X_train_all.iloc[val_idx]
            
            # 1. PARAMETER OPTIMIZATION
            def param_objective(params, df_data):
                A_mod, AL_mod, K_mod = params
                predictions = []
                for _, row in df_data.iterrows():
                    pred = calculate_SRKT2(
                        AL=row['Bio-AL'] + AL_mod,
                        K_avg=row['K_avg'] + K_mod,
                        IOL_power=row['IOL Power'],
                        A_constant=row['A-Constant'] + A_mod
                    )
                    predictions.append(pred)
                return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
            
            param_result = minimize(
                param_objective,
                x0=[0, 0, 0],
                args=(fold_train,),
                bounds=[(-2, 2), (-0.5, 0.5), (-2, 2)],
                method='L-BFGS-B'
            )
            param_results.append(param_result.x)
            
            # 2. MULTIPLICATIVE CORRECTION
            def mult_objective(params, df_data):
                m0, m1, m2 = params
                predictions = []
                for _, row in df_data.iterrows():
                    base_pred = row['SRKT2_Baseline']
                    cct_norm = (row['CCT'] - 600) / 100
                    cct_ratio = row['CCT'] / row['Bio-AL']
                    mult_factor = 1 + m0 + m1 * cct_norm + m2 * cct_ratio
                    predictions.append(base_pred * mult_factor)
                return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
            
            mult_result = minimize(
                mult_objective,
                x0=[0, 0, 0],
                args=(fold_train,),
                bounds=[(-0.5, 0.5), (-0.5, 0.5), (-10, 10)],
                method='L-BFGS-B'
            )
            mult_results.append(mult_result.x)
            
            # 3. SVR CORRECTION
            # Prepare features
            X_svr = pd.DataFrame()
            X_svr['CCT_norm'] = (fold_train['CCT'] - 600) / 100
            X_svr['AL'] = fold_train['Bio-AL']
            X_svr['ACD'] = fold_train['Bio-ACD']
            X_svr['K_mean'] = fold_train['K_avg']
            X_svr['CCT_AL_ratio'] = fold_train['CCT'] / fold_train['Bio-AL']
            
            # Calculate residuals (what SVR needs to correct)
            y_svr = fold_train['PostOP Spherical Equivalent'] - fold_train['SRKT2_Baseline']
            
            # Train SVR
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_svr)
            
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr.fit(X_scaled, y_svr)
            
            svr_models.append(svr)
            svr_scalers.append(scaler)
            
            # 4. ADDITIVE CORRECTION (Quadratic)
            def add_objective(params, df_data):
                a0, a1, a2, a3, a4 = params
                predictions = []
                for _, row in df_data.iterrows():
                    base_pred = row['SRKT2_Baseline']
                    cct_norm = (row['CCT'] - 600) / 100
                    cct_ratio = row['CCT'] / row['Bio-AL']
                    # Quadratic correction
                    correction = a0 + a1 * cct_norm + a2 * cct_ratio + a3 * row['K_avg'] + a4 * cct_norm**2
                    predictions.append(base_pred + correction)
                return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
            
            add_result = minimize(
                add_objective,
                x0=[0, 0, 0, 0, 0],
                args=(fold_train,),
                bounds=[(-2, 2), (-2, 2), (-2, 2), (-0.1, 0.1), (-2, 2)],
                method='L-BFGS-B'
            )
            add_results.append(add_result.x)
            
            print("✓ ", end="")
        
        print("\n\n🔄 Combining all methods...")
        
        # Average parameters across folds
        avg_param = np.mean(param_results, axis=0)
        avg_mult = np.mean(mult_results, axis=0)
        avg_add = np.mean(add_results, axis=0)
        
        A_mod, AL_mod, K_mod = avg_param
        m0, m1, m2 = avg_mult
        a0, a1, a2, a3, a4 = avg_add
        
        print(f"\n📊 Optimized Parameters:")
        print(f"  Parameter: A={A_mod:.3f}, AL={AL_mod:.3f}, K={K_mod:.3f}")
        print(f"  Multiplicative: m0={m0:.3f}, m1={m1:.3f}, m2={m2:.3f}")
        print(f"  Additive (Quad): a0={a0:.3f}, a1={a1:.3f}, a2={a2:.3f}, a3={a3:.3f}, a4={a4:.3f}")
        
        # Apply ALL corrections to test set
        print("\n🎯 Applying full combination to test set:")
        all_predictions = []
        
        for _, row in X_test_all.iterrows():
            # Step 1: Parameter optimization
            param_modified = calculate_SRKT2(
                AL=row['Bio-AL'] + AL_mod,
                K_avg=row['K_avg'] + K_mod,
                IOL_power=row['IOL Power'],
                A_constant=row['A-Constant'] + A_mod
            )
            
            # Step 2: Multiplicative correction
            cct_norm = (row['CCT'] - 600) / 100
            cct_ratio = row['CCT'] / row['Bio-AL']
            mult_factor = 1 + m0 + m1 * cct_norm + m2 * cct_ratio
            after_mult = param_modified * mult_factor
            
            # Step 3: SVR correction (average across fold models)
            X_svr_test = pd.DataFrame({
                'CCT_norm': [cct_norm],
                'AL': [row['Bio-AL']],
                'ACD': [row['Bio-ACD']],
                'K_mean': [row['K_avg']],
                'CCT_AL_ratio': [cct_ratio]
            })
            
            svr_corrections = []
            for svr_model, scaler in zip(svr_models, svr_scalers):
                X_scaled = scaler.transform(X_svr_test)
                svr_pred = svr_model.predict(X_scaled)[0]
                svr_corrections.append(svr_pred)
            avg_svr_correction = np.mean(svr_corrections)
            
            after_svr = after_mult + avg_svr_correction
            
            # Step 4: Additive quadratic correction
            add_correction = a0 + a1 * cct_norm + a2 * cct_ratio + a3 * row['K_avg'] + a4 * cct_norm**2
            
            final_prediction = after_svr + add_correction
            all_predictions.append(final_prediction)
        
        # Calculate performance on test set
        test_mae = mean_absolute_error(X_test_all['PostOP Spherical Equivalent'], all_predictions)
        
        # Calculate on training set for overfitting check
        train_predictions = []
        for _, row in X_train_all.iterrows():
            param_modified = calculate_SRKT2(
                AL=row['Bio-AL'] + AL_mod,
                K_avg=row['K_avg'] + K_mod,
                IOL_power=row['IOL Power'],
                A_constant=row['A-Constant'] + A_mod
            )
            
            cct_norm = (row['CCT'] - 600) / 100
            cct_ratio = row['CCT'] / row['Bio-AL']
            mult_factor = 1 + m0 + m1 * cct_norm + m2 * cct_ratio
            after_mult = param_modified * mult_factor
            
            X_svr_test = pd.DataFrame({
                'CCT_norm': [cct_norm],
                'AL': [row['Bio-AL']],
                'ACD': [row['Bio-ACD']],
                'K_mean': [row['K_avg']],
                'CCT_AL_ratio': [cct_ratio]
            })
            
            svr_corrections = []
            for svr_model, scaler in zip(svr_models, svr_scalers):
                X_scaled = scaler.transform(X_svr_test)
                svr_pred = svr_model.predict(X_scaled)[0]
                svr_corrections.append(svr_pred)
            avg_svr_correction = np.mean(svr_corrections)
            
            after_svr = after_mult + avg_svr_correction
            add_correction = a0 + a1 * cct_norm + a2 * cct_ratio + a3 * row['K_avg'] + a4 * cct_norm**2
            
            final_prediction = after_svr + add_correction
            train_predictions.append(final_prediction)
        
        train_mae = mean_absolute_error(X_train_all['PostOP Spherical Equivalent'], train_predictions)
        
        # Calculate metrics
        improvement = ((baseline_mae - test_mae) / baseline_mae) * 100
        overfit_ratio = train_mae / test_mae
        
        print(f"\n📈 RESULTS FOR SEED {SEED}:")
        print(f"  Baseline MAE: {baseline_mae:.4f} D")
        print(f"  Test MAE: {test_mae:.4f} D")
        print(f"  Train MAE: {train_mae:.4f} D")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Overfit ratio: {overfit_ratio:.3f}")
        
        # Store results
        seed_test_maes_all.append(test_mae)
        seed_train_maes_all.append(train_mae)
        seed_baseline_maes_all.append(baseline_mae)
        seed_improvements_all.append(improvement)
        seed_overfit_ratios_all.append(overfit_ratio)
    
    # Summary statistics
    print("\n" + "="*80)
    print("MULTI-SEED SUMMARY - ALL METHODS COMBINED")
    print("="*80)
    
    print(f"\n📊 TEST SET PERFORMANCE (n={len(SEEDS)} seeds):")
    print(f"  Mean MAE: {np.mean(seed_test_maes_all):.4f} ± {np.std(seed_test_maes_all):.4f} D")
    print(f"  Best MAE: {np.min(seed_test_maes_all):.4f} D")
    print(f"  Worst MAE: {np.max(seed_test_maes_all):.4f} D")
    
    print(f"\n📈 IMPROVEMENT OVER BASELINE:")
    print(f"  Mean: {np.mean(seed_improvements_all):.1f} ± {np.std(seed_improvements_all):.1f}%")
    
    print(f"\n⚠️ OVERFITTING ANALYSIS:")
    print(f"  Mean overfit ratio: {np.mean(seed_overfit_ratios_all):.3f}")
    if np.mean(seed_overfit_ratios_all) < 0.9:
        print("  ❌ HIGH OVERFITTING DETECTED (ratio < 0.9)")
        print("  The model performs much better on training than test data")
    elif np.mean(seed_overfit_ratios_all) < 0.95:
        print("  ⚠️ MODERATE OVERFITTING (ratio 0.9-0.95)")
    else:
        print("  ✅ Low overfitting - good generalization")
    
    print("\n🔬 INTERPRETATION:")
    print("-" * 50)
    
    # Compare with other methods if available
    if 'seed_test_maes_combined' in locals():
        standard_combined_mae = np.mean(seed_test_maes_combined)
        all_methods_mae = np.mean(seed_test_maes_all)
        
        if all_methods_mae < standard_combined_mae:
            improvement_vs_standard = ((standard_combined_mae - all_methods_mae) / standard_combined_mae) * 100
            print(f"✅ ALL methods combined BEATS standard combined by {improvement_vs_standard:.1f}%")
            print("   Redundant corrections appear to be complementary!")
        else:
            worse_by = ((all_methods_mae - standard_combined_mae) / standard_combined_mae) * 100
            print(f"❌ ALL methods combined is {worse_by:.1f}% WORSE than standard combined")
            print("   Redundant corrections lead to overcorrection")
    
    if 'seed_test_maes_mult' in locals():
        svr_mae = np.mean(seed_test_maes_mult)
        all_methods_mae = np.mean(seed_test_maes_all)
        
        if all_methods_mae < svr_mae:
            print(f"\n✅ Full combination outperforms SVR/Mult alone")
        else:
            print(f"\n❌ Adding redundant corrections doesn't help")
            print("   SVR or Multiplicative alone is sufficient for CCT correction")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    
    if np.mean(seed_overfit_ratios_all) < 0.9:
        print("❌ NOT RECOMMENDED - Too many parameters lead to overfitting")
    elif 'seed_test_maes_combined' in locals() and np.mean(seed_test_maes_all) > np.mean(seed_test_maes_combined):
        print("❌ NOT RECOMMENDED - Standard combined (without redundancy) performs better")
    else:
        print("🤔 FURTHER TESTING NEEDED - Results are inconclusive")
    
    print("\nThis was an experimental test of redundant corrections.")
    print("Generally, using either Multiplicative OR SVR (not both) is recommended.")