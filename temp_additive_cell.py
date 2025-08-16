# ADDITIVE CORRECTION WITH POLYNOMIAL TERMS - MULTI-SEED
# ========================================================
# PURPOSE: Create an additive correction with polynomial CCT terms
# NOW WITH QUADRATIC AND CUBIC CCT TERMS for better non-linear modeling

# ⚙️ ACTIVATION CONTROL - Set to True to run full polynomial comparison
RUN_POLYNOMIAL_COMPARISON = False  # 🔴 DISABLED - Using direct quadratic approach instead

if RUN_POLYNOMIAL_COMPARISON:
    print("=" * 80)
    print("ADDITIVE CORRECTION WITH POLYNOMIAL CCT TERMS - MULTI-SEED ANALYSIS")
    print("=" * 80)

    print("\n🎯 TESTING POLYNOMIAL (QUADRATIC & CUBIC) CCT TERMS:")
    print("-" * 50)
    print("• Linear model: a0 + a1*CCT_norm + a2*CCT_ratio + a3*K_avg")
    print("• Quadratic model: + a4*CCT_norm²")  
    print("• Cubic model: + a4*CCT_norm² + a5*CCT_norm³")
    print(f"• Testing {len(SEEDS)} different random seeds: {SEEDS}")
    print("• Each seed: 75/25 train/test split")
    print("• Inner: 5-fold cross-validation")

    from sklearn.model_selection import train_test_split, KFold
    from scipy.optimize import minimize
    import numpy as np

    # Store results for different polynomial degrees
    results_by_degree = {
        'linear': {'test_maes': [], 'train_maes': [], 'improvements': [], 'params': []},
        'quadratic': {'test_maes': [], 'train_maes': [], 'improvements': [], 'params': []},
        'cubic': {'test_maes': [], 'train_maes': [], 'improvements': [], 'params': []}
    }

    print("\n" + "="*80)
    print("RUNNING MULTI-SEED ANALYSIS WITH POLYNOMIAL TERMS")
    print("="*80)

    for seed_idx, SEED in enumerate(SEEDS, 1):
        print(f"\n{'='*40}")
        print(f"SEED {seed_idx}/{len(SEEDS)}: {SEED}")
        print(f"{'='*40}")
        
        # Split data
        X_train_add, X_test_add = train_test_split(df, test_size=0.25, random_state=SEED)
        X_train_add['K_avg'] = (X_train_add['Bio-Ks'] + X_train_add['Bio-Kf']) / 2
        X_test_add['K_avg'] = (X_test_add['Bio-Ks'] + X_test_add['Bio-Kf']) / 2
        
        print(f"📊 Split: {len(X_train_add)} train, {len(X_test_add)} test")
        
        # Calculate baseline
        for dataset in [X_train_add, X_test_add]:
            dataset['SRKT2_Baseline'] = dataset.apply(
                lambda row: calculate_SRKT2(
                    AL=row['Bio-AL'],
                    K_avg=row['K_avg'],
                    IOL_power=row['IOL Power'],
                    A_constant=row['A-Constant']
                ), axis=1
            )
        
        baseline_mae = mean_absolute_error(X_test_add['PostOP Spherical Equivalent'], 
                                           X_test_add['SRKT2_Baseline'])
        
        # Test each polynomial degree
        for degree_name in ['linear', 'quadratic', 'cubic']:
            print(f"\n📐 Testing {degree_name.upper()} model:")
            print("-" * 40)
            
            # Setup K-fold
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_results = []
            fold_maes = []
            
            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_add), 1):
                print(f"  Fold {fold_num}/5: ", end="")
                
                fold_train = X_train_add.iloc[train_idx]
                fold_val = X_train_add.iloc[val_idx]
                
                # Define objective function based on degree
                if degree_name == 'linear':
                    def additive_objective(params, df_data):
                        a0, a1, a2, a3 = params
                        predictions = []
                        for _, row in df_data.iterrows():
                            base_pred = row['SRKT2_Baseline']
                            cct_norm = (row['CCT'] - 600) / 100
                            cct_ratio = row['CCT'] / row['Bio-AL']
                            # Linear only
                            correction = a0 + a1 * cct_norm + a2 * cct_ratio + a3 * row['K_avg']
                            predictions.append(base_pred + correction)
                        return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
                    
                    bounds = [(-2, 2), (-2, 2), (-2, 2), (-0.1, 0.1)]
                    initial = [0, 0, 0, 0]
                    
                elif degree_name == 'quadratic':
                    def additive_objective(params, df_data):
                        a0, a1, a2, a3, a4 = params
                        predictions = []
                        for _, row in df_data.iterrows():
                            base_pred = row['SRKT2_Baseline']
                            cct_norm = (row['CCT'] - 600) / 100
                            cct_ratio = row['CCT'] / row['Bio-AL']
                            # Linear + quadratic
                            correction = (a0 + a1 * cct_norm + a2 * cct_ratio + 
                                        a3 * row['K_avg'] + a4 * cct_norm**2)
                            predictions.append(base_pred + correction)
                        return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
                    
                    bounds = [(-2, 2), (-2, 2), (-2, 2), (-0.1, 0.1), (-1, 1)]
                    initial = [0, 0, 0, 0, 0]
                    
                else:  # cubic
                    def additive_objective(params, df_data):
                        a0, a1, a2, a3, a4, a5 = params
                        predictions = []
                        for _, row in df_data.iterrows():
                            base_pred = row['SRKT2_Baseline']
                            cct_norm = (row['CCT'] - 600) / 100
                            cct_ratio = row['CCT'] / row['Bio-AL']
                            # Linear + quadratic + cubic
                            correction = (a0 + a1 * cct_norm + a2 * cct_ratio + 
                                        a3 * row['K_avg'] + a4 * cct_norm**2 + 
                                        a5 * cct_norm**3)
                            predictions.append(base_pred + correction)
                        return mean_absolute_error(df_data['PostOP Spherical Equivalent'], predictions)
                    
                    bounds = [(-2, 2), (-2, 2), (-2, 2), (-0.1, 0.1), (-1, 1), (-0.5, 0.5)]
                    initial = [0, 0, 0, 0, 0, 0]
                
                # Optimize
                result = minimize(lambda p: additive_objective(p, fold_train), 
                                initial, method='L-BFGS-B', bounds=bounds)
                fold_results.append(result.x)
                
                # Validate
                fold_val_mae = additive_objective(result.x, fold_val)
                fold_maes.append(fold_val_mae)
                print(f"MAE={fold_val_mae:.4f} ", end="")
            
            print()
            avg_cv_mae = np.mean(fold_maes)
            std_cv_mae = np.std(fold_maes)
            print(f"  CV MAE: {avg_cv_mae:.4f} ± {std_cv_mae:.4f} D")
            
            # Final optimization on full training set
            print(f"  Final optimization on full training set...")
            final_result = minimize(lambda p: additive_objective(p, X_train_add), 
                                  initial, method='L-BFGS-B', bounds=bounds)
            
            # Evaluate on training set
            train_mae = additive_objective(final_result.x, X_train_add)
            
            # Evaluate on test set
            test_mae = additive_objective(final_result.x, X_test_add)
            
            improvement = ((baseline_mae - test_mae) / baseline_mae) * 100
            overfit_ratio = test_mae / train_mae if train_mae > 0 else float('inf')
            
            print(f"\n  📈 RESULTS ({degree_name}):")
            print(f"    Train MAE: {train_mae:.4f} D")
            print(f"    Test MAE:  {test_mae:.4f} D")
            print(f"    Baseline:  {baseline_mae:.4f} D")
            print(f"    Improvement: {improvement:.1f}%")
            print(f"    Overfit ratio: {overfit_ratio:.3f}")
            
            # Store results
            results_by_degree[degree_name]['test_maes'].append(test_mae)
            results_by_degree[degree_name]['train_maes'].append(train_mae)
            results_by_degree[degree_name]['improvements'].append(improvement)
            results_by_degree[degree_name]['params'].append(final_result.x)

    # COMPREHENSIVE COMPARISON
    print("\n" + "="*80)
    print("POLYNOMIAL COMPARISON SUMMARY")
    print("="*80)

    print(f"\n📊 PERFORMANCE ACROSS {len(SEEDS)} SEEDS:")
    print("-" * 50)

    for degree_name in ['linear', 'quadratic', 'cubic']:
        results = results_by_degree[degree_name]
        print(f"\n{degree_name.upper()} MODEL:")
        print(f"  Test MAE:     {np.mean(results['test_maes']):.4f} ± {np.std(results['test_maes']):.4f} D")
        print(f"  Train MAE:    {np.mean(results['train_maes']):.4f} ± {np.std(results['train_maes']):.4f} D")
        print(f"  Improvement:  {np.mean(results['improvements']):.1f}% ± {np.std(results['improvements']):.1f}%")
        print(f"  Overfit gap:  {np.mean(results['test_maes']) - np.mean(results['train_maes']):.4f} D")

    # Parameter analysis
    print("\n🔬 PARAMETER ANALYSIS:")
    print("-" * 50)

    # Analyze quadratic coefficients
    quad_params = np.array(results_by_degree['quadratic']['params'])
    if quad_params.shape[1] >= 5:
        quad_coeffs = quad_params[:, 4]  # a4 (quadratic term)
        print(f"\nQuadratic coefficient (a4): {np.mean(quad_coeffs):.4f} ± {np.std(quad_coeffs):.4f}")
        print(f"  Significance: {'YES' if abs(np.mean(quad_coeffs)) > 0.1 else 'MARGINAL'}")

    # Analyze cubic coefficients
    cubic_params = np.array(results_by_degree['cubic']['params'])
    if cubic_params.shape[1] >= 6:
        cubic_coeffs = cubic_params[:, 5]  # a5 (cubic term)
        print(f"\nCubic coefficient (a5): {np.mean(cubic_coeffs):.4f} ± {np.std(cubic_coeffs):.4f}")
        print(f"  Significance: {'YES' if abs(np.mean(cubic_coeffs)) > 0.05 else 'MARGINAL'}")

    # Winner determination
    mean_test_maes = {degree: np.mean(results_by_degree[degree]['test_maes']) 
                      for degree in ['linear', 'quadratic', 'cubic']}
    best_degree = min(mean_test_maes, key=mean_test_maes.get)

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"✅ BEST MODEL: {best_degree.upper()}")
    print(f"   Test MAE: {mean_test_maes[best_degree]:.4f} D")

    if best_degree != 'linear':
        improvement_over_linear = ((mean_test_maes['linear'] - mean_test_maes[best_degree]) / 
                                   mean_test_maes['linear']) * 100
        print(f"   Improvement over linear: {improvement_over_linear:.1f}%")
        print(f"\n   The polynomial terms capture non-linear relationships between")
        print(f"   corneal thickness and refractive error in Fuchs' dystrophy patients.")

    # Store best results for later use
    seed_test_maes_additive = results_by_degree[best_degree]['test_maes']
    seed_train_maes_additive = results_by_degree[best_degree]['train_maes']
    seed_improvements_additive = results_by_degree[best_degree]['improvements']
    seed_additive_params = results_by_degree[best_degree]['params']

    print(f"\n💾 Stored {best_degree} model results for combined approach.")
    
else:
    print("=" * 80)
    print("⏭️ POLYNOMIAL COMPARISON SKIPPED (RUN_POLYNOMIAL_COMPARISON = False)")
    print("=" * 80)
    print("Using direct quadratic approach in next cell instead.")
    print("To enable full comparison: Set RUN_POLYNOMIAL_COMPARISON = True")
    
    # Set best_degree for compatibility
    best_degree = 'quadratic'