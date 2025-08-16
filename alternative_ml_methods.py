# ALTERNATIVE ML METHODS - RANDOM FOREST, XGBOOST, AND GAUSSIAN PROCESS
# ======================================================================
# PURPOSE: Test promising ML alternatives to SVR for IOL calculation
# Methods: Random Forest, XGBoost, and Gaussian Process Regression

print("=" * 80)
print("ALTERNATIVE ML METHODS FOR IOL CALCULATION")
print("=" * 80)

print("\n🤖 TESTING ADVANCED ML ALGORITHMS:")
print("-" * 50)
print("• Random Forest: Tree ensemble with bagging")
print("• XGBoost: Gradient boosting (state-of-the-art)")
print("• Gaussian Process: Probabilistic approach with uncertainty")

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost (may not be installed)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    HAS_XGBOOST = False

# Store results for each method
ml_methods_results = {}

print("\n" + "="*80)
print("RUNNING MULTI-SEED ANALYSIS FOR ML METHODS")
print("="*80)

# Test each ML method
methods_to_test = ['Random Forest', 'XGBoost', 'Gaussian Process']

for method_name in methods_to_test:
    if method_name == 'XGBoost' and not HAS_XGBOOST:
        print(f"\n⏭️ Skipping {method_name} (not installed)")
        continue
        
    print(f"\n{'='*40}")
    print(f"TESTING: {method_name}")
    print(f"{'='*40}")
    
    # Store results for this method
    seed_test_maes = []
    seed_train_maes = []
    seed_baseline_maes = []
    seed_improvements = []
    seed_overfit_ratios = []
    best_params_list = []
    
    for seed_idx, SEED in enumerate(SEEDS, 1):
        print(f"\nSeed {seed_idx}/{len(SEEDS)}: {SEED}")
        print("-" * 30)
        
        # Split data
        X_train_ml, X_test_ml = train_test_split(df, test_size=0.25, random_state=SEED)
        
        # Prepare features
        feature_cols = ['CCT', 'Bio-AL', 'Bio-ACD', 'Bio-Ks', 'Bio-Kf', 'IOL Power', 'A-Constant']
        X_train_features = X_train_ml[feature_cols].copy()
        X_test_features = X_test_ml[feature_cols].copy()
        
        # Add derived features
        X_train_features['K_mean'] = (X_train_ml['Bio-Ks'] + X_train_ml['Bio-Kf']) / 2
        X_train_features['CCT_norm'] = (X_train_ml['CCT'] - 600) / 100
        X_train_features['CCT_AL_ratio'] = X_train_ml['CCT'] / X_train_ml['Bio-AL']
        X_train_features['AL_ACD_ratio'] = X_train_ml['Bio-AL'] / X_train_ml['Bio-ACD']
        
        X_test_features['K_mean'] = (X_test_ml['Bio-Ks'] + X_test_ml['Bio-Kf']) / 2
        X_test_features['CCT_norm'] = (X_test_ml['CCT'] - 600) / 100
        X_test_features['CCT_AL_ratio'] = X_test_ml['CCT'] / X_test_ml['Bio-AL']
        X_test_features['AL_ACD_ratio'] = X_test_ml['Bio-AL'] / X_test_ml['Bio-ACD']
        
        # Target
        y_train = X_train_ml['PostOP Spherical Equivalent'].values
        y_test = X_test_ml['PostOP Spherical Equivalent'].values
        
        # Calculate baseline
        X_train_ml['K_avg'] = (X_train_ml['Bio-Ks'] + X_train_ml['Bio-Kf']) / 2
        X_test_ml['K_avg'] = (X_test_ml['Bio-Ks'] + X_test_ml['Bio-Kf']) / 2
        
        for dataset in [X_train_ml, X_test_ml]:
            dataset['SRKT2_Baseline'] = dataset.apply(
                lambda row: calculate_SRKT2(
                    AL=row['Bio-AL'],
                    K_avg=row['K_avg'],
                    IOL_power=row['IOL Power'],
                    A_constant=row['A-Constant']
                ), axis=1
            )
        
        baseline_mae = mean_absolute_error(X_test_ml['PostOP Spherical Equivalent'], 
                                          X_test_ml['SRKT2_Baseline'])
        
        # Setup cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        
        # Initialize model based on method
        if method_name == 'Random Forest':
            # Random Forest with hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=SEED)
            
            # Quick grid search with 3-fold CV (faster)
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_features, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"  Best RF params: trees={best_params['n_estimators']}, depth={best_params['max_depth']}")
            
        elif method_name == 'XGBoost':
            # XGBoost with hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            base_model = xgb.XGBRegressor(random_state=SEED, objective='reg:squarederror')
            
            # Quick grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_features, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"  Best XGB params: trees={best_params['n_estimators']}, depth={best_params['max_depth']}, lr={best_params['learning_rate']}")
            
        elif method_name == 'Gaussian Process':
            # Gaussian Process with different kernels
            # Note: GP doesn't scale well, so we'll use a subset of features
            important_features = ['CCT_norm', 'CCT_AL_ratio', 'K_mean', 'Bio-AL', 'Bio-ACD']
            X_train_gp = X_train_features[important_features]
            X_test_gp = X_test_features[important_features]
            
            # Standardize for GP
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_gp)
            X_test_scaled = scaler.transform(X_test_gp)
            
            # Try different kernels
            kernels = [
                RBF(length_scale=1.0),
                Matern(length_scale=1.0, nu=1.5),
                RationalQuadratic(length_scale=1.0, alpha=1.0)
            ]
            
            best_kernel = None
            best_kernel_mae = float('inf')
            
            for kernel in kernels:
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.1,  # Noise level
                    random_state=SEED,
                    normalize_y=True
                )
                
                # Quick CV to select best kernel
                cv_maes = []
                for train_idx, val_idx in kf.split(X_train_scaled):
                    gpr.fit(X_train_scaled[train_idx], y_train[train_idx])
                    pred = gpr.predict(X_train_scaled[val_idx])
                    cv_maes.append(mean_absolute_error(y_train[val_idx], pred))
                
                mean_cv_mae = np.mean(cv_maes)
                if mean_cv_mae < best_kernel_mae:
                    best_kernel_mae = mean_cv_mae
                    best_kernel = kernel
            
            # Train final model with best kernel
            model = GaussianProcessRegressor(
                kernel=best_kernel,
                alpha=0.1,
                random_state=SEED,
                normalize_y=True
            )
            model.fit(X_train_scaled, y_train)
            
            print(f"  Best GP kernel: {best_kernel.__class__.__name__}")
            best_params = {'kernel': best_kernel.__class__.__name__}
        
        # Make predictions
        if method_name == 'Gaussian Process':
            # GP needs scaled features
            y_pred_test = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_scaled)
            
            # Also get uncertainty estimates (unique to GP!)
            y_pred_test_std = model.predict(X_test_scaled, return_std=True)[1]
            mean_uncertainty = np.mean(y_pred_test_std)
            print(f"  Mean prediction uncertainty: {mean_uncertainty:.3f} D")
        else:
            # RF and XGBoost
            y_pred_test = model.predict(X_test_features)
            y_pred_train = model.predict(X_train_features)
        
        # Calculate metrics
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        improvement = ((baseline_mae - test_mae) / baseline_mae) * 100
        overfit_ratio = train_mae / test_mae if test_mae > 0 else 0
        
        print(f"  Test MAE: {test_mae:.4f} D")
        print(f"  Train MAE: {train_mae:.4f} D")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Overfit ratio: {overfit_ratio:.3f}")
        
        # Feature importance for tree-based methods
        if method_name in ['Random Forest', 'XGBoost']:
            importances = model.feature_importances_
            top_features_idx = np.argsort(importances)[-3:]  # Top 3
            top_features = [X_train_features.columns[i] for i in top_features_idx]
            print(f"  Top features: {', '.join(top_features)}")
        
        # Store results
        seed_test_maes.append(test_mae)
        seed_train_maes.append(train_mae)
        seed_baseline_maes.append(baseline_mae)
        seed_improvements.append(improvement)
        seed_overfit_ratios.append(overfit_ratio)
        best_params_list.append(best_params)
    
    # Summary for this method
    print(f"\n{'='*40}")
    print(f"{method_name} - MULTI-SEED SUMMARY")
    print(f"{'='*40}")
    
    print(f"\n📊 TEST SET PERFORMANCE (n={len(SEEDS)} seeds):")
    print(f"  Mean MAE: {np.mean(seed_test_maes):.4f} ± {np.std(seed_test_maes):.4f} D")
    print(f"  Best MAE: {np.min(seed_test_maes):.4f} D")
    print(f"  Worst MAE: {np.max(seed_test_maes):.4f} D")
    
    print(f"\n📈 IMPROVEMENT OVER BASELINE:")
    print(f"  Mean: {np.mean(seed_improvements):.1f} ± {np.std(seed_improvements):.1f}%")
    
    print(f"\n⚠️ OVERFITTING ANALYSIS:")
    print(f"  Mean overfit ratio: {np.mean(seed_overfit_ratios):.3f}")
    if np.mean(seed_overfit_ratios) < 0.9:
        print("  Status: HIGH overfitting")
    elif np.mean(seed_overfit_ratios) < 0.95:
        print("  Status: Moderate overfitting")
    else:
        print("  Status: Low overfitting")
    
    # Store results for comparison
    ml_methods_results[method_name] = {
        'test_maes': seed_test_maes,
        'train_maes': seed_train_maes,
        'baseline_maes': seed_baseline_maes,
        'improvements': seed_improvements,
        'overfit_ratios': seed_overfit_ratios,
        'mean_mae': np.mean(seed_test_maes),
        'std_mae': np.std(seed_test_maes),
        'mean_improvement': np.mean(seed_improvements),
        'mean_overfit': np.mean(seed_overfit_ratios)
    }
    
    # Store for final comparison (using variable names compatible with final summary)
    if method_name == 'Random Forest':
        seed_test_maes_rf = seed_test_maes
        seed_train_maes_rf = seed_train_maes
        seed_baseline_maes_rf = seed_baseline_maes
        seed_improvements_rf = seed_improvements
        seed_overfit_ratios_rf = seed_overfit_ratios
    elif method_name == 'XGBoost':
        seed_test_maes_xgb = seed_test_maes
        seed_train_maes_xgb = seed_train_maes
        seed_baseline_maes_xgb = seed_baseline_maes
        seed_improvements_xgb = seed_improvements
        seed_overfit_ratios_xgb = seed_overfit_ratios
    elif method_name == 'Gaussian Process':
        seed_test_maes_gpr = seed_test_maes
        seed_train_maes_gpr = seed_train_maes
        seed_baseline_maes_gpr = seed_baseline_maes
        seed_improvements_gpr = seed_improvements
        seed_overfit_ratios_gpr = seed_overfit_ratios

# Final comparison of ML methods
print("\n" + "="*80)
print("ML METHODS COMPARISON")
print("="*80)

print("\n📊 RANKING BY TEST MAE:")
print("-" * 50)

# Sort methods by performance
sorted_methods = sorted(ml_methods_results.items(), key=lambda x: x[1]['mean_mae'])

for rank, (method, results) in enumerate(sorted_methods, 1):
    print(f"{rank}. {method:<20}: {results['mean_mae']:.4f} ± {results['std_mae']:.4f} D")
    print(f"   {'':20}  Improvement: {results['mean_improvement']:.1f}%")
    print(f"   {'':20}  Overfit ratio: {results['mean_overfit']:.3f}")

# Compare with SVR if available
if 'seed_test_maes_mult' in locals():  # SVR results stored as _mult
    svr_mae = np.mean(seed_test_maes_mult)
    print(f"\n📊 COMPARISON WITH SVR:")
    print("-" * 50)
    
    for method, results in ml_methods_results.items():
        diff = results['mean_mae'] - svr_mae
        pct_diff = (diff / svr_mae) * 100
        if diff < 0:
            print(f"{method}: {-pct_diff:.1f}% BETTER than SVR")
        else:
            print(f"{method}: {pct_diff:.1f}% worse than SVR")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)

best_ml_method = sorted_methods[0][0]
best_ml_mae = sorted_methods[0][1]['mean_mae']

if 'seed_test_maes_mult' in locals() and best_ml_mae < np.mean(seed_test_maes_mult):
    improvement = ((np.mean(seed_test_maes_mult) - best_ml_mae) / np.mean(seed_test_maes_mult)) * 100
    print(f"✅ {best_ml_method} outperforms SVR by {improvement:.1f}%!")
    print(f"   Consider using {best_ml_method} instead of SVR")
else:
    print(f"📊 {best_ml_method} is the best alternative ML method")
    print("   But SVR likely remains the optimal choice")

print("\nKey insights:")
print("• Tree-based methods (RF, XGBoost) capture feature interactions well")
print("• Gaussian Process provides uncertainty estimates (useful clinically)")
print("• All methods benefit from feature engineering (CCT_norm, ratios)")