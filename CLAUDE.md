# IOL Calculation Optimization for Fuchs' Dystrophy Patients

## Project Overview
Optimizing IOL power calculations for patients with Fuchs' dystrophy undergoing combined phacoemulsification and DMEK surgery. The edematous corneas in these patients cause standard formulas to fail.

## Current Results
- **Baseline SRK/T2:** MAE = 1.3591 D
- **Multiplicative Correction:** MAE = 0.9311 D (31.5% improvement)
- **Combined All Methods:** MAE = 0.8914 D (34.4% improvement)

## Key Findings
1. CCT (Central Corneal Thickness) features account for 75.5% of prediction importance
2. CCT_ratio (CCT/AL) is the most important feature
3. Combining parameter modification + multiplicative + additive corrections yields best results

## Ideas for Further Improvement

### 1. Advanced Feature Engineering
```python
# Add quadratic/interaction terms
CCT²
CCT³  
(CCT-600)² × K_avg
AL² / CCT
CCT × K_avg / AL
log(CCT/550)  # Log transform relative to normal
```
Ridge analysis showed CCT_ratio is critical - explore more complex interactions.

### 2. Patient Segmentation
```python
# Different formulas for subgroups
if CCT > 650:  # Severe edema
    use_severe_parameters
elif CCT < 580:  # Mild edema
    use_mild_parameters
else:  # Moderate
    use_standard_parameters
```
Consider separate optimization for different edema severity levels.

### 3. Outlier Analysis
- Identify 10-15 patients with worst prediction errors
- Analyze what they have in common
- Look for hidden patterns (extreme AL, abnormal K, very high/low IOL powers)
- Consider robust optimization that down-weights outliers

### 4. Non-Linear Parameter Dependencies
Instead of linear:
```python
nc = base + coef × CCT_norm
```
Try quadratic or exponential:
```python
nc = base + coef1 × CCT_norm + coef2 × CCT_norm²
# or
nc = base × exp(coef × CCT_norm)
```

### 5. Expand Parameter Bounds
Current bounds might be too restrictive:
```python
# Current
bounds = [(-0.5, 0.5)]

# Try wider
bounds = [(-1.0, 1.0)]
# or even
bounds = [(-2.0, 2.0)]
```
Let optimization explore wider parameter space.

### 6. Hybrid Machine Learning for Residuals
```python
# Use ML only to correct remaining errors
Residual = Actual - Combined_Formula_Prediction
Train RandomForest/XGBoost on residuals with all features
Final = Combined_Formula + ML_residual_correction
```
This maintains interpretability while capturing complex patterns.

### 7. Cross-Validation for Parameter Selection
Use nested CV to avoid overfitting:
- Outer loop: train/test split
- Inner loop: parameter optimization
- This gives honest performance estimates

### 8. Investigate Lens-Specific Effects
- Check if certain IOL models/A-constants behave differently
- Consider lens-specific corrections

### 9. Time-Based Analysis
If surgery dates available:
- Check if results improved over time (learning curve)
- Consider temporal validation

### 10. Bootstrap Confidence Intervals
- Use bootstrap resampling to get robust parameter estimates
- Report confidence intervals for all parameters

## Implementation Priority
1. **Outlier analysis** (quick to do, high insight)
2. **Non-linear parameters** (moderate effort, likely improvement)
3. **Feature engineering** (moderate effort, proven by Ridge)
4. **Patient segmentation** (if clear CCT clusters exist)
5. **Hybrid ML** (complex but potentially powerful)

## Formula for Publication
Current best (Combined):
```
1. Calculate CCT_norm = (CCT - 600) / 100
2. Calculate CCT_ratio = CCT / AL
3. Modify SRK/T2 parameters based on CCT
4. Apply multiplicative correction
5. Add final correction term
```

Target: Achieve >40% improvement (MAE < 0.8 D)