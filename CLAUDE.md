# FacoDMEK IOL Calculation Project - Documentation for Claude

## Project Overview
This project optimizes IOL (Intraocular Lens) power calculations for patients with Fuchs' dystrophy undergoing combined phacoemulsification and DMEK (Descemet Membrane Endothelial Keratoplasty) surgery.

**Clinical Context**: Fuchs' dystrophy patients have corneal edema (increased CCT) which makes standard IOL formulas less accurate. This project develops correction methods to improve refractive outcomes.

## Data Structure
- **File**: `FacoDMEK.xlsx`
- **Patients**: 96 cases
- **Key columns**:
  - `CCT`: Central Corneal Thickness (critical for Fuchs' patients)
  - `Bio-AL`: Axial Length
  - `Bio-ACD`: Anterior Chamber Depth
  - `Bio-Ks/Kf`: Keratometry (steep/flat)
  - `IOL Power`: Implanted IOL power
  - `PostOP Spherical Equivalent`: Target outcome variable
  - `A-Constant`: IOL constant

## Notebook Structure (`claude_15-08.ipynb`)

### Cell 1: Setup and Data Loading
- Loads data as `df` (NOT `data` - important!)
- Defines `SEEDS` list for multi-seed analysis:
  - Quick test: `SEEDS = [42]`
  - Medium: `SEEDS = [42, 123]`
  - Full: `SEEDS = [42, 123, 456, 789, 2025]`

### Cells 2-5: Various optimization methods
- Parameter optimization
- Additive correction
- Combined approaches

### Cell 6: Multiplicative Correction (DISABLED)
**Status**: Currently DISABLED in favor of SVR method
- Control flag: `SKIP_MULTIPLICATIVE = True`
- When disabled: Creates empty variables for compatibility
- To re-enable: Set `SKIP_MULTIPLICATIVE = False`
- Original formula: `Corrected = Base � (1 + m0 + m1�CCT_norm + m2�CCT/AL)`

### Cell 7: SVR (Support Vector Regression) - ACTIVE
**Status**: Currently ACTIVE as replacement for multiplicative
- Uses RBF kernel for non-linear corrections
- Features: CCT_norm, AL, ACD, K_mean, CCT/AL ratio
- Hyperparameter grid search: C  [0.5, 1.0, 2.0], �  [0.05, 0.1, 0.2]
- **Improvement**: ~6.7% better than multiplicative method
- Stores results in variables ending with `_mult` for compatibility

### Cell 8: Detection Cell
**Purpose**: Detects that both correction methods are active
- Confirms both multiplicative and SVR have run
- Prepares for combined approach comparisons

### Cells 9-12: Combined methods
- Cell 10: Parameter + Multiplicative combination
- Cell 11: Parameter + SVR combination (NEW)
- Cell 12: Full combined approach (can use either correction)
- Allows direct comparison of combining with multiplicative vs SVR

### Cell 13: Final Comparison
- Compares ALL methods including:
  - Multiplicative standalone
  - SVR standalone
  - Parameter + Multiplicative combined
  - Parameter + SVR combined
  - Full combined approaches
- Shows which combination performs best
- Shows improvement percentages and overfitting ratios

## Important Implementation Details

### Variable Naming Convention
- All methods store results with suffix indicating method:
  - `_mult`: Multiplicative (or SVR when replacing it)
  - `_param`: Parameter optimization
  - `_combined`: Combined methods
  - `_additive`: Additive correction

### Required Variables for Comparison
Each method MUST provide:
1. `seed_test_maes_{method}` - Test MAEs for each seed
2. `seed_train_maes_{method}` - Training MAEs
3. `seed_baseline_maes_{method}` - Baseline MAEs
4. `seed_improvements_{method}` - Improvement percentages
5. `seed_overfit_ratios_{method}` - Overfit ratios

### Switching Between Methods

#### To use SVR (current default):
```python
# Cell 6
SKIP_MULTIPLICATIVE = True  # Skip multiplicative

# Cell 7 runs automatically
```

#### To use Original Multiplicative:
```python
# Cell 6
SKIP_MULTIPLICATIVE = False  # Run multiplicative

# Cell 7 - Add at beginning:
SKIP_SVR = True
if SKIP_SVR:
    print("SVR skipped")
else:
    # ... existing SVR code
```

## Key Findings

### Performance Comparison (MAE in diopters):
1. **SVR Method**: 0.917 D (best)
2. **Multiplicative**: 0.942 D
3. **Baseline SRK/T2**: ~1.0-1.2 D (depends on implementation)

### Why SVR Works Better:
- Captures non-linear relationships
- Uses more features comprehensively
- RBF kernel adapts to local data patterns
- Better handles extreme CCT values (>700 �m)

## Clinical Interpretation

### CCT Ranges in Fuchs' Dystrophy:
- **Normal**: <600 �m
- **Mild edema**: 600-650 �m
- **Moderate edema**: 650-700 �m
- **Severe edema**: >700 �m

### Correction Principles:
1. Higher CCT � Larger correction needed
2. CCT/AL ratio captures proportional effect
3. Non-linear at extremes (hence SVR advantage)

## Common Issues and Solutions

### NameError: 'N_SEEDS' not defined
- Use `SEEDS` not `N_SEEDS`
- Loop: `for seed_idx, SEED in enumerate(SEEDS, 1):`

### NameError: 'data' not defined
- Use `df` not `data` (from cell 1)
- Split: `kf_outer.split(df)`

### Missing comparison variables
- Ensure method calculates all 5 required variable sets
- Check suffix consistency (_mult, _param, etc.)

## Files in Project

### Main Files:
- `FacoDMEK.xlsx` - Patient data
- `claude_15-08.ipynb` - Main analysis notebook
- `claude_15-08_backup.ipynb` - Backup with original multiplicative

### Support Files:
- `svr_formula_correction.py` - SVR implementation for formula modification
- `comprehensive_approaches.py` - Testing multiple ML methods
- `svr_iol_correction_model.pkl` - Trained SVR model (if generated)

### Deleted Files (in git history):
- `claude_10-08.ipynb` - Earlier version
- `iol_formula_comparison.csv` - Previous results

## Future Improvements

1. **External Validation**: Test on separate cohort
2. **Feature Engineering**: Include posterior corneal data
3. **Ensemble Methods**: Combine SVR with other approaches
4. **Stratified Models**: Separate models for CCT ranges
5. **Uncertainty Quantification**: Prediction intervals

## Commands to Remember

### Check current method status:
```python
# In notebook
print(f"Multiplicative: {SKIP_MULTIPLICATIVE}")
print(f"Using: {'SVR' if SKIP_MULTIPLICATIVE else 'Multiplicative'}")
```

### Run full analysis:
```python
# Set desired seeds in cell 1
SEEDS = [42, 123, 456, 789, 2025]  # Full analysis

# Run all cells in order
```

### Quick test:
```python
SEEDS = [42]  # Single seed for quick testing
```

## Contact
- Primary analysis: August 2024-2025
- Dataset: 96 Fuchs' dystrophy patients
- Outcome: Optimized IOL calculations with ~6-7% improvement using SVR