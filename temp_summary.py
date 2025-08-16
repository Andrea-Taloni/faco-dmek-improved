# MULTI-SEED COMPARISON - FINAL COMPREHENSIVE SUMMARY
# ====================================================
# PURPOSE: Compare all methods across multiple seeds for robust conclusions

print("=" * 80)
print("MULTI-SEED ANALYSIS - COMPREHENSIVE COMPARISON")
print("=" * 80)

print(f"\n🔬 VALIDATION SETUP:")
print("-" * 50)
print(f"• Seeds tested: {SEEDS}")
print(f"• Train/Test split: 75%/25% (72/24 patients)")
print(f"• Inner validation: 5-fold CV")
print(f"• All results on holdout test sets")

# Check which methods have been run
available_methods = []
for method in ['parameter', 'multiplicative', 'additive', 'combined', 'fixed_combined']:
    if method in multi_seed_results and multi_seed_results[method]:
        available_methods.append(method)

if not available_methods:
    print("\n⚠️ No multi-seed results found yet!")
    print("Please run the optimization cells first.")
else:
    print(f"\n✅ Methods analyzed: {', '.join(available_methods)}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON ACROSS METHODS AND SEEDS")
    print("="*80)
    
    # Detailed table by seed
    print("\n📊 DETAILED RESULTS BY SEED:")
    print("-" * 80)
    print(f"{'Method':<20} | ", end="")
    for seed in SEEDS:
        print(f"Seed {seed:3} | ", end="")
    print(f"{'Mean ± Std':<15} | {'Best':<6} | {'Worst':<6}")
    print("-" * 80)
    
    for method in available_methods:
        results = multi_seed_results[method]
        print(f"{method.capitalize():<20} | ", end="")
        for mae in results['test_maes']:
            print(f"{mae:7.4f} | ", end="")
        mean_mae = results['mean_mae']
        std_mae = results['std_mae']
        print(f"{mean_mae:.4f} ± {std_mae:.4f} | ", end="")
        print(f"{min(results['test_maes']):.4f} | {max(results['test_maes']):.4f}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    print("\n📈 MEAN PERFORMANCE (averaged across seeds):")
    print("-" * 50)
    
    # Sort methods by mean MAE
    sorted_methods = sorted(available_methods, 
                          key=lambda m: multi_seed_results[m]['mean_mae'])
    
    for rank, method in enumerate(sorted_methods, 1):
        results = multi_seed_results[method]
        mean_mae = results['mean_mae']
        std_mae = results['std_mae']
        mean_imp = results['mean_improvement']
        
        print(f"{rank}. {method.capitalize():<20}: MAE = {mean_mae:.4f} ± {std_mae:.4f} D")
        print(f"   {'':20}  Improvement = {mean_imp:.1f}%")
    
    # Best overall method
    best_method = sorted_methods[0]
    best_results = multi_seed_results[best_method]
    
    print(f"\n🏆 BEST METHOD: {best_method.upper()}")
    print(f"   Mean MAE: {best_results['mean_mae']:.4f} ± {best_results['std_mae']:.4f} D")
    print(f"   Mean Improvement: {best_results['mean_improvement']:.1f}%")
    
    # Robustness analysis
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS")
    print("="*80)
    
    print("\n📊 STABILITY ACROSS SEEDS (Coefficient of Variation):")
    print("-" * 50)
    
    stability_scores = []
    for method in available_methods:
        results = multi_seed_results[method]
        cv = (results['std_mae'] / results['mean_mae']) * 100
        stability_scores.append((method, cv))
    
    # Sort by stability (lower CV is better)
    stability_scores.sort(key=lambda x: x[1])
    
    for method, cv in stability_scores:
        if cv < 5:
            status = "✅ Excellent"
        elif cv < 10:
            status = "✅ Good"
        elif cv < 15:
            status = "⚠️ Moderate"
        else:
            status = "⚠️ Variable"
        print(f"  {method.capitalize():<20}: CV = {cv:5.1f}%  {status}")
    
    # Range analysis
    print("\n📊 PERFORMANCE RANGE (max - min across seeds):")
    print("-" * 50)
    
    for method in available_methods:
        results = multi_seed_results[method]
        mae_range = max(results['test_maes']) - min(results['test_maes'])
        print(f"  {method.capitalize():<20}: {mae_range:.4f} D")
    
    # Statistical significance insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("\n💡 STATISTICAL CONCLUSIONS:")
    print("-" * 50)
    
    # Check if best method is consistently best
    best_count = 0
    for i in range(len(SEEDS)):
        seed_maes = {m: multi_seed_results[m]['test_maes'][i] for m in available_methods}
        if min(seed_maes, key=seed_maes.get) == best_method:
            best_count += 1
    
    consistency = (best_count / len(SEEDS)) * 100
    print(f"• {best_method.capitalize()} was best in {best_count}/{len(SEEDS)} seeds ({consistency:.0f}%)")
    
    # Check overlap in confidence intervals
    if len(available_methods) > 1:
        print("\n• Confidence intervals (mean ± std):")
        for method in sorted_methods[:3]:  # Top 3 methods
            results = multi_seed_results[method]
            lower = results['mean_mae'] - results['std_mae']
            upper = results['mean_mae'] + results['std_mae']
            print(f"  {method.capitalize():<20}: [{lower:.4f}, {upper:.4f}] D")
    
    # Clinical relevance
    print("\n📏 CLINICAL RELEVANCE:")
    print("-" * 50)
    
    baseline_mean = np.mean(multi_seed_results[available_methods[0]]['baseline_maes'])
    for method in sorted_methods:
        results = multi_seed_results[method]
        mean_mae = results['mean_mae']
        
        if mean_mae < 0.5:
            clinical = "Excellent (< 0.5 D)"
        elif mean_mae < 0.75:
            clinical = "Good (< 0.75 D)"
        elif mean_mae < 1.0:
            clinical = "Acceptable (< 1.0 D)"
        else:
            clinical = "Poor (≥ 1.0 D)"
        
        print(f"  {method.capitalize():<20}: {clinical}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n✅ FINAL RECOMMENDATIONS:")
    print("-" * 50)
    
    # Find most stable method
    most_stable = min(stability_scores, key=lambda x: x[1])
    
    print(f"1. Best performance: {best_method.capitalize()} (MAE = {best_results['mean_mae']:.4f} D)")
    print(f"2. Most stable: {most_stable[0].capitalize()} (CV = {most_stable[1]:.1f}%)")
    
    if best_method == most_stable[0]:
        print(f"\n🎯 {best_method.capitalize()} is both best performing AND most stable!")
        print("   This is the recommended approach for clinical use.")
    else:
        print(f"\n⚖️ Trade-off detected:")
        print(f"   • {best_method.capitalize()}: Better performance but less stable")
        print(f"   • {most_stable[0].capitalize()}: More stable but slightly worse performance")
        print("   Choose based on clinical priorities.")
    
    print("\n📝 PUBLICATION-READY SUMMARY:")
    print("-" * 50)
    print(f"Using {len(SEEDS)} different random seeds for validation,")
    print(f"{best_method.capitalize()} achieved the best mean MAE of {best_results['mean_mae']:.3f} ± {best_results['std_mae']:.3f} D,")
    print(f"representing a {best_results['mean_improvement']:.1f}% improvement over baseline SRK/T2.")
    
print("\n" + "="*80)
print("END OF MULTI-SEED ANALYSIS")
print("="*80)