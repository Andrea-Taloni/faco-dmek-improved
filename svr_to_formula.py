"""
Convert SVR model to an approximate explicit formula
This allows clinical use without the SVR model
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import sympy as sp

class SVRFormulaExtractor:
    """
    Extracts an approximate explicit formula from trained SVR model
    """
    
    def __init__(self, svr_model, scaler, feature_names):
        self.svr_model = svr_model
        self.scaler = scaler
        self.feature_names = feature_names
        self.formula = None
        self.coefficients = None
        
    def fit_polynomial_approximation(self, X_train, degree=2):
        """
        Fit a polynomial that approximates the SVR predictions
        """
        # Get SVR predictions
        X_scaled = self.scaler.transform(X_train)
        svr_predictions = self.svr_model.predict(X_scaled)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X_train)
        
        # Fit linear regression to approximate SVR
        lr = LinearRegression()
        lr.fit(X_poly, svr_predictions)
        
        # Store coefficients
        self.coefficients = lr.coef_
        self.intercept = lr.intercept_
        self.poly = poly
        
        # Generate symbolic formula
        self._generate_formula(degree)
        
        # Calculate approximation quality
        poly_predictions = lr.predict(X_poly)
        mae = mean_absolute_error(svr_predictions, poly_predictions)
        r2 = lr.score(X_poly, svr_predictions)
        
        return {
            'mae': mae,
            'r2': r2,
            'formula': self.formula,
            'coefficients': self.coefficients
        }
    
    def _generate_formula(self, degree):
        """
        Generate symbolic formula from polynomial coefficients
        """
        # Create symbolic variables
        symbols = {}
        for name in self.feature_names[:5]:  # Limit to first 5 features
            if 'CCT' in name:
                symbols['CCT'] = sp.Symbol('CCT')
            elif 'AL' in name and 'CCT_AL' not in name:
                symbols['AL'] = sp.Symbol('AL')
            elif 'ACD' in name and 'ACD_AL' not in name:
                symbols['ACD'] = sp.Symbol('ACD')
            elif 'K_mean' in name:
                symbols['K'] = sp.Symbol('K')
            elif 'CCT_AL' in name:
                symbols['CCT_AL'] = sp.Symbol('CCT/AL')
        
        # Build formula
        feature_names = self.poly.get_feature_names_out(self.feature_names)
        formula_parts = [f"{self.intercept:.4f}"]
        
        for i, (coef, name) in enumerate(zip(self.coefficients, feature_names)):
            if abs(coef) > 0.001:  # Only include significant terms
                # Parse feature name and create term
                if 'CCT_norm^2' in name:
                    term = f"{coef:.4f}*((CCT-600)/100)^2"
                elif 'CCT_norm' in name and 'AL' in name:
                    term = f"{coef:.4f}*((CCT-600)/100)*AL"
                elif 'CCT_norm' in name:
                    term = f"{coef:.4f}*((CCT-600)/100)"
                elif 'AL^2' in name:
                    term = f"{coef:.4f}*AL^2"
                elif 'AL' in name and 'CCT_AL' not in name:
                    term = f"{coef:.4f}*AL"
                elif 'ACD' in name:
                    term = f"{coef:.4f}*ACD"
                elif 'K_mean' in name:
                    term = f"{coef:.4f}*K"
                elif 'CCT_AL' in name:
                    term = f"{coef:.4f}*(CCT/AL)"
                else:
                    continue
                
                if coef > 0 and i > 0:
                    term = "+" + term
                formula_parts.append(term)
        
        self.formula = " ".join(formula_parts)
        
    def create_clinical_formula(self, base_formula='SRKT2'):
        """
        Create the complete clinical formula
        """
        if self.formula:
            return f"{base_formula}_corrected = {base_formula}_base + ({self.formula})"
        return None


def extract_svr_formula(df, svr_model=None, scaler=None):
    """
    Main function to extract formula from SVR model
    """
    print("="*60)
    print("EXTRACTING EXPLICIT FORMULA FROM SVR MODEL")
    print("="*60)
    
    # If no model provided, train a simple one
    if svr_model is None:
        print("\nTraining SVR model first...")
        
        # Prepare features
        X = pd.DataFrame()
        X['CCT_norm'] = (df['CCT'] - 600) / 100
        X['AL'] = df['Bio-AL']
        X['ACD'] = df['Bio-ACD']
        X['K_mean'] = (df['Bio-Ks'] + df['Bio-Kf']) / 2
        X['CCT_AL'] = df['CCT'] / df['Bio-AL']
        
        y = df['PostOP Spherical Equivalent'].values
        
        # Train SVR
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        svr_model.fit(X_scaled, y)
        
        feature_names = X.columns.tolist()
    else:
        feature_names = ['CCT_norm', 'AL', 'ACD', 'K_mean', 'CCT_AL']
        X = pd.DataFrame()
        X['CCT_norm'] = (df['CCT'] - 600) / 100
        X['AL'] = df['Bio-AL']
        X['ACD'] = df['Bio-ACD']
        X['K_mean'] = (df['Bio-Ks'] + df['Bio-Kf']) / 2
        X['CCT_AL'] = df['CCT'] / df['Bio-AL']
    
    # Extract formula
    extractor = SVRFormulaExtractor(svr_model, scaler, feature_names)
    
    print("\n1. LINEAR APPROXIMATION:")
    result_linear = extractor.fit_polynomial_approximation(X, degree=1)
    print(f"   R² = {result_linear['r2']:.3f}")
    print(f"   Approximation MAE = {result_linear['mae']:.4f} D")
    print(f"   Formula: Correction = {extractor.formula}")
    
    print("\n2. QUADRATIC APPROXIMATION:")
    result_quad = extractor.fit_polynomial_approximation(X, degree=2)
    print(f"   R² = {result_quad['r2']:.3f}")
    print(f"   Approximation MAE = {result_quad['mae']:.4f} D")
    print(f"   Formula: Correction = {extractor.formula}")
    
    # Create clinical formulas
    print("\n3. CLINICAL FORMULAS:")
    print("-"*60)
    
    # Simplified linear formula
    print("\nSIMPLIFIED LINEAR (easiest to use):")
    coefs = result_linear['coefficients']
    intercept = extractor.intercept
    formula_simple = f"SRKT2_corrected = SRKT2_base + {intercept:.3f}"
    
    # Add only most important terms
    important_terms = []
    if abs(coefs[0]) > 0.01:  # CCT_norm
        important_terms.append(f"{coefs[0]:.3f}*((CCT-600)/100)")
    if len(coefs) > 1 and abs(coefs[1]) > 0.01:  # AL
        important_terms.append(f"{coefs[1]:.3f}*AL")
    if len(coefs) > 4 and abs(coefs[4]) > 0.01:  # CCT_AL
        important_terms.append(f"{coefs[4]:.3f}*(CCT/AL)")
    
    if important_terms:
        formula_simple += " + " + " + ".join(important_terms)
    
    print(f"   {formula_simple}")
    
    print("\nFULL QUADRATIC (most accurate):")
    print(f"   {extractor.create_clinical_formula('SRKT2')}")
    
    print("\n4. COMPARISON WITH MULTIPLICATIVE:")
    print("-"*60)
    print("Multiplicative: SRKT2 × (1 + m0 + m1×CCT_norm + m2×CCT/AL)")
    print("SVR Linear:     SRKT2 + (a0 + a1×CCT_norm + a2×AL + a3×CCT/AL)")
    print("\nKey difference: SVR uses ADDITION not MULTIPLICATION")
    print("This allows for more flexible corrections across the range")
    
    return extractor, formula_simple


if __name__ == "__main__":
    # Test with sample data
    df = pd.read_excel('FacoDMEK.xlsx', sheet_name='Data')
    
    extractor, formula = extract_svr_formula(df)
    
    print("\n" + "="*60)
    print("FORMULA READY FOR CLINICAL USE:")
    print(formula)
    print("="*60)