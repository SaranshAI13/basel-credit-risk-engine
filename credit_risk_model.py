import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import warnings
warnings.filterwarnings("ignore")

class WholesaleCreditRiskModel:
    def __init__(self, model_type="Random Forest"):
        self.model_type = model_type
        self.corp_scaler = StandardScaler()
        self.sov_scaler = StandardScaler()
        
        if model_type == "Logistic Regression":
            self.corp_clf = LogisticRegression(random_state=42, max_iter=1000)
            self.sov_clf = LogisticRegression(random_state=42, max_iter=1000)
        else: # Random Forest
            self.corp_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            self.sov_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            
        self.corp_features = ['debt_to_equity', 'interest_coverage', 'ebitda_margin', 'rating_ordinal']
        self.sov_features = ['yield_spread_bps', 'debt_to_gdp', 'rating_ordinal']
        
        self.is_trained = False
        self.corp_metrics = {}
        self.sov_metrics = {}

    def train(self, df):
        from joblib import parallel_backend
        
        # 1. Split into Corporate and Sovereign subsets
        corp_df = df[df['entity_type'] == 'Corporate'].copy()
        sov_df = df[df['entity_type'] == 'Sovereign'].copy()
        
        with parallel_backend('sequential'):
            # 2. Train Corporate PD Model
            X_corp = corp_df[self.corp_features]
            y_corp = corp_df['historic_default']
            
            X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
                X_corp, y_corp, test_size=0.2, random_state=42, stratify=y_corp
            )
            
            X_c_train_scaled = self.corp_scaler.fit_transform(X_c_train)
            X_c_test_scaled = self.corp_scaler.transform(X_c_test)
            
            self.corp_clf.fit(X_c_train_scaled, y_c_train)
            
            # Evaluate Corporate Model
            y_c_pred = self.corp_clf.predict(X_c_test_scaled)
            y_c_prob = self.corp_clf.predict_proba(X_c_test_scaled)[:, 1]
            
            # ROC AUC
            c_auc = roc_auc_score(y_c_test, y_c_prob)
            c_report = classification_report(y_c_test, y_c_pred, output_dict=True)
            
            self.corp_metrics = {
                "auc": c_auc,
                "report": c_report,
                "test_y": y_c_test.values,
                "test_prob": y_c_prob
            }
            
            # 3. Train Sovereign PD Model
            X_sov = sov_df[self.sov_features]
            y_sov = sov_df['historic_default']
            
            X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
                X_sov, y_sov, test_size=0.2, random_state=42, stratify=y_sov
            )
            
            X_s_train_scaled = self.sov_scaler.fit_transform(X_s_train)
            X_s_test_scaled = self.sov_scaler.transform(X_s_test)
            
            self.sov_clf.fit(X_s_train_scaled, y_s_train)
            
            # Evaluate Sovereign Model
            y_s_pred = self.sov_clf.predict(X_s_test_scaled)
            y_s_prob = self.sov_clf.predict_proba(X_s_test_scaled)[:, 1]
            
            s_auc = roc_auc_score(y_s_test, y_s_prob)
            s_report = classification_report(y_s_test, y_s_pred, output_dict=True)
            
            self.sov_metrics = {
                "auc": s_auc,
                "report": s_report,
                "test_y": y_s_test.values,
                "test_prob": y_s_prob
            }
        
        self.is_trained = True
        return self

    def predict_pd(self, df):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        from joblib import parallel_backend
        df_out = df.copy()
        pds = np.zeros(len(df))
        
        with parallel_backend('sequential'):
            # Corporate predictions (vectorized)
            corp_mask = df['entity_type'] == 'Corporate'
            if corp_mask.any():
                X_corp = df.loc[corp_mask, self.corp_features]
                X_corp_scaled = self.corp_scaler.transform(X_corp)
                pds[corp_mask] = self.corp_clf.predict_proba(X_corp_scaled)[:, 1]
                
            # Sovereign predictions (vectorized)
            sov_mask = df['entity_type'] == 'Sovereign'
            if sov_mask.any():
                X_sov = df.loc[sov_mask, self.sov_features]
                X_sov_scaled = self.sov_scaler.transform(X_sov)
                pds[sov_mask] = self.sov_clf.predict_proba(X_sov_scaled)[:, 1]
                
        df_out['predicted_pd'] = pds
        return df_out

    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        importances = {}
        if self.model_type == "Random Forest":
            importances['Corporate'] = dict(zip(self.corp_features, self.corp_clf.feature_importances_))
            importances['Sovereign'] = dict(zip(self.sov_features, self.sov_clf.feature_importances_))
        else: # Logistic Regression
            corp_coef = np.abs(self.corp_clf.coef_[0])
            corp_coef_norm = corp_coef / np.sum(corp_coef)
            importances['Corporate'] = dict(zip(self.corp_features, corp_coef_norm))
            
            sov_coef = np.abs(self.sov_clf.coef_[0])
            sov_coef_norm = sov_coef / np.sum(sov_coef)
            importances['Sovereign'] = dict(zip(self.sov_features, sov_coef_norm))
            
        return importances


# --- Basel III IRB Calculations ---

def basel_correlation_r(pd_val, entity_type, annual_sales_m=None):
    """
    Computes Basel III asset correlation (R) for corporate and sovereign exposures.
    Includes corporate SME size adjustment if sales are provided and < 50 million.
    """
    # PD floor at 0.03% (0.0003) and capped at 99.9%
    pd_bounded = np.clip(pd_val, 0.0003, 0.999)
    
    # Standard Basel corporate and sovereign asset correlation formula
    exponent = -50.0 * pd_bounded
    factor = (1.0 - np.exp(exponent)) / (1.0 - np.exp(-50.0))
    R = 0.12 * factor + 0.24 * (1.0 - factor)
    
    # SME Size Adjustment
    if entity_type == 'Corporate' and annual_sales_m is not None:
        sales = np.clip(annual_sales_m, 5.0, 50.0) # size correlation adjustment applies between €5M and €50M
        R_adj = R - 0.04 * (1.0 - (sales - 5.0) / 45.0)
        return R_adj
        
    return R

def maturity_adjustment_b(pd_val):
    """
    Computes maturity adjustment factor b(PD) under Basel III.
    """
    pd_bounded = np.clip(pd_val, 0.0003, 0.999)
    return (0.11852 - 0.05478 * np.log(pd_bounded)) ** 2

def basel_capital_requirement_k(pd_val, lgd_val, R, maturity):
    """
    Computes K (regulatory capital requirement percentage) under Basel III IRB approach.
    """
    pd_bounded = np.clip(pd_val, 0.0003, 0.999)
    lgd_bounded = np.clip(lgd_val, 0.0, 1.0)
    
    # Inverse normal CDF of PD and 99.9% confidence level
    g_pd = norm.ppf(pd_bounded)
    g_999 = norm.ppf(0.999)
    
    # Capital requirement calculation (excluding maturity adjustment)
    numerator = g_pd + np.sqrt(R) * g_999
    denominator = np.sqrt(1.0 - R)
    norm_val = norm.cdf(numerator / denominator)
    
    # Maturity adjustment
    b = maturity_adjustment_b(pd_val)
    # Standard maturity is M, default F-IRB uses 2.5 years
    M = np.clip(maturity, 1.0, 10.0)
    maturity_factor = (1.0 + (M - 2.5) * b) / (1.0 - 1.5 * b)
    
    # Standard Basel IRB capital requirement formula
    K = (lgd_bounded * norm_val - pd_bounded * lgd_bounded) * maturity_factor
    return max(0.0, K)


# --- HQLA (Liquidity Risk) & IRRBB Calculations ---

def classify_hqla(entity_type, rating, seniority):
    """
    Categorizes exposure under Basel III High-Quality Liquid Assets (HQLA):
    - Level 1 HQLA (0% Haircut): AAA to AA- Sovereigns
    - Level 2A HQLA (15% Haircut): A+ to A- Sovereigns, AAA to A- Corporates (must be Senior)
    - Level 2B HQLA (50% Haircut): BBB+ to BBB- Corporates (must be Senior)
    - Non-HQLA (100% Haircut): Subordinated debt, or ratings BB+ and below.
    """
    if seniority == 'Subordinated':
        return 'Non-HQLA', 1.0
        
    rating_high = ['AAA', 'AA']
    rating_med = ['A']
    rating_low = ['BBB']
    
    # Standardize rating string representation
    r_clean = rating.replace('/C', '').strip()
    
    if entity_type == 'Sovereign':
        if r_clean in ['AAA', 'AA']:
            return 'Level 1 HQLA', 0.0
        elif r_clean in ['A']:
            return 'Level 2A HQLA', 0.15
    elif entity_type == 'Corporate':
        if r_clean in ['AAA', 'AA', 'A']:
            return 'Level 2A HQLA', 0.15
        elif r_clean in ['BBB']:
            return 'Level 2B HQLA', 0.50
            
    return 'Non-HQLA', 1.0


def calculate_portfolio_metrics(df):
    """
    Applies Credit, Liquidity, and Interest Rate calculations to the loan book dataframe.
    """
    df_calc = df.copy()
    
    # 1. Precalculate PD (already predicted) and LGD
    df_calc['pd'] = np.clip(df_calc['predicted_pd'], 0.0003, 0.999)
    df_calc['lgd'] = 1.0 - df_calc['recovery_rate']
    
    # 2. Basel IRB calculations
    df_calc['correlation_R'] = df_calc.apply(
        lambda row: basel_correlation_r(
            row['pd'], row['entity_type'], row['annual_sales_m']
        ), axis=1
    )
    
    df_calc['capital_requirement_K'] = df_calc.apply(
        lambda row: basel_capital_requirement_k(
            row['pd'], row['lgd'], row['correlation_R'], row['remaining_maturity']
        ), axis=1
    )
    
    df_calc['rwa_m'] = 12.5 * df_calc['capital_requirement_K'] * df_calc['ead_m']
    df_calc['expected_loss_m'] = df_calc['pd'] * df_calc['lgd'] * df_calc['ead_m']
    
    # Capital buffers
    df_calc['min_capital_buffer_m'] = df_calc['capital_requirement_K'] * df_calc['ead_m'] # 8% RWA
    df_calc['cet1_4_5_m'] = df_calc['rwa_m'] * 0.045
    df_calc['cet1_7_0_m'] = df_calc['rwa_m'] * 0.070 # 4.5% min + 2.5% CCB
    
    # 3. HQLA calculations
    hqla_info = df_calc.apply(
        lambda row: classify_hqla(
            row['entity_type'], row['credit_rating'], row['seniority']
        ), axis=1
    )
    df_calc['hqla_class'] = [h[0] for h in hqla_info]
    df_calc['hqla_haircut'] = [h[1] for h in hqla_info]
    df_calc['eligible_hqla_m'] = df_calc['ead_m'] * (1.0 - df_calc['hqla_haircut'])
    
    # 4. Interest Rate Risk (IRRBB) calculations
    # Modified Duration = Remaining Maturity / (1 + Yield)
    df_calc['duration'] = df_calc['remaining_maturity'] / (1.0 + df_calc['yield_to_maturity'])
    df_calc['dv01_k'] = df_calc['duration'] * df_calc['ead_m'] * 0.1 # In thousands: Duration * EAD * 1000 * 0.0001 = Duration * EAD * 0.1
    
    return df_calc


# --- Scenario Stress Testing ---

def apply_stress_scenario(df, scenario_name):
    """
    Shocks portfolio financial ratios and yields according to macroeconomic stress scenarios:
    - Baseline: Normal conditions.
    - Global Liquidity Squeeze: Corporate debt-to-equity increases by 30%, interest coverage falls by 30%,
      sovereign spreads widen by 300 bps, and all interest rate yields shift up by +200 bps.
    - Sovereign Debt Crisis: Sovereign spreads widen by 500 bps, sovereign debt-to-gdp increases by 20%,
      sovereign recovery rates decrease by 20%, and sovereign yields spike by +400 bps (+100 bps for corporates).
      High-debt sovereigns (>80% debt-to-gdp) default probability is scaled by 2.0.
    """
    df_stressed = df.copy()
    
    if scenario_name == "Baseline":
        return df_stressed
        
    elif scenario_name == "Global Liquidity Squeeze":
        # Corporate stress
        df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'debt_to_equity'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'debt_to_equity'] * 1.30, 0.1, 5.0
        )
        df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'interest_coverage'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'interest_coverage'] * 0.70, -2.0, 15.0
        )
        
        # Sovereign stress
        df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_spread_bps'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_spread_bps'] + 300.0, 10.0, 1200.0
        )
        
        # Yield Curve Parallel shift (+200 bps)
        df_stressed['yield_to_maturity'] = np.clip(df_stressed['yield_to_maturity'] + 0.02, 0.0, 0.20)
        
    elif scenario_name == "Sovereign Debt Crisis":
        # Sovereign stress
        df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_spread_bps'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_spread_bps'] + 500.0, 10.0, 1200.0
        )
        df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'debt_to_gdp'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'debt_to_gdp'] * 1.20, 0.20, 1.50
        )
        
        # Yield Curve spike: +400 bps sovereign, +100 bps corporate contagion
        df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_to_maturity'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_to_maturity'] + 0.04, 0.0, 0.25
        )
        df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'yield_to_maturity'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'yield_to_maturity'] + 0.01, 0.0, 0.20
        )
        
        # Sovereign recovery rate drops 20% relative (LGD increases)
        df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'recovery_rate'] = np.clip(
            df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'recovery_rate'] * 0.80, 0.0, 1.0
        )
        
    return df_stressed


def process_stressed_pd(model, df_stressed, scenario_name):
    """
    Predicts the PD using the model on stressed inputs.
    If Sovereign Debt Crisis, double the PD for sovereigns with debt-to-gdp > 80%.
    """
    df_pred = model.predict_pd(df_stressed)
    
    if scenario_name == "Sovereign Debt Crisis":
        # Double default probability of high-debt sovereigns (> 80% Debt/GDP)
        high_debt_mask = (df_pred['entity_type'] == 'Sovereign') & (df_pred['debt_to_gdp'] > 0.80)
        df_pred.loc[high_debt_mask, 'predicted_pd'] = np.clip(
            df_pred.loc[high_debt_mask, 'predicted_pd'] * 2.0, 0.0003, 0.999
        )
        
        # Apply standard contagion factor (+10% relative PD) to corporates
        df_pred.loc[df_pred['entity_type'] == 'Corporate', 'predicted_pd'] = np.clip(
            df_pred.loc[df_pred['entity_type'] == 'Corporate', 'predicted_pd'] * 1.10, 0.0003, 0.999
        )
        
    return df_pred
