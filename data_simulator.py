import os
import pandas as pd
import numpy as np

def generate_wholesale_portfolio(seed=42):
    np.random.seed(seed)
    n_accounts = 1000
    
    # 70% Corporate, 30% Sovereign
    n_corporate = 700
    n_sovereign = 300
    
    # Base lists for names
    corp_prefixes = ["Aero", "Global", "Apex", "Nordic", "Pacific", "Summit", "Delta", "Core", "Vanguard", 
                     "Alpha", "Titan", "Vertex", "United", "Quantum", "Dynamic", "Nova", "Beacon", "Crest", "Helix", "Synergy"]
    corp_suffixes = ["Industries", "Technologies", "Energy", "Utilities", "Logistics", "Holdings", "Group", 
                     "Resources", "Telecom", "Pharmaceuticals", "Infrastructure", "Ventures", "Services", "Enterprises"]
    
    sov_countries = [
        "Republic of Patria", "Kingdom of Solaria", "Commonwealth of Orion", "Sovereign State of Elysia", 
        "Federation of Zenith", "United Provinces of Arcadia", "Emirate of Al-Jamil", "Republic of Thalassa", 
        "Kingdom of Freya", "Union of Boreas", "Republic of Zephyr", "Empire of Aethelgard", "Republic of Kaelen", 
        "Sovereign Territory of Oakhaven", "Republic of Valoria", "Federal Republic of Oromia", "Democratic Republic of Gaea", 
        "Kingdom of Hyperborea", "Republic of Austri", "State of Vesper", "Union of Valerius", "Republic of Eldoria",
        "Kingdom of Novaria", "Sovereign State of Thaloria", "Federation of Asturia", "Republic of Lysandra"
    ]
    
    regions = ['US', 'UK', 'Germany', 'Brazil', 'India', 'South Africa']
    sectors = ['Technology', 'Energy', 'Utilities', 'Finance', 'Telecom']
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC/C']
    rating_mapping = {r: i+1 for i, r in enumerate(ratings)} # AAA=1, ..., CCC/C=7
    
    data = []
    
    # --- Generate Corporates (700 accounts) ---
    for i in range(1, n_corporate + 1):
        client_id = f"CORP_{i:03d}"
        client_name = f"{np.random.choice(corp_prefixes)} {np.random.choice(corp_suffixes)}"
        entity_type = "Corporate"
        sector = np.random.choice(sectors)
        region = np.random.choice(regions)
        
        # Rating distribution: skewed towards BBB/A
        rating = np.random.choice(ratings, p=[0.05, 0.10, 0.25, 0.35, 0.15, 0.07, 0.03])
        rating_ord = rating_mapping[rating]
        
        # Financial Ratios: correlated with credit rating
        # Debt-to-Equity (0.1x to 5.0x)
        de_mean = {1: 0.3, 2: 0.5, 3: 0.9, 4: 1.6, 5: 2.6, 6: 3.8, 7: 4.8}[rating_ord]
        de = np.clip(np.random.lognormal(mean=np.log(de_mean), sigma=0.25), 0.1, 5.0)
        
        # Interest Coverage Ratio (-2.0x to 15.0x)
        icr_mean = {1: 12.0, 2: 10.0, 3: 7.0, 4: 4.5, 5: 2.2, 6: 0.8, 7: -0.5}[rating_ord]
        icr = np.clip(np.random.normal(loc=icr_mean, scale=1.5), -2.0, 15.0)
        
        # EBITDA Margin (-10% to 50%)
        margin_mean = {1: 0.40, 2: 0.35, 3: 0.28, 4: 0.20, 5: 0.12, 6: 0.05, 7: -0.05}[rating_ord]
        ebitda_margin = np.clip(np.random.normal(loc=margin_mean, scale=0.08), -0.10, 0.50)
        
        # Annual Sales (€5M to €500M)
        annual_sales = np.clip(np.random.lognormal(mean=np.log(60), sigma=1.0), 5.0, 500.0)
        
        # Treasury Coupon & YTM (rating-driven)
        coupon_base = {1: 0.020, 2: 0.025, 3: 0.035, 4: 0.048, 5: 0.065, 6: 0.090, 7: 0.120}[rating_ord]
        coupon_rate = np.clip(np.random.normal(loc=coupon_base, scale=0.005), 0.010, 0.150)
        ytm = coupon_rate + np.random.normal(loc=0.005, scale=0.002) # typical slight spread
        
        # Remaining Maturity (1.0 to 10.0 years)
        remaining_maturity = np.round(np.random.uniform(1.0, 10.0), 1)
        
        # EAD ($5M to $500M)
        ead = np.clip(np.random.lognormal(mean=np.log(35), sigma=0.8), 5.0, 500.0)
        
        # Seniority & Collateral
        seniority = np.random.choice(["Senior Secured", "Senior Unsecured", "Subordinated"], p=[0.40, 0.45, 0.15])
        if seniority == "Senior Secured":
            collateral_type = np.random.choice(["Real Estate", "Financial Collateral", "Physical Plant"], p=[0.40, 0.40, 0.20])
        elif seniority == "Senior Unsecured":
            collateral_type = np.random.choice(["Unsecured", "Financial Collateral"], p=[0.85, 0.15])
        else:
            collateral_type = "Unsecured"
            
        # Recovery Rate (strongly driven by seniority & collateral)
        if collateral_type == "Financial Collateral":
            rec_rate = np.random.normal(loc=0.85, scale=0.04)
        elif collateral_type == "Real Estate":
            rec_rate = np.random.normal(loc=0.75, scale=0.06)
        elif collateral_type == "Physical Plant":
            rec_rate = np.random.normal(loc=0.60, scale=0.08)
        elif seniority == "Senior Unsecured":
            rec_rate = np.random.normal(loc=0.40, scale=0.08)
        else: # Subordinated, Unsecured
            rec_rate = np.random.normal(loc=0.15, scale=0.05)
        rec_rate = np.clip(rec_rate, 0.0, 1.0)
        
        # Probability of default (logistic function of rating and financials)
        # Calibrated to give overall Corporate default rate around 5-6%
        z = -7.2 + 0.85 * rating_ord + 0.4 * de - 0.2 * icr - 2.5 * ebitda_margin
        pd_prob = 1 / (1 + np.exp(-z))
        historic_default = np.random.binomial(1, pd_prob)
        
        data.append({
            "client_id": client_id,
            "client_name": client_name,
            "entity_type": entity_type,
            "industry_sector": sector,
            "region": region,
            "credit_rating": rating,
            "rating_ordinal": rating_ord,
            "debt_to_equity": np.round(de, 2),
            "interest_coverage": np.round(icr, 2),
            "ebitda_margin": np.round(ebitda_margin, 4),
            "annual_sales_m": np.round(annual_sales, 2),
            "yield_spread_bps": np.nan,
            "debt_to_gdp": np.nan,
            "coupon_rate": np.round(coupon_rate, 4),
            "yield_to_maturity": np.round(ytm, 4),
            "remaining_maturity": remaining_maturity,
            "ead_m": np.round(ead, 2),
            "seniority": seniority,
            "collateral_type": collateral_type,
            "historic_default": historic_default,
            "recovery_rate": np.round(rec_rate, 4)
        })

    # --- Generate Sovereigns (300 accounts) ---
    for i in range(1, n_sovereign + 1):
        client_id = f"SOV_{i:03d}"
        client_name = sov_countries[i % len(sov_countries)]
        # ensure unique names if we run out
        if i >= len(sov_countries):
            client_name = f"{client_name} II"
            
        entity_type = "Sovereign"
        sector = "Sovereign"
        region = np.random.choice(regions)
        
        # Rating distribution: slightly more balanced
        rating = np.random.choice(ratings, p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05])
        rating_ord = rating_mapping[rating]
        
        # Sovereign yield spread over US Treasuries (bps: 10 to 1200)
        spread_base = {1: 20, 2: 45, 3: 90, 4: 220, 5: 420, 6: 750, 7: 1100}[rating_ord]
        spread = np.clip(np.random.normal(loc=spread_base, scale=spread_base * 0.15), 10, 1200)
        
        # Debt-to-GDP (20% to 150% -> 0.20 to 1.50)
        dgdp_base = {1: 0.30, 2: 0.45, 3: 0.60, 4: 0.75, 5: 0.95, 6: 1.15, 7: 1.35}[rating_ord]
        debt_to_gdp = np.clip(np.random.normal(loc=dgdp_base, scale=0.12), 0.20, 1.50)
        
        # Treasury Coupon & YTM (rating-driven)
        # Yield is US 10Y (say 3.5%) + spread in %
        us_treasury_rate = 0.035
        ytm = us_treasury_rate + (spread / 10000.0)
        coupon_rate = ytm - np.random.normal(loc=0.002, scale=0.001) # coupon close to YTM
        coupon_rate = np.clip(coupon_rate, 0.005, 0.150)
        
        remaining_maturity = np.round(np.random.uniform(1.0, 10.0), 1)
        ead = np.clip(np.random.lognormal(mean=np.log(50), sigma=0.7), 5.0, 500.0)
        
        # Sovereigns are usually Senior Unsecured, occasionally Secured or Subordinated
        seniority = np.random.choice(["Senior Unsecured", "Senior Secured", "Subordinated"], p=[0.85, 0.10, 0.05])
        if seniority == "Senior Secured":
            collateral_type = "Financial Collateral"
            rec_rate = np.random.normal(loc=0.85, scale=0.05)
        elif seniority == "Senior Unsecured":
            collateral_type = "Unsecured"
            rec_rate = np.random.normal(loc=0.45, scale=0.08) # sovereign recoveries can be lower or higher, standard 45% LGD
        else: # Subordinated
            collateral_type = "Unsecured"
            rec_rate = np.random.normal(loc=0.20, scale=0.05)
        rec_rate = np.clip(rec_rate, 0.0, 1.0)
        
        # Probability of default (logistic function of rating and sovereign metrics)
        # Calibrated to give overall Sovereign default rate around 3-4%
        z = -8.2 + 0.80 * rating_ord + 0.0015 * spread + 1.5 * debt_to_gdp
        pd_prob = 1 / (1 + np.exp(-z))
        historic_default = np.random.binomial(1, pd_prob)
        
        data.append({
            "client_id": client_id,
            "client_name": client_name,
            "entity_type": entity_type,
            "industry_sector": sector,
            "region": region,
            "credit_rating": rating,
            "rating_ordinal": rating_ord,
            "debt_to_equity": np.nan,
            "interest_coverage": np.nan,
            "ebitda_margin": np.nan,
            "annual_sales_m": np.nan,
            "yield_spread_bps": np.round(spread, 1),
            "debt_to_gdp": np.round(debt_to_gdp, 4),
            "coupon_rate": np.round(coupon_rate, 4),
            "yield_to_maturity": np.round(ytm, 4),
            "remaining_maturity": remaining_maturity,
            "ead_m": np.round(ead, 2),
            "seniority": seniority,
            "collateral_type": collateral_type,
            "historic_default": historic_default,
            "recovery_rate": np.round(rec_rate, 4)
        })
        
    df = pd.DataFrame(data)
    
    # Verify directory exists and save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "wholesale_credit_data.csv"), index=False)
    print(f"Dataset generated with {len(df)} rows at {os.path.join(data_dir, 'wholesale_credit_data.csv')}.")
    print(f"Corporate default rate: {df[df['entity_type']=='Corporate']['historic_default'].mean():.2%}")
    print(f"Sovereign default rate: {df[df['entity_type']=='Sovereign']['historic_default'].mean():.2%}")

if __name__ == "__main__":
    generate_wholesale_portfolio()
