import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")

# Import custom quantitative modules
from credit_risk_model import (
    WholesaleCreditRiskModel,
    calculate_portfolio_metrics,
    apply_stress_scenario,
    process_stressed_pd
)

# Page configuration
st.set_page_config(
    page_title="Basel III Credit & Treasury Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS injection
st.markdown("""
<style>
    /* Light Mode Gradient Header */
    .header-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    }
    .header-title {
        color: #1E3A8A; /* Royal Dark Navy */
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #475569; /* Slate Gray */
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Card layouts (Light Mode) */
    .metric-card {
        background-color: #FFFFFF; /* Pure White */
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 4px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease-in-out, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #2563EB; /* Royal Blue highlight */
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B; /* Medium Slate */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A; /* Dark Slate */
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #059669; /* Emerald Green */
        margin-top: 0.3rem;
    }
    .metric-sub-red {
        font-size: 0.8rem;
        color: #DC2626; /* Ruby Red */
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Data Loading & Training Functions ---

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "wholesale_credit_data.csv")
    if not os.path.exists(csv_path):
        # Fallback to generate data if it doesn't exist
        from data_simulator import generate_wholesale_portfolio
        generate_wholesale_portfolio()
    df = pd.read_csv(csv_path)
    return df

def get_trained_model(model_type):
    df = load_data()
    model = WholesaleCreditRiskModel(model_type=model_type)
    model.train(df)
    return model


# --- Main Application Logic ---

# 1. Title Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Basel III Credit & Treasury Risk Engine</h1>
    <p class="header-subtitle">Chief Investment Office, Treasury & Corporate (CTC) Risk Management Suite</p>
</div>
""", unsafe_allow_html=True)

# 2. Load Data and Sidebar controls
df_raw = load_data()

st.sidebar.markdown("### ⚙️ Risk Configuration")
model_choice = st.sidebar.selectbox(
    "Credit PD Model Classifier",
    ["Logistic Regression", "Random Forest"]
)

scenario_choice = st.sidebar.selectbox(
    "Macro Stress Testing Scenario",
    ["Baseline", "Global Liquidity Squeeze", "Sovereign Debt Crisis"]
)

# Detailed Scenario Explanations in Sidebar
scenario_details = {
    "Baseline": "Normal market environment with baseline estimated PDs and recovery rates.",
    "Global Liquidity Squeeze": "⚠️ Moderate stress:\n- Corporate D/E +30%, ICR -30%\n- Sovereign Yield Spreads +300 bps\n- Parallel Interest Rate Shift of +200 bps",
    "Sovereign Debt Crisis": "🚨 Severe stress:\n- Sovereign Yield Spreads +500 bps, Debt-to-GDP +20%\n- Sovereign Recovery Rates -20% (LGD increases)\n- High-debt (>80% GDP) Sovereigns default probability doubles\n- Yield Curve shifts: Sovereigns +400 bps, Corporates +100 bps"
}
st.sidebar.info(scenario_details[scenario_choice])

# Set default shock variables based on scenario choice
if scenario_choice == "Baseline":
    default_de = 1.0
    default_icr = 1.0
    default_spread = 0.0
    default_gdp = 1.0
    default_yield_shift = 0.0
    default_rec_scale = 1.0
    default_high_debt_pd = 1.0
    default_corp_contagion = 1.0
elif scenario_choice == "Global Liquidity Squeeze":
    default_de = 1.3
    default_icr = 0.7
    default_spread = 300.0
    default_gdp = 1.0
    default_yield_shift = 200.0
    default_rec_scale = 1.0
    default_high_debt_pd = 1.0
    default_corp_contagion = 1.0
else: # Sovereign Debt Crisis
    default_de = 1.0
    default_icr = 1.0
    default_spread = 500.0
    default_gdp = 1.2
    default_yield_shift = 400.0 # using sovereign shift as main parallel yield shift default
    default_rec_scale = 0.8
    default_high_debt_pd = 2.0
    default_corp_contagion = 1.1

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Customize Stress Shocks")
customize_shocks = st.sidebar.checkbox("Override Scenario Shocks", value=False)

if customize_shocks:
    de_scale = st.sidebar.slider("Corporate D/E Multiplier", 1.0, 2.0, default_de, 0.05)
    icr_scale = st.sidebar.slider("Corporate ICR Multiplier", 0.5, 1.0, default_icr, 0.05)
    spread_increase = st.sidebar.slider("Sovereign Spread Increase (bps)", 0, 1000, int(default_spread), 50)
    gdp_scale = st.sidebar.slider("Sovereign Debt-to-GDP Multiplier", 1.0, 1.8, default_gdp, 0.05)
    yield_shift_bps = st.sidebar.slider("Parallel Yield Shift (bps)", -200, 1000, int(default_yield_shift), 50)
    rec_scale = st.sidebar.slider("Sovereign Recovery Scale", 0.5, 1.0, default_rec_scale, 0.05)
    high_debt_pd_scale = st.sidebar.slider("High-Debt Sov PD Multiplier", 1.0, 3.0, default_high_debt_pd, 0.1)
    corp_pd_contagion = st.sidebar.slider("Corporate PD Contagion", 1.0, 2.0, default_corp_contagion, 0.05)
else:
    de_scale = default_de
    icr_scale = default_icr
    spread_increase = default_spread
    gdp_scale = default_gdp
    yield_shift_bps = default_yield_shift
    rec_scale = default_rec_scale
    high_debt_pd_scale = default_high_debt_pd
    corp_pd_contagion = default_corp_contagion

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏦 Regulatory Standards")
st.sidebar.markdown("**Basel III Framework:**")
st.sidebar.markdown("- **PD Floor:** 0.03% (supervisory minimum)")
st.sidebar.markdown("- **Maturity (M):** Standard 2.5y default / Variable")
st.sidebar.markdown("- **Capital Ratio:** 8.0% Regulatory Min")
st.sidebar.markdown("- **CET1 Capital Min:** 4.5%")
st.sidebar.markdown("- **CET1 + CCB Min:** 7.0%")

# Run Risk Engine calculations
model = get_trained_model(model_choice)

# Apply shocks dynamically to df_raw
df_stressed = df_raw.copy()

# Corporate shocks
df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'debt_to_equity'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'debt_to_equity'] * de_scale, 0.1, 5.0
)
df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'interest_coverage'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'interest_coverage'] * icr_scale, -2.0, 15.0
)

# Sovereign shocks
df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_spread_bps'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_spread_bps'] + spread_increase, 10.0, 1200.0
)
df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'debt_to_gdp'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'debt_to_gdp'] * gdp_scale, 0.20, 1.50
)
df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'recovery_rate'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'recovery_rate'] * rec_scale, 0.0, 1.0
)

# Yield shocks for duration / IRRBB
yield_shock_bps_sov = yield_shift_bps
yield_shock_bps_corp = yield_shift_bps if not (scenario_choice == "Sovereign Debt Crisis" and not customize_shocks) else 100

df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_to_maturity'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Sovereign', 'yield_to_maturity'] + (yield_shock_bps_sov / 10000.0), 0.0, 0.30
)
df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'yield_to_maturity'] = np.clip(
    df_stressed.loc[df_stressed['entity_type'] == 'Corporate', 'yield_to_maturity'] + (yield_shock_bps_corp / 10000.0), 0.0, 0.30
)

# Re-run PD models on stressed financials
df_pred = model.predict_pd(df_stressed)

# Apply dynamic PD multipliers (high-debt sovereign doubling, contagion)
high_debt_mask = (df_pred['entity_type'] == 'Sovereign') & (df_pred['debt_to_gdp'] > 0.80)
df_pred.loc[high_debt_mask, 'predicted_pd'] = np.clip(
    df_pred.loc[high_debt_mask, 'predicted_pd'] * high_debt_pd_scale, 0.0003, 0.999
)
df_pred.loc[df_pred['entity_type'] == 'Corporate', 'predicted_pd'] = np.clip(
    df_pred.loc[df_pred['entity_type'] == 'Corporate', 'predicted_pd'] * corp_pd_contagion, 0.0003, 0.999
)

# Calculate Basel & CTC metrics
portfolio_df = calculate_portfolio_metrics(df_pred)

# Calculate Basel & CTC metrics for baseline to compare
df_base_pred = model.predict_pd(df_raw)
baseline_portfolio_df = calculate_portfolio_metrics(df_base_pred)


# --- Dashboard Tabs ---
tab_exec, tab_explorer, tab_tech = st.tabs(["📊 Executive & Business Risk", "🔍 Account Risk Explorer", "🔬 Technical & Quantitative Validation"])

# ==========================================
# TAB 1: EXECUTIVE & BUSINESS RISK
# ==========================================
with tab_exec:
    
    # --- KPI Cards Row ---
    st.markdown("### Portfolio Key Risk Indicators (KRIs)")
    
    # Metrics calculations
    total_ead = portfolio_df['ead_m'].sum()
    total_el = portfolio_df['expected_loss_m'].sum()
    total_rwa = portfolio_df['rwa_m'].sum()
    min_cap_req = portfolio_df['min_capital_buffer_m'].sum()
    total_hqla = portfolio_df['eligible_hqla_m'].sum()
    
    # Calculate Interest Rate Risk impact (Valuation drop)
    # Valuation Change = - Duration * EAD * Yield Shock (bps/10000)
    def calc_val_change(row):
        shock = yield_shock_bps_sov if row['entity_type'] == 'Sovereign' else yield_shock_bps_corp
        shock_decimal = shock / 10000.0
        return -row['duration'] * row['ead_m'] * shock_decimal
        
    portfolio_df['val_change_m'] = portfolio_df.apply(calc_val_change, axis=1)
    total_val_change = portfolio_df['val_change_m'].sum()
    
    # Baseline comparison metrics
    base_el = baseline_portfolio_df['expected_loss_m'].sum()
    base_rwa = baseline_portfolio_df['rwa_m'].sum()
    base_min_cap = baseline_portfolio_df['min_capital_buffer_m'].sum()
    base_hqla = baseline_portfolio_df['eligible_hqla_m'].sum()
    
    # Compute relative shifts
    el_diff_pct = ((total_el - base_el) / base_el) * 100 if base_el > 0 else 0
    rwa_diff_pct = ((total_rwa - base_rwa) / base_rwa) * 100 if base_rwa > 0 else 0
    hqla_diff_pct = ((total_hqla - base_hqla) / base_hqla) * 100 if base_hqla > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Portfolio Exposure (EAD)</div>
            <div class="metric-val">${total_ead/1000:.2f}B</div>
            <div class="metric-sub">${total_ead:,.1f} Million</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        el_color = "metric-sub-red" if el_diff_pct > 1 else "metric-sub"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Loss (EL)</div>
            <div class="metric-val">${total_el:,.1f}M</div>
            <div class="{el_color}">
                {"+" if el_diff_pct >= 0 else ""}{el_diff_pct:.1f}% vs. Baseline<br>
                ({(total_el / total_ead) * 100:.3f}% of Book)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        rwa_color = "metric-sub-red" if rwa_diff_pct > 1 else "metric-sub"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk-Weighted Assets (RWA)</div>
            <div class="metric-val">${total_rwa/1000:.2f}B</div>
            <div class="{rwa_color}">
                {"+" if rwa_diff_pct >= 0 else ""}{rwa_diff_pct:.1f}% vs. Baseline<br>
                (Avg RW: {(total_rwa / total_ead) * 100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Min Capital Buffer (8% RWA)</div>
            <div class="metric-val">${min_cap_req:,.1f}M</div>
            <div class="metric-sub">
                CET1 Min (4.5%): ${total_rwa * 0.045:,.1f}M<br>
                CET1 + CCB (7%): ${total_rwa * 0.070:,.1f}M
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        # Check if interest rate valuation drop is non-zero
        val_color = "metric-sub-red" if total_val_change < 0 else "metric-sub"
        val_text = f"Eligible HQLA: ${total_hqla:,.1f}M"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IRRBB & HQLA Liquidity</div>
            <div class="metric-val" style="color: {'#DC2626' if total_val_change < 0 else '#0F172A'}">${total_val_change:,.1f}M</div>
            <div class="{val_color}">
                EVE Sensitivity (DV01: ${portfolio_df['dv01_k'].sum():,.1f}K)<br>
                HQLA/EAD: {(total_hqla/total_ead)*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # --- Noob-Friendly KRI Explainer ---
    with st.expander("🧑‍🏫 New to Banking? Click here for a plain-English explanation of these 5 numbers"):
        st.markdown("""
**Card 1 — Total Portfolio Exposure (EAD = $52.99B)**
This is the total amount of money our bank has lent out to companies and governments worldwide. If every single borrower defaulted at once, this is the maximum we could lose.

**Card 2 — Expected Loss (EL = $2,071.5M)**
In any normal year, some borrowers will default and we will lose some money. This is our best estimate of how much money we expect to lose on average per year. Think of it like a provision set aside in advance.

**Card 3 — Risk-Weighted Assets (RWA = $76.13B)**
Not all loans are equally risky. A loan to the US Government is safer than a loan to a small risky company. RWA adjusts our total loans by their riskiness. Even though we lent $52.99B, the risk-adjusted equivalent is $76.13B.

**Card 4 — Min Capital Buffer ($6,090.4M)**
Regulators say: you must keep 8% of your RWA as your own cash safety net. This $6.09B is the minimum equity our bank must hold at all times so that if loans go bad, depositors do not lose their savings.

**Card 5 — IRRBB & HQLA Liquidity ($0.0M)**
If interest rates suddenly rise, existing loans lose market value (like old bonds becoming less valuable). The $0.0M shows the current valuation loss from rate changes. HQLA/EAD 43.4% means 43.4% of our loan book is in safe government bonds we can sell instantly in a crisis.
        """)

    st.markdown("---")
    
    # --- Visualization Section ---
    st.markdown("### Portfolio Risk Concentration & Allocation Analysis")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Sector Concentration Map (Treemap) - Clickable drill-down with Others grouping
        sector_df = portfolio_df.groupby('industry_sector').agg(
            ead_m=('ead_m', 'sum'),
            rwa_m=('rwa_m', 'sum'),
            el_m=('expected_loss_m', 'sum')
        ).reset_index()
        sector_df['risk_weight_pct'] = (sector_df['rwa_m'] / sector_df['ead_m']) * 100

        # Always show Top 5 sectors by EAD, rest go into 'Others'
        sector_df_sorted = sector_df.sort_values('ead_m', ascending=False)
        TOP_N_SECTORS = 5
        main_sector_names = sector_df_sorted.head(TOP_N_SECTORS)['industry_sector'].tolist()
        small_sector_names = sector_df_sorted.tail(len(sector_df_sorted) - TOP_N_SECTORS)['industry_sector'].tolist()

        main_sectors = sector_df_sorted.head(TOP_N_SECTORS).copy()
        if small_sector_names:
            other_sectors_rows = sector_df[sector_df['industry_sector'].isin(small_sector_names)]
            others_row = pd.DataFrame([{
                'industry_sector': 'Others',
                'ead_m': other_sectors_rows['ead_m'].sum(),
                'rwa_m': other_sectors_rows['rwa_m'].sum(),
                'el_m': other_sectors_rows['el_m'].sum(),
                'risk_weight_pct': (other_sectors_rows['rwa_m'].sum() / other_sectors_rows['ead_m'].sum()) * 100
            }])
            main_sectors = pd.concat([main_sectors, others_row], ignore_index=True)
        
        fig_sec = px.treemap(
            main_sectors,
            path=['industry_sector'],
            values='ead_m',
            color='risk_weight_pct',
            color_continuous_scale='Turbo',
            labels={'risk_weight_pct': 'Avg Risk Weight %', 'ead_m': 'Exposure ($M)'},
            title="Portfolio Exposure Concentration & Risk-Weight by Sector"
        )
        fig_sec.update_layout(template="plotly_white", margin=dict(t=40, l=10, r=10, b=10), height=380)
        fig_sec.update_traces(
            hovertemplate="<b>%{label}</b><br>Exposure: $%{value:,.1f}M<br>Avg Risk Weight: %{color:.1f}%<extra></extra>"
        )
        st.plotly_chart(fig_sec, use_container_width=True)
        st.info("💡 **Simplifying the Treemap:** Box size = how much money we lent to that sector. Color (red = risky, blue = safe) = Basel III penalty. Smallest sectors are grouped into 'Others'. Select a sector below to see its top firms.")

        # --- Sector Drill-Down ---
        all_sectors = sorted(portfolio_df['industry_sector'].dropna().unique().tolist())
        selected_sector = st.selectbox("Select a Sector to See Its Top 10 Firms", ["— Select a Sector —"] + all_sectors, key="sector_drilldown")

        sector_firms = None
        if selected_sector != "— Select a Sector —":
            sector_firms = portfolio_df[portfolio_df['industry_sector'] == selected_sector].sort_values('ead_m', ascending=False).head(10)
            st.markdown(f"**Top 10 Firms in {selected_sector} Sector:**")

        if sector_firms is not None and not sector_firms.empty:
            display_cols = ['client_name', 'entity_type', 'industry_sector', 'region', 'credit_rating', 'ead_m', 'rwa_m', 'expected_loss_m']
            display_df = sector_firms[display_cols].copy()
            display_df.columns = ['Client Name', 'Type', 'Sector', 'Region', 'Rating', 'EAD ($M)', 'RWA ($M)', 'Exp. Loss ($M)']
            display_df = display_df.round(2)
            st.dataframe(display_df, use_container_width=True)
            # Mini metrics for selected sector - using markdown to avoid truncation
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.8rem;color:#64748B;text-transform:uppercase'>Total Sector EAD</div><div style='font-size:1.4rem;font-weight:700;color:#0F172A'>${sector_firms['ead_m'].sum():,.1f}M</div></div>", unsafe_allow_html=True)
            with mc2:
                rw_val = (sector_firms['rwa_m'].sum()/sector_firms['ead_m'].sum())*100
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.8rem;color:#64748B;text-transform:uppercase'>Avg Risk Weight</div><div style='font-size:1.4rem;font-weight:700;color:#0F172A'>{rw_val:.1f}%</div></div>", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.8rem;color:#64748B;text-transform:uppercase'>Expected Loss</div><div style='font-size:1.4rem;font-weight:700;color:#DC2626'>${sector_firms['expected_loss_m'].sum():,.1f}M</div></div>", unsafe_allow_html=True)


    with row1_col2:
        # Regional Exposure Concentration (Bar chart) - Clickable drill-down with Others grouping
        region_agg = portfolio_df.groupby(['region', 'entity_type']).agg(
            ead_m=('ead_m', 'sum'),
            rwa_m=('rwa_m', 'sum'),
            el_m=('expected_loss_m', 'sum')
        ).reset_index()

        # Always show Top 5 regions by EAD, rest go into 'Others'
        region_totals = region_agg.groupby('region')['ead_m'].sum().reset_index()
        region_totals.columns = ['region', 'total_ead']
        region_totals = region_totals.sort_values('total_ead', ascending=False)
        TOP_N_REGIONS = 5
        main_region_names = region_totals.head(TOP_N_REGIONS)['region'].tolist()
        small_region_names = region_totals.tail(len(region_totals) - TOP_N_REGIONS)['region'].tolist()

        region_df = region_agg[region_agg['region'].isin(main_region_names)].copy()
        if small_region_names:
            others_corp = region_agg[(region_agg['region'].isin(small_region_names)) & (region_agg['entity_type'] == 'Corporate')]
            others_sov = region_agg[(region_agg['region'].isin(small_region_names)) & (region_agg['entity_type'] == 'Sovereign')]
            if not others_corp.empty:
                region_df = pd.concat([region_df, pd.DataFrame([{'region': 'Others', 'entity_type': 'Corporate', 'ead_m': others_corp['ead_m'].sum(), 'rwa_m': others_corp['rwa_m'].sum(), 'el_m': others_corp['el_m'].sum()}])], ignore_index=True)
            if not others_sov.empty:
                region_df = pd.concat([region_df, pd.DataFrame([{'region': 'Others', 'entity_type': 'Sovereign', 'ead_m': others_sov['ead_m'].sum(), 'rwa_m': others_sov['rwa_m'].sum(), 'el_m': others_sov['el_m'].sum()}])], ignore_index=True)

        
        fig_reg = px.bar(
            region_df,
            x='region',
            y='ead_m',
            color='entity_type',
            color_discrete_map={'Corporate': '#1E3A8A', 'Sovereign': '#10B981'},
            title="Exposure (EAD) Concentration by Region & Entity Type",
            labels={'ead_m': 'Exposure EAD ($ Millions)', 'region': 'Region'},
            barmode='group'
        )
        fig_reg.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        fig_reg.update_traces(
            hovertemplate="<b>%{x}</b><br>Exposure: $%{y:,.1f}M<extra></extra>"
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        st.info("💡 **Simplifying Regional Risk:** Blue = Corporate loans, Green = Government bonds. Taller bar = more money lent there. Small regions are grouped into 'Others'. Select a region below to see detailed breakdown.")

        # --- Region Drill-Down ---
        all_regions = sorted(portfolio_df['region'].dropna().unique().tolist())
        selected_region = st.selectbox("Select a Region to See Detailed Breakdown", ["— Select a Region —"] + all_regions, key="region_drilldown")

        region_detail = None
        if selected_region != "— Select a Region —":
            region_detail = portfolio_df[portfolio_df['region'] == selected_region]
            st.markdown(f"**{selected_region} — Risk Breakdown:**")
        if region_detail is not None:

            # Summary metrics - using markdown to avoid truncation
            rmc1, rmc2, rmc3, rmc4 = st.columns(4)
            with rmc1:
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.75rem;color:#64748B;text-transform:uppercase'>Total EAD</div><div style='font-size:1.3rem;font-weight:700;color:#0F172A'>${region_detail['ead_m'].sum():,.0f}M</div></div>", unsafe_allow_html=True)
            with rmc2:
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.75rem;color:#64748B;text-transform:uppercase'>Total RWA</div><div style='font-size:1.3rem;font-weight:700;color:#0F172A'>${region_detail['rwa_m'].sum():,.0f}M</div></div>", unsafe_allow_html=True)
            with rmc3:
                rw_reg = (region_detail['rwa_m'].sum()/region_detail['ead_m'].sum())*100
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.75rem;color:#64748B;text-transform:uppercase'>Avg Risk Weight</div><div style='font-size:1.3rem;font-weight:700;color:#0F172A'>{rw_reg:.1f}%</div></div>", unsafe_allow_html=True)
            with rmc4:
                st.markdown(f"<div style='text-align:center'><div style='font-size:0.75rem;color:#64748B;text-transform:uppercase'>Exp. Loss</div><div style='font-size:1.3rem;font-weight:700;color:#DC2626'>${region_detail['expected_loss_m'].sum():,.0f}M</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            # Rating distribution in that region
            rating_dist = region_detail.groupby('credit_rating')['ead_m'].sum().reset_index()
            rating_dist.columns = ['Credit Rating', 'EAD ($M)']
            st.markdown("Rating Distribution:")
            fig_reg_rating = px.bar(
                rating_dist,
                x='Credit Rating',
                y='EAD ($M)',
                color_discrete_sequence=['#1E3A8A'],
                height=200
            )
            fig_reg_rating.update_layout(template="plotly_white", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_reg_rating, use_container_width=True)
            # Top clients in that region
            top_region_clients = region_detail.sort_values('ead_m', ascending=False).head(8)
            disp2 = top_region_clients[['client_name', 'entity_type', 'region', 'credit_rating', 'ead_m', 'expected_loss_m', 'hqla_class']].copy()
            disp2.columns = ['Client Name', 'Type', 'Region', 'Rating', 'EAD ($M)', 'Exp. Loss ($M)', 'HQLA Class']
            st.dataframe(disp2.round(2), use_container_width=True)


    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Rating-wise EAD and average RWA Bar Chart
        rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC/C']
        rating_df = portfolio_df.groupby('credit_rating').agg(
            ead_m=('ead_m', 'sum'),
            rwa_m=('rwa_m', 'sum')
        ).reindex(rating_order).reset_index()
        rating_df['avg_rw'] = (rating_df['rwa_m'] / rating_df['ead_m']) * 100
        
        fig_rat = go.Figure()
        fig_rat.add_trace(go.Bar(
            x=rating_df['credit_rating'],
            y=rating_df['ead_m'],
            name='Exposure (EAD) $M',
            marker_color='#1E3A8A',
            yaxis='y'
        ))
        fig_rat.add_trace(go.Scatter(
            x=rating_df['credit_rating'],
            y=rating_df['avg_rw'],
            name='Avg Risk Weight %',
            line=dict(color='#10B981', width=3),
            yaxis='y2'
        ))
        
        fig_rat.update_layout(
            title="Exposure (EAD) and Avg Risk Weight by Credit Rating",
            yaxis=dict(title="EAD ($ Millions)", side="left"),
            yaxis2=dict(title="Average Risk Weight (%)", side="right", overlaying="y", range=[0, 400]),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        st.plotly_chart(fig_rat, use_container_width=True)
        st.info("💡 **Simplifying Ratings & Risk Weights:** Blue bars = how much we lent to each rating group. Green line = penalty multiplier (Risk Weight). As rating falls from AAA to CCC, the green line shoots up — meaning the bank must lock more cash for riskier borrowers.")
        
    with row2_col2:
        # HQLA Liquidity Composition
        hqla_df = portfolio_df.groupby('hqla_class').agg(
            ead_m=('ead_m', 'sum'),
            eligible_hqla_m=('eligible_hqla_m', 'sum')
        ).reset_index()
        
        fig_liq = px.pie(
            hqla_df,
            values='ead_m',
            names='hqla_class',
            color='hqla_class',
            color_discrete_map={
                'Level 1 HQLA': '#10B981',      # emerald green
                'Level 2A HQLA': '#2563EB',     # royal blue
                'Level 2B HQLA': '#F59E0B',     # amber
                'Non-HQLA': '#EF4444'           # ruby red
            },
            hole=0.4,
            title="Basel III HQLA Liquidity Class Exposure Distribution"
        )
        fig_liq.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_liq, use_container_width=True)
        st.info("💡 **Simplifying HQLA Liquidity:** Green (Level 1) = safest cash equivalent (US Treasuries, 0% haircut). Blue (Level 2A) = 15% haircut. Orange (Level 2B) = 50% haircut. Red (Non-HQLA) = cannot be counted as emergency cash at all (100% haircut).")

    st.markdown("---")

    # --- Portfolio Health Scorecard ---
    st.markdown("### 🏥 Portfolio Health Scorecard")
    st.markdown("A quick-glance regulatory health check across five key dimensions of the portfolio.")

    def score_traffic_light(value, green_thresh, amber_thresh, higher_is_better=True):
        if higher_is_better:
            if value >= green_thresh: return "🟢 Healthy"
            elif value >= amber_thresh: return "🟡 Watch"
            else: return "🔴 At Risk"
        else:
            if value <= green_thresh: return "🟢 Healthy"
            elif value <= amber_thresh: return "🟡 Watch"
            else: return "🔴 At Risk"

    avg_rw = (total_rwa / total_ead) * 100
    el_pct = (total_el / total_ead) * 100
    hqla_ratio = (total_hqla / total_ead) * 100
    capital_ratio = (min_cap_req / total_rwa) * 100
    non_hqla_pct = (portfolio_df[portfolio_df['hqla_class'] == 'Non-HQLA']['ead_m'].sum() / total_ead) * 100

    scorecard_data = [
        {
            "Metric": "Avg Risk Weight",
            "Value": f"{avg_rw:.1f}%",
            "Benchmark": "< 100% = Healthy",
            "Status": score_traffic_light(avg_rw, 100, 130, higher_is_better=False),
            "What it means": "Lower = safer loan portfolio composition"
        },
        {
            "Metric": "Expected Loss Rate",
            "Value": f"{el_pct:.2f}%",
            "Benchmark": "< 3% = Healthy",
            "Status": score_traffic_light(el_pct, 3.0, 6.0, higher_is_better=False),
            "What it means": "Estimated annual loan loss as % of book"
        },
        {
            "Metric": "HQLA / EAD Ratio",
            "Value": f"{hqla_ratio:.1f}%",
            "Benchmark": "> 30% = Healthy",
            "Status": score_traffic_light(hqla_ratio, 30, 15, higher_is_better=True),
            "What it means": "Emergency liquid cash available vs. total exposure"
        },
        {
            "Metric": "Capital Adequacy",
            "Value": f"{capital_ratio:.1f}%",
            "Benchmark": "> 8% = Compliant",
            "Status": score_traffic_light(capital_ratio, 8.0, 6.0, higher_is_better=True),
            "What it means": "Actual capital buffer vs. regulatory minimum"
        },
        {
            "Metric": "Non-HQLA Concentration",
            "Value": f"{non_hqla_pct:.1f}%",
            "Benchmark": "< 40% = Healthy",
            "Status": score_traffic_light(non_hqla_pct, 40, 55, higher_is_better=False),
            "What it means": "Portion of book that cannot be liquidated in a crisis"
        },
    ]
    scorecard_df = pd.DataFrame(scorecard_data)
    st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # --- Basel III for Business Reference ---
    with st.expander("📚 Basel III Regulatory Framework & CIB Metrics - Business Reference Guide"):
        st.markdown("""
        ### Basel III Foundation IRB (Internal Ratings-Based) Approach
        Under the Basel Framework, banks are permitted to use their internal quantitative models to estimate the credit risk parameters of their counterparties. In **Foundation IRB (F-IRB)**, banks estimate the **Probability of Default (PD)** while other parameters like **Loss Given Default (LGD)** and **Maturity (M)** are determined by regulatory guidelines.
        
        #### Key Mathematical Mechanics:
        
        1. **Asset Correlation (\(R\))**:
           Reflects how a borrower's default risk changes with the broader economic cycle. Larger corporate and sovereign assets are assumed to have higher systemic correlations.
           $$R = 0.12 \\times \\left(\\frac{1 - e^{-50 \\times PD}}{1 - e^{-50}}\\right) + 0.24 \\times \\left(1 - \\frac{1 - e^{-50 \\times PD}}{1 - e^{-50}}\\right)$$
           *For Small and Medium Entities (SMEs) with Sales (\(S\)) between €5M and €50M, correlation is adjusted downward (SME Size Adjustment) to reflect their lower systematic exposure.*
        
        2. **Maturity Adjustment factor (\(b(PD)\))**:
           Adjusts risk weight for the maturity slope. Longer maturities increase capital requirements, with the effect scaling based on PD:
           $$b(PD) = (0.11852 - 0.05478 \\times \\ln(PD))^2$$
        
        3. **Regulatory Capital Requirement (\(K\))**:
           Represents the capital charge ratio (percentage of exposure). Uses the Gaussian copula framework (single systemic factor) set at a **99.9% confidence level**:
           $$K = \\left[ LGD \\times N\\left( \\frac{N^{-1}(PD)}{\\sqrt{1-R}} + \\sqrt{\\frac{R}{1-R}} \\cdot N^{-1}(0.999) \\right) - PD \\times LGD \\right] \\times \\frac{1 + (M - 2.5) \\cdot b(PD)}{1 - 1.5 \\cdot b(PD)}$$
           Where \(N(x)\) is the cumulative standard normal distribution, \(N^{-1}(p)\) is the inverse cumulative normal distribution, and \(M\) is the remaining maturity in years.
        
        4. **Risk-Weighted Assets (RWA) & Capital Buffer**:
           $$RWA = 12.5 \\times K \\times EAD$$
           Multiplying by 12.5 means the capital charge (\(K \\times EAD\)) represents exactly **8.0%** of RWA. The bank must maintain at least:
           - **4.5% of RWA** in Common Equity Tier 1 (CET1) capital.
           - **7.0% of RWA** including the Capital Conservation Buffer (CCB).
           
        ---
        
        ### CTC Treasury Risk Metrics
        
        #### 1. Liquidity Coverage Ratio (LCR) & High-Quality Liquid Assets (HQLA)
        The LCR requires banks to hold an amount of HQLA that exceeds expected net cash outflows over a 30-day stress period. 
        - **Level 1 HQLA** (AAA/AA sovereigns) can be liquidated with **0% haircut**.
        - **Level 2A HQLA** (A sovereigns, AAA/A corporates) are subject to a **15% haircut**.
        - **Level 2B HQLA** (BBB corporates) are subject to a **50% haircut**.
        - Non-HQLA (below BBB or subordinated) have a **100% haircut** (cannot be counted).
        
        #### 2. Interest Rate Risk in the Banking Book (IRRBB)
        Treasury portfolios are exposed to interest rate movements. We model this using:
        - **Modified Duration (\(D_{mod}\))**: The price sensitivity of the asset to changes in yields.
          $$D_{mod} = \\frac{\\text{Maturity}}{1 + \\text{YTM}}$$
        - **DV01**: The dollar change in portfolio value for a 1 basis point (0.01%) parallel shift in the yield curve.
          $$DV01 = D_{mod} \\times EAD \\times 0.0001$$
        - **Economic Value of Equity (EVE) Sensitivity**: The net change in portfolio valuation due to interest rate shocks (\(\\Delta y\)).
        """)

# ==========================================
# TAB 2: INDIVIDUAL CLIENT RISK EXPLORER
# ==========================================
with tab_explorer:
    st.markdown("### 🔍 Individual Account Risk Explorer")
    st.markdown("Search and select any corporate or sovereign borrower to review their detailed credit rating, financial ratios, Basel III capital requirements, and treasury liquidity metrics.")
    
    # Search bar / Selectbox with search
    search_query = st.text_input("Search Client by Name or Client ID", "")
    
    # Filter clients based on search query
    explorer_df = portfolio_df.copy()
    if search_query:
        # Case insensitive search
        mask = explorer_df['client_name'].str.contains(search_query, case=False) | \
               explorer_df['client_id'].astype(str).str.contains(search_query)
        filtered_df = explorer_df[mask]
    else:
        filtered_df = explorer_df
        
    if filtered_df.empty:
        st.warning("No clients found matching the search criteria.")
    else:
        # Limit list for selection
        client_options = filtered_df.apply(lambda row: f"{row['client_name']} ({row['client_id']})", axis=1).tolist()
        selected_client_str = st.selectbox("Select Client Profile to Deep-Dive", client_options)
        
        # Extract client ID from selection
        selected_id = selected_client_str.split('(')[-1].replace(')', '')
        client_row = explorer_df[explorer_df['client_id'] == selected_id].iloc[0]
        
        # Layout the Client details
        col_c1, col_c2 = st.columns([1, 2])
        
        with col_c1:
            st.markdown("#### 👤 Client Information")
            st.markdown(f"""
            *   **Client Name:** {client_row['client_name']}
            *   **Client ID:** {client_row['client_id']}
            *   **Entity Type:** {client_row['entity_type']}
            *   **Region:** {client_row['region']}
            *   **Industry Sector:** {client_row['industry_sector']}
            """)
            
            st.markdown("#### 📈 Credit Assessment")
            st.markdown(f"""
            *   **Credit Rating:** `{client_row['credit_rating']}`
            *   **Basel Risk Weight (RW):** `{(client_row['rwa_m'] / client_row['ead_m']) * 100:.1f}%`
            *   **EAD (Exposure):** `${client_row['ead_m']:.2f} Million`
            *   **Basel Capital Charge:** `${(client_row['capital_requirement_K'] * client_row['ead_m']) * 1000:.2f} Thousand`
            *   **Loss Given Default (LGD):** `{client_row['lgd'] * 100:.1f}%`
            *   **Seniority:** {client_row['seniority']}
            """)
            
        with col_c2:
            st.markdown("#### 📊 Financial Ratios & Credit Driver Metrics")
            
            # Check entity type for specific variables
            if client_row['entity_type'] == 'Corporate':
                st.markdown(f"""
                *   **Debt-to-Equity (D/E) Ratio:** `{client_row['debt_to_equity']:.2f}x`
                *   **Interest Coverage Ratio (ICR):** `{client_row['interest_coverage']:.2f}x`
                *   **EBITDA Margin:** `{client_row['ebitda_margin'] * 100:.1f}%`
                """)
                icr_val = client_row['interest_coverage']
                icr_status = "Healthy (Low Risk)" if icr_val > 3.0 else "Stressed (Elevated Risk)" if icr_val > 1.5 else "Critical Default Risk"
                st.markdown(f"**Corporate Health Check:** Corporate interest coverage indicates `{icr_status}` state.")
            else:  # Sovereign
                st.markdown(f"""
                *   **National Debt-to-GDP:** `{client_row['debt_to_gdp'] * 100:.1f}%`
                *   **Yield Spread (over US Treasuries):** `{client_row['yield_spread_bps']:.0f} bps`
                """)
                spread_val = client_row['yield_spread_bps']
                spread_status = "Stable Sovereign Risk" if spread_val < 150 else "Elevated Risk Premium" if spread_val < 400 else "High Credit Distress"
                st.markdown(f"**Sovereign Health Check:** Credit market spread indicates `{spread_status}`.")
            
            st.markdown("#### 🏦 Treasury Liquidity & IRRBB Risk")
            st.markdown(f"""
            *   **HQLA Eligibility Class:** `{client_row['hqla_class']}`
            *   **Eligible Liquid Asset Value:** `${client_row['eligible_hqla_m']:.2f} Million` (After Basel haircut)
            *   **Residual Maturity:** `{client_row['remaining_maturity']:.2f} Years`
            *   **Yield to Maturity (YTM):** `{client_row['yield_to_maturity'] * 100:.2f}%`
            *   **Modified Duration:** `{client_row['duration']:.2f} Years`
            *   **DV01 Sensitivity:** `${client_row['dv01_k']:.3f} Thousand` (Valuation impact of 1bp rate shift)
            """)

# ==========================================
# TAB 3: TECHNICAL & DEVELOPER TAB
# ==========================================
with tab_tech:
    st.markdown("### Machine Learning Model Diagnostics & Validation")
    
    col_t1, col_t2 = st.columns([1, 2])
    
    with col_t1:
        st.markdown("#### Model Parameters")
        st.write(f"**Selected Architecture:** {model_choice}")
        if model_choice == "Random Forest":
            st.code("""
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    stratify=True
)
            """, language="python")
        else:
            st.code("""
LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
            """, language="python")
            
        st.markdown("#### Model Performance Metrics (OOS Test Set)")
        
        # Display validation metrics
        corp_auc = model.corp_metrics['auc']
        sov_auc = model.sov_metrics['auc']
        
        st.metric("Corporate Model ROC-AUC", f"{corp_auc:.4f}")
        st.metric("Sovereign Model ROC-AUC", f"{sov_auc:.4f}")
        
        # Display small classification table
        st.markdown("**Corporate Classification Metrics:**")
        corp_rep = model.corp_metrics['report']['macro avg']
        st.dataframe(pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Score': [corp_rep['precision'], corp_rep['recall'], corp_rep['f1-score']]
        }).set_index('Metric'), use_container_width=True)
        
        st.markdown("**Sovereign Classification Metrics:**")
        sov_rep = model.sov_metrics['report']['macro avg']
        st.dataframe(pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Score': [sov_rep['precision'], sov_rep['recall'], sov_rep['f1-score']]
        }).set_index('Metric'), use_container_width=True)
        
    with col_t2:
        st.markdown("#### Out-of-Sample ROC-AUC Curves")
        
        # Compute ROC curves
        c_fpr, c_tpr, _ = roc_curve(model.corp_metrics['test_y'], model.corp_metrics['test_prob'])
        s_fpr, s_tpr, _ = roc_curve(model.sov_metrics['test_y'], model.sov_metrics['test_prob'])
        
        fig_roc = go.Figure()
        
        # Corporate
        fig_roc.add_trace(go.Scatter(
            x=c_fpr, y=c_tpr,
            mode='lines',
            name=f'Corporate Portfolio (AUC = {corp_auc:.3f})',
            line=dict(color='#00B4D8', width=3)
        ))
        
        # Sovereign
        fig_roc.add_trace(go.Scatter(
            x=s_fpr, y=s_tpr,
            mode='lines',
            name=f'Sovereign Portfolio (AUC = {sov_auc:.3f})',
            line=dict(color='#00FFA3', width=3)
        ))
        
        # Diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess (AUC = 0.50)',
            line=dict(color='#64748B', dash='dash')
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.7)"),
            margin=dict(t=30, l=10, r=10, b=10),
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info("💡 **Simplifying the ROC-AUC Curve:** This chart measures how accurately our machine learning model classifies defaults. An Area Under the Curve (AUC) of 1.0 means perfect classification (identifying every default), while 0.5 is equivalent to a coin toss.")
        
    st.markdown("---")
    
    col_feat1, col_feat2 = st.columns(2)
    
    # Feature Importances
    importances = model.get_feature_importance()
    
    with col_feat1:
        st.markdown("#### Corporate PD Model: Feature Importance")
        corp_feat_df = pd.DataFrame({
            'Feature': list(importances['Corporate'].keys()),
            'Importance': list(importances['Corporate'].values())
        }).sort_values('Importance', ascending=True)
        
        # Clean labels for presentation
        label_map = {
            'debt_to_equity': 'Debt-to-Equity (D/E)',
            'interest_coverage': 'Interest Coverage Ratio (ICR)',
            'ebitda_margin': 'EBITDA Margin',
            'rating_ordinal': 'Credit Rating (Ordinal)'
        }
        corp_feat_df['Feature'] = corp_feat_df['Feature'].map(label_map)
        
        fig_c_feat = px.bar(
            corp_feat_df,
            y='Feature',
            x='Importance',
            orientation='h',
            title="Corporate Feature Contribution to Default Risk",
            color_discrete_sequence=['#1E3A8A']
        )
        fig_c_feat.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c_feat, use_container_width=True)
        
    with col_feat2:
        st.markdown("#### Sovereign PD Model: Feature Importance")
        sov_feat_df = pd.DataFrame({
            'Feature': list(importances['Sovereign'].keys()),
            'Importance': list(importances['Sovereign'].values())
        }).sort_values('Importance', ascending=True)
        
        label_map_sov = {
            'yield_spread_bps': 'Sovereign Yield Spread (bps)',
            'debt_to_gdp': 'Debt-to-GDP Ratio',
            'rating_ordinal': 'Credit Rating (Ordinal)'
        }
        sov_feat_df['Feature'] = sov_feat_df['Feature'].map(label_map_sov)
        
        fig_s_feat = px.bar(
            sov_feat_df,
            y='Feature',
            x='Importance',
            orientation='h',
            title="Sovereign Feature Contribution to Default Risk",
            color_discrete_sequence=['#10B981']
        )
        fig_s_feat.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s_feat, use_container_width=True)
        
    st.info("💡 **Simplifying Feature Importance:** This shows which inputs the model values most when predicting default. Corporates are driven by cash flow (EBITDA margin and Interest Coverage) and leverage. Sovereigns are driven by yield spreads (bps) over US Treasuries and national Debt-to-GDP ratios.")
    st.markdown("---")
    
    # Non-linear Basel RWA demonstration
    st.markdown("#### Basel III Credit Risk Non-linearity (Risk Weight vs. Probability of Default)")
    
    # Generate Basel III IRB benchmark curves to show the non-linear relationship of PD vs Risk Weight
    pd_range = np.logspace(np.log10(0.0003), np.log10(0.20), 100)
    from credit_risk_model import basel_capital_requirement_k
    
    rw_curve_data = []
    # Calculate for typical corporate LGDs
    for lgd in [0.35, 0.45, 0.75]:
        lgd_label = f"LGD = {lgd*100:.0f}% ({'Secured' if lgd==0.35 else 'Unsecured' if lgd==0.45 else 'Subordinated'})"
        for pd_val in pd_range:
            # Corporate R formula without SME adjustment
            exponent = -50.0 * pd_val
            factor = (1.0 - np.exp(exponent)) / (1.0 - np.exp(-50.0))
            R = 0.12 * factor + 0.24 * (1.0 - factor)
            # default M = 2.5y
            K = basel_capital_requirement_k(pd_val, lgd, R, 2.5)
            rw = K * 12.5 * 100 # Risk weight percentage
            rw_curve_data.append({
                'PD': pd_val * 100,
                'Risk Weight %': rw,
                'LGD Class': lgd_label
            })
            
    rw_df = pd.DataFrame(rw_curve_data)
    fig_nonlin = px.line(
        rw_df,
        x='PD',
        y='Risk Weight %',
        color='LGD Class',
        color_discrete_sequence=['#10B981', '#2563EB', '#EF4444'],
        title="Basel III Risk-Weight (%) vs. Client Probability of Default (PD) and Seniority (LGD)",
        labels={'PD': 'Client Probability of Default (PD %)', 'Risk Weight %': 'Basel Risk Weight (%)'}
    )
    fig_nonlin.update_layout(
        xaxis_type="log",
        xaxis=dict(tickformat=".3f"),
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_nonlin, use_container_width=True)
    st.info("💡 **Simplifying the Basel IRB Curve:** This visualizes the exponential relationship between client default probability (PD) and the required Risk Weight. The 'cliff-effect' is clear: as PD moves past 1%, capital requirements surge. Subordinated claims (high LGD) push the entire curve upward, forcing significantly more capital backing.")

# --- Footer ---
st.markdown("---")
footer_html = """
<div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #E2E8F0; font-family: 'Outfit', sans-serif;">
    <p style="margin: 0; color: #64748B; font-size: 0.85rem; font-weight: 500; letter-spacing: 0.5px;">
        BASEL III CREDIT & TREASURY RISK ENGINE
    </p>
    <p style="margin: 5px 0 15px 0; color: #0F172A; font-size: 1.05rem; font-weight: 600;">
        Made by Saransh Nijhawan
    </p>
    <div style="display: flex; justify-content: center; gap: 15px; align-items: center;">
        <a href="https://www.linkedin.com/in/saransh-nijhawan8142/" target="_blank" style="text-decoration: none; display: inline-flex; align-items: center;">
            <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white&labelColor=0A66C2" alt="LinkedIn" style="border-radius: 4px; height: 26px;">
        </a>
        <a href="mailto:nijhawansaransh2005@gmail.com" style="text-decoration: none; display: inline-flex; align-items: center;">
            <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat&logo=gmail&logoColor=white&labelColor=EA4335" alt="Gmail" style="border-radius: 4px; height: 26px;">
        </a>
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
