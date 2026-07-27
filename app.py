# ============================================================
# ⚡ POWER PLANT OUTPUT PREDICTOR - STREAMLIT APP
# ============================================================
# Run with:  streamlit run app.py
# Model:     Gradient Boosting Regressor (best from pipeline)
# Target:    PE — Net hourly electrical energy output (MW)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Power Plant Output Predictor",
                   page_icon="⚡", layout="wide")

# ============================================================
# THEME — industrial control-room styling
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

/* ---- layered ambient background: blueprint grid + energy glows ---- */
.stApp {
    background:
        radial-gradient(1100px 520px at 85% -10%, rgba(255,176,46,.09), transparent 60%),
        radial-gradient(900px 500px at 5% 110%, rgba(67,217,217,.07), transparent 60%),
        linear-gradient(rgba(148,180,226,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,180,226,.035) 1px, transparent 1px),
        #0c121d;
    background-size: auto, auto, 44px 44px, 44px 44px, auto;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: #c8d6e8; }

/* ---- typography ---- */
.plant-title {
    font-family: 'Chakra Petch', sans-serif; font-size: 2.6rem; font-weight: 700;
    letter-spacing: .5px; color: #f2f7ff; margin-bottom: 2px; line-height: 1.1;
}
.plant-title .amber { color: #ffb02e; text-shadow: 0 0 22px rgba(255,176,46,.45); }
.sub-title { color: #7d92ad; margin-top: -6px; margin-bottom: 18px; font-size: 1.02rem; }
h2, h3 { font-family: 'Chakra Petch', sans-serif; color: #e8f0fc !important; letter-spacing: .3px; }
.subhead {
    font-family: 'Chakra Petch', sans-serif; font-size: 1.15rem; font-weight: 600;
    color: #e8f0fc; border-left: 4px solid #ffb02e; padding-left: 12px; margin: 26px 0 10px 0;
}

/* ---- status strip chips ---- */
.chip-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0 24px 0; }
.chip {
    font-family: 'IBM Plex Mono', monospace; font-size: .82rem;
    background: #111a29; border: 1px solid #26374f; border-radius: 8px;
    padding: 10px 16px; color: #8fa3bd; transition: all .22s ease;
}
.chip b { color: #ffb02e; font-size: 1rem; display: block; margin-top: 2px; }
.chip:hover { transform: translateY(-3px); border-color: #ffb02e; box-shadow: 0 8px 20px rgba(0,0,0,.35); }
.chip .cyan { color: #43d9d9; }
.live-dot {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    background: #3ddc84; margin-right: 7px;
    box-shadow: 0 0 0 0 rgba(61,220,132,.6); animation: pulse 1.8s infinite;
}
@keyframes pulse { to { box-shadow: 0 0 0 11px rgba(61,220,132,0); } }

/* ---- panels ---- */
.panel {
    background: #101a2a; border: 1px solid #24344e; border-radius: 10px;
    padding: 22px; transition: border-color .25s ease, box-shadow .25s ease;
}
.panel:hover { border-color: #3a5378; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.panel-label {
    font-family: 'Chakra Petch', sans-serif; font-size: .8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 2px; color: #5f7896; margin-bottom: 14px;
}

/* ---- prediction gauge ---- */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; padding: 10px 0 4px 0; }
.gauge {
    width: 270px; height: 135px; border-radius: 270px 270px 0 0; position: relative; overflow: hidden;
    box-shadow: 0 0 34px rgba(255,176,46,.12);
}
.gauge::after {
    content: ''; position: absolute; left: 30px; right: 30px; top: 30px; bottom: -32px;
    background: #101a2a; border-radius: 999px 999px 0 0; border-top: 1px solid #24344e;
}
.gauge-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 2.5rem; font-weight: 700;
    color: #ffb02e; margin-top: -38px; z-index: 2; text-shadow: 0 0 24px rgba(255,176,46,.4);
}
.gauge-unit { font-family: 'IBM Plex Mono', monospace; color: #5f7896; font-size: .85rem; z-index: 2; }
.gauge-range { display: flex; justify-content: space-between; width: 270px;
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: #5f7896; margin-top: 6px; }

/* ---- feature importance bars ---- */
.fi-row { display: flex; align-items: center; gap: 12px; margin: 11px 0; }
.fi-label { font-family: 'IBM Plex Mono', monospace; width: 42px; color: #8fa3bd; font-size: .85rem; }
.fi-track { flex: 1; height: 14px; background: #16233a; border-radius: 7px; overflow: hidden; }
.fi-fill { height: 100%; border-radius: 7px; background: linear-gradient(90deg, #ff8c1a, #ffb02e);
    transition: width .6s ease, filter .2s ease; }
.fi-row:hover .fi-fill { filter: brightness(1.25); }
.fi-pct { font-family: 'IBM Plex Mono', monospace; width: 46px; text-align: right;
    color: #ffb02e; font-size: .82rem; }

/* ---- widgets ---- */
.stButton > button {
    background: #ffb02e; color: #0c121d; font-family: 'Chakra Petch', sans-serif;
    font-weight: 700; font-size: 1rem; border: none; border-radius: 8px;
    padding: .65rem 1.5rem; transition: all .2s ease; width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); background: #ffc258;
    box-shadow: 0 8px 22px rgba(255,176,46,.35); color: #0c121d; }
[data-baseweb="slider"] > div > div > div { background-color: #ffb02e !important; }
[data-baseweb="slider"] > div > div { background-color: rgba(255,176,46,.35) !important; }
.stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }

/* ---- hide streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- dark matplotlib theme (matches app) ----
plt.rcParams.update({
    'figure.facecolor': '#0c121d', 'axes.facecolor': '#0c121d',
    'axes.edgecolor': '#2a3a52', 'axes.labelcolor': '#c8d6e8',
    'text.color': '#c8d6e8', 'xtick.color': '#8fa3bd', 'ytick.color': '#8fa3bd',
    'grid.color': '#1c2940', 'axes.grid': True, 'grid.alpha': .35,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif',
})
AMBER, CYAN = '#ffb02e', '#43d9d9'

# ============================================================
# LOAD DATA + TRAIN MODEL (cached — runs once)
# ============================================================
FEATURE_META = {
    'AT': ('Ambient Temperature', '°C'),
    'V':  ('Exhaust Vacuum', 'cm Hg'),
    'AP': ('Ambient Pressure', 'mbar'),
    'RH': ('Relative Humidity', '%'),
}

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('Power_Plant_Energy_Output.csv')
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop('PE', axis=1)
    y = df['PE']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # holdout R² for the status readout
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))

    # refit on full data for production predictions
    model.fit(X_scaled, y)
    return df, model, scaler, r2

df, model, scaler, R2 = load_data_and_model()
FEATURES = list(FEATURE_META.keys())
PE_MIN, PE_MAX, PE_MEAN = df['PE'].min(), df['PE'].max(), df['PE'].mean()

# ============================================================
# SIDEBAR - NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## ⚡ PP·CONTROL")
    page = st.radio("Navigate", ["🎛️ Prediction", "📊 Visualizations", "📋 Dataset"])
    st.divider()
    st.info(f"**Model:** Gradient Boosting\n\n"
            f"**R² score:** {R2:.4f}\n\n"
            f"**Samples:** {len(df):,}\n\n"
            f"**Features:** {len(FEATURES)}")
    st.caption("AT · V · AP · RH → PE (MW)")

# ============================================================
# SHARED HEADER — plant status strip
# ============================================================
def status_strip():
    st.markdown(
        '<p class="plant-title">⚡ COMBINED CYCLE <span class="amber">POWER PLANT</span></p>'
        '<p class="sub-title">Hourly electrical output predictor — gas turbine sensor telemetry</p>',
        unsafe_allow_html=True)
    st.markdown(f"""
    <div class="chip-row">
        <div class="chip"><span class="live-dot"></span>SYSTEM ONLINE<b>ACTIVE</b></div>
        <div class="chip">SENSOR SAMPLES<b>{len(df):,}</b></div>
        <div class="chip">INPUT CHANNELS<b>{len(FEATURES)}</b></div>
        <div class="chip">MODEL R²<b class="cyan">{R2:.4f}</b></div>
        <div class="chip">OUTPUT RANGE<b>{PE_MIN:.0f}–{PE_MAX:.0f} MW</b></div>
    </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: PREDICTION
# ============================================================
if page == "🎛️ Prediction":
    status_strip()

    col_in, col_out = st.columns([1.1, 1])

    # ---- LEFT: sensor control panel ----
    with col_in:
        st.markdown('<div class="panel"><div class="panel-label">▸ Sensor Input Panel</div>', unsafe_allow_html=True)
        st.caption("Adjust the turbine sensors — the prediction updates live.")

        vals = {}
        for f in FEATURES:
            name, unit = FEATURE_META[f]
            vals[f] = st.slider(
                f"**{f}** — {name} ({unit})",
                float(round(df[f].min(), 1)), float(round(df[f].max(), 1)),
                float(round(df[f].median(), 1)), 0.1)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- RIGHT: live output readout ----
    with col_out:
        input_df = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
        pred = model.predict(scaler.transform(input_df))[0]

        pct = float(np.clip((pred - PE_MIN) / (PE_MAX - PE_MIN), 0, 1))
        deg = pct * 180
        delta = pred - PE_MEAN

        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">▸ Predicted Output</div>
            <div class="gauge-wrap">
                <div class="gauge" style="background: conic-gradient(from 270deg at 50% 100%,
                     #ff8c1a 0deg, #3ddc84 {deg:.1f}deg,
                     rgba(255,255,255,.05) {deg:.1f}deg 180deg, transparent 180deg);"></div>
                <div class="gauge-value">{pred:.2f}</div>
                <div class="gauge-unit">MEGAWATTS</div>
                <div class="gauge-range"><span>{PE_MIN:.0f}</span><span>{PE_MAX:.0f}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

        if delta >= 0:
            st.success(f"▲ **{delta:+.2f} MW** vs. fleet average ({PE_MEAN:.2f} MW) — strong output window")
        else:
            st.warning(f"▼ **{delta:+.2f} MW** vs. fleet average ({PE_MEAN:.2f} MW) — below-average output")

        # ---- feature importance readout ----
        imps = model.feature_importances_
        imps_norm = imps / imps.max() * 100
        rows = "".join(
            f'<div class="fi-row"><span class="fi-label">{f}</span>'
            f'<div class="fi-track"><div class="fi-fill" style="width:{w:.1f}%"></div></div>'
            f'<span class="fi-pct">{v*100:.0f}%</span></div>'
            for f, w, v in zip(FEATURES, imps_norm, imps))
        st.markdown(f'<div class="panel"><div class="panel-label">▸ Model Feature Importance</div>{rows}</div>',
                    unsafe_allow_html=True)

# ============================================================
# PAGE 2: VISUALIZATIONS (same 4 plots as the notebook)
# ============================================================
elif page == "📊 Visualizations":
    status_strip()

    # PLOT 1: Target Distribution
    st.markdown('<div class="subhead">1 · Target Distribution (PE)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['PE'], bins=30, color=AMBER, edgecolor='#0c121d', alpha=.85)
    axes[0].set_title('Power Output Distribution'); axes[0].set_xlabel('PE (MW)'); axes[0].set_ylabel('Frequency')
    axes[0].axvline(PE_MEAN, color=CYAN, linestyle='--', label=f'Mean: {PE_MEAN:.2f}'); axes[0].legend()
    bp = axes[1].boxplot(df['PE'], vert=True, patch_artist=True,
                         boxprops=dict(facecolor='#1b2a44', color=CYAN),
                         medianprops=dict(color=AMBER, linewidth=2),
                         whiskerprops=dict(color='#5a7396'), capprops=dict(color='#5a7396'),
                         flierprops=dict(marker='o', markerfacecolor='#5a7396', markersize=3, alpha=.5))
    axes[1].set_title('Power Output Boxplot'); axes[1].set_ylabel('PE (MW)')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    # PLOT 2: Feature Distributions
    st.markdown('<div class="subhead">2 · Sensor Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for idx, f in enumerate(FEATURES):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df[f], bins=25, color=CYAN, edgecolor='#0c121d', alpha=.8)
        name, unit = FEATURE_META[f]
        ax.set_title(f'{f} — {name}'); ax.set_xlabel(f'{f} ({unit})'); ax.set_ylabel('Frequency')
    plt.suptitle('Feature Distributions', color='#e8f0fc', fontsize=14)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    # PLOT 3: Features vs Target
    st.markdown('<div class="subhead">3 · Sensors vs. Output</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for idx, f in enumerate(FEATURES):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(df[f], df['PE'], alpha=.3, s=10, color=AMBER)
        name, unit = FEATURE_META[f]
        ax.set_title(f'{f} vs PE'); ax.set_xlabel(f'{f} ({unit})'); ax.set_ylabel('PE (MW)')
    plt.suptitle('Features vs Target', color='#e8f0fc', fontsize=14)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    # PLOT 4: Correlation Heatmap
    st.markdown('<div class="subhead">4 · Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=1.5, linecolor='#0c121d',
                cbar_kws={'shrink': .8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

# ============================================================
# PAGE 3: DATASET
# ============================================================
else:
    status_strip()

    st.markdown('<div class="subhead">Dataset Preview</div>', unsafe_allow_html=True)
    st.write(f"Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="subhead">Column Glossary</div>', unsafe_allow_html=True)
    glossary = pd.DataFrame([
        {'Code': 'AT', 'Description': 'Ambient Temperature', 'Unit': '°C',
         'Min': df['AT'].min(), 'Max': df['AT'].max(), 'Mean': round(df['AT'].mean(), 2)},
        {'Code': 'V', 'Description': 'Exhaust Vacuum', 'Unit': 'cm Hg',
         'Min': df['V'].min(), 'Max': df['V'].max(), 'Mean': round(df['V'].mean(), 2)},
        {'Code': 'AP', 'Description': 'Ambient Pressure', 'Unit': 'mbar',
         'Min': df['AP'].min(), 'Max': df['AP'].max(), 'Mean': round(df['AP'].mean(), 2)},
        {'Code': 'RH', 'Description': 'Relative Humidity', 'Unit': '%',
         'Min': df['RH'].min(), 'Max': df['RH'].max(), 'Mean': round(df['RH'].mean(), 2)},
        {'Code': 'PE', 'Description': 'Net Electrical Output (target)', 'Unit': 'MW',
         'Min': df['PE'].min(), 'Max': df['PE'].max(), 'Mean': round(df['PE'].mean(), 2)},
    ])
    st.table(glossary)

    st.markdown('<div class="subhead">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)
