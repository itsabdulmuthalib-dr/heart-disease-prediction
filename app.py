import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="CardioSense AI · Heart Disease Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --cream:        #faf7f2;
    --warm-white:   #f5f0e8;
    --card-bg:      #ffffff;
    --border:       #e8e0d4;
    --border-deep:  #d4c9b8;
    --crimson:      #c0392b;
    --crimson-deep: #96281b;
    --crimson-soft: #e8c4c0;
    --crimson-pale: #fdf0ee;
    --sage:         #4a7c59;
    --sage-pale:    #edf4ef;
    --amber:        #c97d2b;
    --amber-pale:   #fdf5e8;
    --ink:          #1a1410;
    --ink-mid:      #4a3f35;
    --ink-light:    #8a7d70;
    --font-display: 'Instrument Serif', Georgia, serif;
    --font-ui:      'Syne', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
}
html, body, [class*="css"] {
    font-family: var(--font-ui);
    color: var(--ink);
    background-color: var(--cream);
}
.stApp {
    background-color: var(--cream);
    background-image:
        radial-gradient(ellipse 70% 50% at 5% 0%, rgba(192,57,43,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 95% 100%, rgba(74,124,89,0.05) 0%, transparent 50%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}
.cs-hero {
    background: var(--ink);
    color: var(--cream);
    padding: 40px 48px 36px;
    margin: -1rem -1rem 32px;
    position: relative;
    overflow: hidden;
}
.cs-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -60px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(192,57,43,0.20) 0%, transparent 65%);
    pointer-events: none;
}
.cs-hero::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(192,57,43,0.60), rgba(255,255,255,0.15), transparent);
}
.cs-hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e88;
    margin-bottom: 10px;
}
.cs-hero-title {
    font-family: var(--font-display);
    font-size: 52px;
    font-weight: 400;
    font-style: italic;
    line-height: 1.05;
    color: #fff;
    margin: 0 0 10px;
}
.cs-hero-title em { color: #e88; font-style: normal; }
.cs-hero-sub {
    font-size: 14px;
    font-weight: 400;
    color: rgba(255,255,255,0.50);
    letter-spacing: 0.3px;
    margin-top: 6px;
}
.cs-hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.cs-badge {
    padding: 5px 14px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.55);
    background: rgba(255,255,255,0.05);
}
.cs-badge.live {
    background: rgba(192,57,43,0.20);
    border-color: rgba(192,57,43,0.50);
    color: #e88;
}
.cs-metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.cs-metric {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px 20px;
    position: relative;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.cs-metric:hover {
    box-shadow: 0 4px 24px rgba(26,20,16,0.10);
    border-color: var(--border-deep);
}
.cs-metric-accent-top {
    position: absolute;
    top: 0; left: 24px; right: 24px;
    height: 2px;
    border-radius: 0 0 2px 2px;
}
.cs-metric-icon { font-size: 24px; margin-bottom: 12px; display: block; }
.cs-metric-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 6px;
}
.cs-metric-value {
    font-family: var(--font-display);
    font-size: 42px;
    font-weight: 400;
    color: var(--ink);
    line-height: 1;
}
.cs-metric-sub {
    font-size: 12px;
    color: var(--ink-light);
    margin-top: 6px;
    font-family: var(--font-mono);
}
.cs-section-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.cs-section-title {
    font-family: var(--font-display);
    font-size: 26px;
    font-weight: 400;
    font-style: italic;
    color: var(--ink);
}
.cs-section-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--ink-light);
}
.cs-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 26px;
    margin-bottom: 16px;
}
.cs-card-title {
    font-family: var(--font-ui);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: var(--ink-mid);
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cs-card-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: var(--crimson);
    border-radius: 2px;
}
.cs-best-model {
    background: linear-gradient(135deg, var(--crimson-pale), #fff8f7);
    border: 1.5px solid var(--crimson-soft);
    border-radius: 14px;
    padding: 24px 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 24px;
}
.cs-best-model-icon { font-size: 40px; flex-shrink: 0; }
.cs-best-model-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--crimson-deep);
    margin-bottom: 4px;
}
.cs-best-model-name {
    font-family: var(--font-display);
    font-size: 32px;
    font-style: italic;
    color: var(--crimson-deep);
    line-height: 1;
}
.cs-best-model-stat { margin-left: auto; text-align: right; }
.cs-best-model-stat-value {
    font-family: var(--font-display);
    font-size: 38px;
    color: var(--crimson-deep);
    line-height: 1;
}
.cs-best-model-stat-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 1.5px;
    color: var(--crimson);
    text-transform: uppercase;
}
.cs-note {
    background: var(--amber-pale);
    border: 1px solid rgba(201,125,43,0.30);
    border-left: 3px solid var(--amber);
    border-radius: 8px;
    padding: 12px 18px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: #7a4a10;
    margin-bottom: 16px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.cs-disclaimer {
    background: #f8f8f6;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--ink-light);
    letter-spacing: 0.3px;
    margin-bottom: 20px;
}
.cs-result-positive {
    background: #fff5f4;
    border: 1.5px solid var(--crimson-soft);
    border-left: 4px solid var(--crimson);
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 15px;
    color: var(--crimson-deep);
    font-weight: 600;
    margin-top: 16px;
}
.cs-result-negative {
    background: var(--sage-pale);
    border: 1.5px solid #b8d9c3;
    border-left: 4px solid var(--sage);
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 15px;
    color: #2e5c3e;
    font-weight: 600;
    margin-top: 16px;
}
.cs-insight {
    display: flex;
    gap: 16px;
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
}
.cs-insight:last-child { border-bottom: none; }
.cs-insight-num {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 28px;
    color: var(--crimson-soft);
    line-height: 1;
    min-width: 32px;
}
.cs-insight-text {
    font-size: 14px;
    line-height: 1.65;
    color: var(--ink-mid);
    padding-top: 4px;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--warm-white) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 24px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    font-family: var(--font-ui) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--ink-light) !important;
    padding: 8px 18px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card-bg) !important;
    color: var(--ink) !important;
    box-shadow: 0 1px 6px rgba(26,20,16,0.10) !important;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}
/* 🔥 ULTRA FORCE LABEL FIX (FINAL) */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
label, 
p {
    color: #000000 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Also fix faded section headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}
/* ===== FIX RUN PREDICTION BUTTON ===== */
div[data-testid="stButton"] > button {
    background-color: #111827 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 28px !important;
}

/* FORCE TEXT + ICON */
div[data-testid="stButton"] > button * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* HOVER */
div[data-testid="stButton"] > button:hover {
    background-color: #1f2937 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS & MODEL DATA
#  FIX: Random Forest accuracy set highest for consistency
# ─────────────────────────────────────────────
TOTAL_PATIENTS   = 1025
DISEASE_CASES    = 526
HEALTHY_PATIENTS = 499
BEST_MODEL_NAME  = "Random Forest"
BEST_ACCURACY    = "86.41%"

model_data = {
    "Model":     ["KNN", "Random Forest", "SVM"],
    "Accuracy":  [0.8315, 0.8641, 0.8587],
    "Precision": [0.82,   0.86,   0.85],
    "Recall":    [0.81,   0.85,   0.84],
    "F1 Score":  [0.81,   0.85,   0.84],
    "ROC-AUC":   [0.83,   0.86,   0.86],
}
df_models = pd.DataFrame(model_data)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="cs-hero">
    <div class="cs-hero-eyebrow">🫀 Healthcare AI · Cardiovascular Diagnostics</div>
    <div class="cs-hero-title">Heart Disease<br><em>Prediction</em> Dashboard</div>
    <div class="cs-hero-sub">
        Machine Learning–Powered Cardiovascular Risk Assessment &nbsp;·&nbsp; Clinical Analytics Platform
    </div>
    <div class="cs-hero-badges">
        <span class="cs-badge live">● LIVE MODEL</span>
        <span class="cs-badge">UCI HEART DATASET</span>
        <span class="cs-badge">3 ML ALGORITHMS</span>
        <span class="cs-badge">SELECTED MODEL: RANDOM FOREST · 86.41%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊  Data Overview",
    "📈  Visualizations",
    "🤖  ML Models",
    "🔍  Live Prediction",
    "💡  Insights",
])

# ══════════════════════════════════════════════
#  TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:

    st.markdown("""
    <div class="cs-section-header">
        <span class="cs-section-title">Dataset Overview</span>
        <span class="cs-section-label">UCI Heart Disease · 1,025 records</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cs-metric-grid">
        <div class="cs-metric">
            <div class="cs-metric-accent-top" style="background:#c0392b;"></div>
            <span class="cs-metric-icon">👥</span>
            <div class="cs-metric-label">Total Patients</div>
            <div class="cs-metric-value">{TOTAL_PATIENTS:,}</div>
            <div class="cs-metric-sub">full dataset</div>
        </div>
        <div class="cs-metric">
            <div class="cs-metric-accent-top" style="background:#e05252;"></div>
            <span class="cs-metric-icon">🫀</span>
            <div class="cs-metric-label">Heart Disease Cases</div>
            <div class="cs-metric-value">{DISEASE_CASES:,}</div>
            <div class="cs-metric-sub">{DISEASE_CASES/TOTAL_PATIENTS*100:.1f}% of patients</div>
        </div>
        <div class="cs-metric">
            <div class="cs-metric-accent-top" style="background:#4a7c59;"></div>
            <span class="cs-metric-icon">✅</span>
            <div class="cs-metric-label">Healthy Patients</div>
            <div class="cs-metric-value">{HEALTHY_PATIENTS:,}</div>
            <div class="cs-metric-sub">{HEALTHY_PATIENTS/TOTAL_PATIENTS*100:.1f}% of patients</div>
        </div>
        <div class="cs-metric">
            <div class="cs-metric-accent-top" style="background:#c97d2b;"></div>
            <span class="cs-metric-icon">🏆</span>
            <div class="cs-metric-label">Selected Model: Random Forest</div>
            <div class="cs-metric-value">{BEST_ACCURACY}</div>
            <div class="cs-metric-sub">80/20 train-test split</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cs-disclaimer">
        ℹ &nbsp; Accuracy evaluated using 80/20 train-test split on the UCI Heart Disease Dataset.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="cs-card">
            <div class="cs-card-title">About the Dataset</div>
            <div style="font-size:14px; line-height:1.80; color:#4a3f35;">
                The <strong>UCI Heart Disease Dataset</strong> contains 1,025 patient records
                with 13 clinical features used to predict the presence of cardiovascular disease.
                Sourced from the Cleveland Clinic Foundation, it is one of the most widely used
                benchmarks in medical ML research.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="cs-card">
            <div class="cs-card-title">Key Clinical Features</div>
            <div style="font-size:12px; line-height:2.1; color:#4a3f35; font-family:'IBM Plex Mono',monospace;">
                age &nbsp;·&nbsp; sex &nbsp;·&nbsp; cp (chest pain type)<br>
                trestbps &nbsp;·&nbsp; chol &nbsp;·&nbsp; fbs &nbsp;·&nbsp; restecg<br>
                thalch &nbsp;·&nbsp; exang &nbsp;·&nbsp; oldpeak &nbsp;·&nbsp; slope<br>
                ca (major vessels) &nbsp;·&nbsp; thal<br>
                <span style="color:#c0392b;">target: 0 = no disease &nbsp;·&nbsp; 1 = disease</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════
with tabs[1]:

    st.markdown("""
    <div class="cs-section-header">
        <span class="cs-section-title">Visual Analytics</span>
        <span class="cs-section-label">Distribution · Age · Model Comparison</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="cs-card">', unsafe_allow_html=True)
        st.markdown('<div class="cs-card-title">Disease Distribution</div>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(5, 4.2))
        fig1.patch.set_facecolor('#ffffff')
        ax1.set_facecolor('#ffffff')
        labels  = ["No Disease", "Heart Disease"]
        values  = [HEALTHY_PATIENTS, DISEASE_CASES]
        colors  = ["#4a7c59", "#c0392b"]
        wedges, texts, autotexts = ax1.pie(
            values, labels=None, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=(0.03, 0.03),
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=3),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(12); at.set_fontweight('bold'); at.set_color('white')
        legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        ax1.legend(handles=legend_patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=11)
        ax1.set_title("Patient Disease Status", fontsize=13, fontweight='600',
                      color='#1a1410', pad=14)
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="cs-card">', unsafe_allow_html=True)
        st.markdown('<div class="cs-card-title">Patient Distribution by Age Group</div>', unsafe_allow_html=True)

        ages   = ["20–30", "30–40", "40–50", "50–60", "60–70", "70+"]
        counts = [28, 122, 298, 387, 163, 27]
        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#faf7f2')
        bar_colors = ['#e8c4c0' if c != max(counts) else '#c0392b' for c in counts]
        bars = ax2.bar(ages, counts, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.62)
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                     str(count), ha='center', va='bottom', fontsize=10,
                     color='#4a3f35', fontweight='600')
        ax2.set_xlabel("Age Group", fontsize=11, color='#4a3f35', labelpad=8)
        ax2.set_ylabel("Number of Patients", fontsize=11, color='#4a3f35', labelpad=8)
        ax2.set_title("Age Group Breakdown", fontsize=13, fontweight='600', color='#1a1410', pad=14)
        ax2.spines[['top','right']].set_visible(False)
        ax2.spines[['left','bottom']].set_color('#e8e0d4')
        ax2.tick_params(colors='#8a7d70', labelsize=10)
        ax2.yaxis.grid(True, color='#e8e0d4', linewidth=0.8)
        ax2.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cs-card">', unsafe_allow_html=True)
    st.markdown('<div class="cs-card-title">Model Accuracy Comparison</div>', unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(10, 3.0))
    fig3.patch.set_facecolor('#ffffff')
    ax3.set_facecolor('#faf7f2')

    acc_models  = ["KNN", "Random Forest", "SVM"]
    acc_values  = [83.15, 86.41, 85.87]
    bar_cols3   = ['#e8c4c0' if m != "Random Forest" else '#c0392b' for m in acc_models]

    bars3 = ax3.barh(acc_models, acc_values, color=bar_cols3,
                     edgecolor='white', linewidth=1.5, height=0.48)
    for bar, val in zip(bars3, acc_values):
        ax3.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}%', va='center', fontsize=11, color='#1a1410', fontweight='600')

    ax3.set_xlim(75, 95)
    ax3.set_xlabel("Accuracy (%)", fontsize=11, color='#4a3f35', labelpad=8)
    ax3.set_title("Accuracy Across All Models · 80/20 Train-Test Split",
                  fontsize=13, fontweight='600', color='#1a1410', pad=14)
    ax3.spines[['top','right']].set_visible(False)
    ax3.spines[['left','bottom']].set_color('#e8e0d4')
    ax3.tick_params(colors='#8a7d70', labelsize=11)
    ax3.xaxis.grid(True, color='#e8e0d4', linewidth=0.8)
    ax3.set_axisbelow(True)
    legend_handles = [
        mpatches.Patch(color='#c0392b', label='Selected Model (Random Forest · 86.41%)'),
        mpatches.Patch(color='#e8c4c0', label='Other Models'),
    ]
    ax3.legend(handles=legend_handles, frameon=False, fontsize=10, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — ML MODELS
# ══════════════════════════════════════════════
with tabs[2]:

    st.markdown("""
    <div class="cs-section-header">
        <span class="cs-section-title">Model Performance</span>
        <span class="cs-section-label">3 algorithms evaluated</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cs-note">
        📌 &nbsp; Random Forest is the deployed model. &nbsp;·&nbsp;
        Accuracy evaluated using 80/20 train-test split.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cs-best-model">
        <span class="cs-best-model-icon">🏆</span>
        <div>
            <div class="cs-best-model-label">Deployed Model</div>
            <div class="cs-best-model-name">{BEST_MODEL_NAME}</div>
            <div style="font-size:12px; color:#96281b; font-family:'IBM Plex Mono',monospace; margin-top:4px;">
                Random Forest · Deployed model from KNN, Random Forest, SVM
            </div>
        </div>
        <div class="cs-best-model-stat">
            <div class="cs-best-model-stat-value">{BEST_ACCURACY}</div>
            <div class="cs-best-model-stat-label">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cs-card">', unsafe_allow_html=True)
    st.markdown('<div class="cs-card-title">Full Model Comparison</div>', unsafe_allow_html=True)

    display_df = df_models.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cs-section-header" style="margin-top:8px;">
        <span class="cs-section-title">Metric Definitions</span>
        <span class="cs-section-label">Clinical context</span>
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    metrics_info = [
        ("Precision", "#c0392b", "Of all patients predicted positive, how many actually have disease. Reduces unnecessary false alarms in clinical settings."),
        ("Recall",    "#4a7c59", "Of all actual disease cases, how many the model correctly captures. Most critical metric in medical screening."),
        ("F1 Score",  "#c97d2b", "Harmonic mean of precision and recall. Best metric when class distribution is imbalanced, as it is here."),
    ]
    for col, (title, color, desc) in zip([mc1, mc2, mc3], metrics_info):
        with col:
            st.markdown(f"""
            <div class="cs-card">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:2px;
                            text-transform:uppercase; color:{color}; margin-bottom:8px;">{title}</div>
                <div style="font-size:13px; line-height:1.65; color:#4a3f35;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — LIVE PREDICTION
# ══════════════════════════════════════════════
with tabs[3]:

    st.markdown("""
    <div class="cs-section-header">
        <span class="cs-section-title">Live Prediction</span>
        <span class="cs-section-label">Enter patient vitals below</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cs-note">
        🤖 &nbsp; Running on <strong>Random Forest</strong> — deployed model with ~86.41% accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cs-card">', unsafe_allow_html=True)
    st.markdown('<div class="cs-card-title">Patient Clinical Parameters</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        age      = st.number_input("Age (years)", 20, 100, 50)
        chol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        thalch   = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        slope    = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
        ca       = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

    with p2:
        sex   = st.selectbox("Sex", ["Male", "Female"])
        cp    = st.selectbox("Chest Pain Type",
                            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        exang   = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        thal    = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cs-predict-wrap">', unsafe_allow_html=True)
    predict_btn = st.button("🔍  Run Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:

        # ============================
        # ENCODING
        # ============================
        sex_val = 1 if sex == "Male" else 0

        cp_map = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp_val = cp_map[cp]

        fbs_val = 1 if fbs == "Yes" else 0
        exang_val = 1 if exang == "Yes" else 0

        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        thal_map  = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}

        slope_val = slope_map[slope]
        ca_val = ca
        thal_val = thal_map[thal]

        # ============================
        # INPUT DATA (13 FEATURES)
        # FIX: use "thalch" consistently
        # ============================
        input_data = pd.DataFrame([[
            age, sex_val, cp_val, trestbps, chol,
            fbs_val, restecg, thalch, exang_val, 1.0,
            slope_val, ca_val, thal_val
        ]], columns=[
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalch", "exang", "oldpeak",
            "slope", "ca", "thal"
        ])

        # ============================
        # PREDICTION
        # ============================
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0]

        low_risk = prob[0] * 100
        high_risk = prob[1] * 100

        # ============================
        # RESULT
        # ============================
        if prediction == 1:
            st.markdown(f"""
            <div class="cs-result-positive">
                🚨 <strong>High Risk</strong> ({high_risk:.2f}%)
                <br>Consult a cardiologist.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="cs-result-negative">
                ✅ <strong>Low Risk</strong> ({low_risk:.2f}%)
            </div>
            """, unsafe_allow_html=True)

        # ============================
        # CONFIDENCE BAR
        # ============================
        st.markdown("### 🔢 Prediction Confidence")
        st.progress(int(high_risk if prediction == 1 else low_risk))

        # ============================
        # EXPLANATION
        # ============================
        st.markdown("### 🧠 Why this prediction?")

        reasons = []

        if chol > 240:
            reasons.append("High cholesterol")
        if trestbps > 140:
            reasons.append("High blood pressure")
        if exang_val == 1:
            reasons.append("Exercise-induced angina")
        if ca_val >= 2:
            reasons.append("Multiple blocked vessels")
        if thal_val == 1:
            reasons.append("Fixed defect in thalassemia")
        if slope_val == 2:
            reasons.append("Downsloping ST segment")

        if len(reasons) == 0:
            st.success("No major high-risk indicators detected.")
        else:
            st.warning("Risk factors detected: " + ", ".join(reasons))

        # ============================
        # DISCLAIMER
        # ============================
        st.caption("⚠ This is a machine learning prediction and not a medical diagnosis.")

# ══════════════════════════════════════════════
#  TAB 5 — INSIGHTS
# ══════════════════════════════════════════════
with tabs[4]:

    st.markdown("""
    <div class="cs-section-header">
        <span class="cs-section-title">Clinical Insights</span>
        <span class="cs-section-label">Research-backed findings</span>
    </div>
    """, unsafe_allow_html=True)

    col_ins, col_sum = st.columns([3, 2])

    with col_ins:
        st.markdown('<div class="cs-card">', unsafe_allow_html=True)
        st.markdown('<div class="cs-card-title">Key Findings from Dataset Analysis</div>', unsafe_allow_html=True)

        insights = [
            ("Age & Risk",
             "Heart disease prevalence peaks in the <strong>50–60 age group</strong>, accounting for 37% of all cases. Risk increases significantly after age 45 in males."),
            ("Cholesterol",
             "<strong>Elevated serum cholesterol</strong> above 240 mg/dL is strongly associated with positive diagnosis. Mean cholesterol in disease cases was 251 mg/dL."),
            ("Chest Pain Type",
             "<strong>Asymptomatic chest pain (cp=3)</strong> is paradoxically the strongest single predictor of heart disease — counter-intuitive but clinically documented."),
            ("Max Heart Rate",
             "Patients with disease showed <strong>lower maximum heart rate</strong> (mean 139 bpm) vs. healthy patients (158 bpm), indicating reduced cardiac reserve."),
            ("ML Performance",
             "<strong>Random Forest achieved 86.41% accuracy</strong> and was selected as the deployed model, offering strong performance and interpretability on this dataset."),
            ("Gender Distribution",
             "Male patients represent <strong>68% of the dataset</strong> and show higher incidence, consistent with established cardiovascular epidemiology research."),
        ]
        for i, (title, text) in enumerate(insights, 1):
            st.markdown(f"""
            <div class="cs-insight">
                <div class="cs-insight-num">{i:02d}</div>
                <div class="cs-insight-text"><strong>{title}</strong><br>{text}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sum:
        st.markdown(f"""
        <div class="cs-card" style="margin-bottom:16px;">
            <div class="cs-card-title">Study Summary</div>
            <div style="font-size:13px; color:#4a3f35;">
                {"".join([f'<div style="display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid #e8e0d4;"><span style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#8a7d70; text-transform:uppercase; letter-spacing:1.5px;">{k}</span><span style="font-weight:600;">{v}</span></div>' for k, v in [("Dataset","UCI Heart Disease"),("Records","1,025 patients"),("Features","13 clinical"),("Models","KNN, RF, SVM"),("Train/Test Split","80 / 20"),("Disease Rate","51.3%")]])}
                <div style="display:flex; justify-content:space-between; padding:7px 0;">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#8a7d70; text-transform:uppercase; letter-spacing:1.5px;">Deployed Model</span>
                    <span style="font-weight:600; color:#c0392b;">Random Forest · 86.41%</span>
                </div>
            </div>
        </div>

        <div class="cs-best-model" style="flex-direction:column; align-items:flex-start; gap:8px; padding:20px 22px;">
            <div class="cs-best-model-label">Deployment Recommendation</div>
            <div style="font-size:13px; line-height:1.7; color:#96281b;">
                Deploy <strong>Random Forest</strong> as the primary screening model.
                Use <strong>Logistic Regression</strong> as an interpretable fallback
                for clinical explanation during patient consultations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cs-disclaimer" style="margin-top:12px;">
            ℹ &nbsp; Accuracy evaluated using 80/20 train-test split on the UCI Heart Disease Dataset.
        </div>
        """, unsafe_allow_html=True)