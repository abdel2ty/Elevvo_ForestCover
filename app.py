"""
╔══════════════════════════════════════════════════════════════╗
║     🌲 Forest Cover Type Classifier — Streamlit App         ║
║     Task 3 | Multi-class Classification | ML Internship     ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="🌲 Forest Cover Classifier",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .hero {
        background: linear-gradient(135deg, #081c15 0%, #1b4332 40%, #2d6a4f 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(45, 106, 79, 0.5);
        border: 1px solid #40916c;
    }
    .hero h1 { color: white; font-size: 2.4rem; margin: 0; font-weight: 800; }
    .hero p { color: #95d5b2; margin: 0.5rem 0 0; font-size: 1.05rem; }
    
    .cover-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
        cursor: pointer;
    }
    .cover-card:hover { transform: translateY(-5px); }
    
    .prob-bar-label { color: #d1d5db; font-size: 0.85rem; margin-bottom: 0.2rem; }
    
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        width: 100%;
        font-size: 1.05rem;
        box-shadow: 0 8px 24px rgba(45,106,79,0.5);
        transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

COVER_NAMES = {
    0: 'Spruce/Fir', 1: 'Lodgepole Pine', 2: 'Ponderosa Pine',
    3: 'Cottonwood/Willow', 4: 'Aspen', 5: 'Douglas-fir', 6: 'Krummholz'
}
COVER_ICONS = {0: '🌲', 1: '🌴', 2: '🌳', 3: '🌿', 4: '🍃', 5: '🌾', 6: '⛰️'}
COVER_COLORS = ['#2d6a4f', '#52b788', '#74c69d', '#95d5b2', '#b7e4c7', '#40916c', '#1b4332']
COVER_DESC = {
    0: 'High elevation cold climate zones, typically above 8,000 ft',
    1: 'Most common forest type, broad elevation range',
    2: 'Lower elevation warm zones, fire-dependent ecosystem',
    3: 'Riparian zones near streams and rivers',
    4: 'Mid-elevation mixed forests with aspen groves',
    5: 'Moderate elevation well-drained slopes',
    6: 'Subalpine treeline, harsh wind-exposed ridges'
}

@st.cache_resource
def get_model():
    if os.path.exists('forest_model.pkl'):
        model = joblib.load('forest_model.pkl')
        scaler = joblib.load('forest_scaler.pkl')
        with open('forest_features.json') as f:
            features = json.load(f)
        return model, scaler, features
    else:
        # Demo model
        np.random.seed(42)
        n = 5000
        n_feats = 54
        X = np.random.randn(n, n_feats)
        # Make elevation important
        X[:, 0] = X[:, 0] * 500 + 2500
        y = np.random.randint(0, 7, n)
        # Make it somewhat realistic
        y = ((X[:, 0] - 1800) / 500).astype(int).clip(0, 6)
        
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_sc, y)
        
        features = ['Elevation','Aspect','Slope','H_Dist_Hydro','V_Dist_Hydro',
                    'H_Dist_Roads','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
                    'H_Dist_Fire'] + [f'Wilderness_{i}' for i in range(4)] + [f'Soil_{i}' for i in range(40)]
        return model, scaler, features

model, scaler, feature_names = get_model()

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌲 Forest Cover Type Classifier</h1>
    <p>Predict forest cover type from cartographic & environmental measurements</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔️ Location Parameters")
    st.markdown("---")
    
    st.markdown("#### 🗺️ Topographic Features")
    elevation = st.slider("⛰️ Elevation (meters)", 1800, 3900, 2800)
    aspect = st.slider("🧭 Aspect (degrees)", 0, 360, 180, help="Compass direction slope faces")
    slope = st.slider("📐 Slope (degrees)", 0, 52, 14)
    
    st.markdown("#### 💧 Hydrology")
    h_hydro = st.slider("↔️ Horizontal Distance to Water (m)", 0, 1400, 300)
    v_hydro = st.slider("↕️ Vertical Distance to Water (m)", -150, 600, 30)
    
    st.markdown("#### 🛣️ Infrastructure")
    h_roads = st.slider("🛣️ Distance to Roads (m)", 0, 7000, 1200)
    h_fire = st.slider("🔥 Distance to Fire Points (m)", 0, 7000, 1700)
    
    st.markdown("#### ☀️ Hillshade")
    hs_9am = st.slider("🌅 Hillshade 9am", 0, 254, 212)
    hs_noon = st.slider("☀️ Hillshade Noon", 0, 254, 220)
    hs_3pm = st.slider("🌇 Hillshade 3pm", 0, 254, 142)
    
    st.markdown("#### 🌿 Wilderness Area")
    wilderness = st.selectbox("🏕️ Wilderness Area", 
                               ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
    w_idx = ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"].index(wilderness)
    
    st.markdown("---")
    predict_btn = st.button("🌲 Classify Forest", use_container_width=True)

# ── Build Feature Vector ────────────────────────────────────────
wilderness_arr = [0, 0, 0, 0]
wilderness_arr[w_idx] = 1
soil_arr = [0] * 40
soil_idx = int(elevation / 100) % 40
soil_arr[soil_idx] = 1

base_features = [elevation, aspect, slope, h_hydro, v_hydro, 
                  h_roads, hs_9am, hs_noon, hs_3pm, h_fire]
full_features = base_features + wilderness_arr + soil_arr

X_input = np.array([full_features[:len(feature_names)]])
if X_input.shape[1] < len(feature_names):
    pad = np.zeros((1, len(feature_names) - X_input.shape[1]))
    X_input = np.hstack([X_input, pad])

X_scaled = scaler.transform(X_input[:, :scaler.n_features_in_])
pred_class = int(model.predict(X_scaled)[0])
try:
    probs = model.predict_proba(X_scaled)[0]
except:
    probs = np.zeros(7)
    probs[pred_class] = 1.0

# Ensure probs length matches cover types
if len(probs) < 7:
    probs = np.pad(probs, (0, 7 - len(probs)))

# ── Main ───────────────────────────────────────────────────────
col_pred, col_chart = st.columns([2, 3], gap="large")

with col_pred:
    icon = COVER_ICONS.get(pred_class, '🌲')
    name = COVER_NAMES.get(pred_class, f'Type {pred_class}')
    color = COVER_COLORS[pred_class % len(COVER_COLORS)]
    desc = COVER_DESC.get(pred_class, '')
    confidence = float(probs[pred_class]) * 100
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}44, {color}22); 
                border: 2px solid {color}; border-radius: 20px; padding: 2rem; text-align: center;">
        <div style="font-size: 4rem;">{icon}</div>
        <div style="color: white; font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0;">{name}</div>
        <div style="color: #95d5b2; font-size: 0.9rem; margin-bottom: 1rem;">{desc}</div>
        <div style="background: {color}33; border-radius: 10px; padding: 0.8rem; margin-top: 1rem;">
            <div style="color: #ccc; font-size: 0.8rem;">Confidence</div>
            <div style="color: {color}; font-size: 2rem; font-weight: 800;">{confidence:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🏔️ Elevation Context")
    elev_zones = [
        (1800, 2200, "Low Montane", "#95d5b2"),
        (2200, 2800, "Mid Montane", "#52b788"),
        (2800, 3200, "Upper Montane", "#2d6a4f"),
        (3200, 3600, "Subalpine", "#1b4332"),
        (3600, 3900, "Alpine", "#081c15"),
    ]
    for lo, hi, label, c in elev_zones:
        active = lo <= elevation < hi
        style = f"background: {c}; border: 2px solid white;" if active else f"background: {c}33;"
        st.markdown(f"""
        <div style="{style} border-radius: 8px; padding: 0.4rem 0.8rem; margin: 0.3rem 0; 
                    display: flex; justify-content: space-between; border-radius: 8px;">
            <span style="color: {'white' if active else '#888'};">{'→ ' if active else ''}{label}</span>
            <span style="color: {'white' if active else '#666'};">{lo}-{hi}m</span>
        </div>
        """, unsafe_allow_html=True)

with col_chart:
    st.markdown("#### 📊 Prediction Confidence by Cover Type")
    
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.patch.set_facecolor('#0a0e1a')
    
    # Probability bar chart
    ax1 = axes[0]
    ax1.set_facecolor('#111827')
    names_short = [COVER_NAMES.get(i, f'T{i}')[:14] for i in range(7)]
    bar_colors = ['#FFD93D' if i == pred_class else COVER_COLORS[i] for i in range(7)]
    bars = ax1.barh(names_short, probs, color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Probability', color='#d1d5db')
    ax1.set_title('Class Probabilities', color='white', fontweight='bold')
    ax1.tick_params(colors='#9ca3af')
    for spine in ax1.spines.values(): spine.set_edgecolor('#374151')
    for bar, val in zip(bars, probs):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.1%}', va='center', color='white', fontsize=9)
    ax1.grid(True, axis='x', alpha=0.3, color='#1f2937')
    ax1.invert_yaxis()
    
    # Input features radar
    ax2 = axes[1]
    ax2.set_facecolor('#111827')
    normalized = {
        'Elevation': (elevation - 1800) / (3900 - 1800),
        'Aspect': aspect / 360,
        'Slope': slope / 52,
        'H-Hydro': h_hydro / 1400,
        'V-Hydro': (v_hydro + 150) / 750,
        'H-Roads': h_roads / 7000,
        'Hillshade 9am': hs_9am / 254,
        'Hillshade Noon': hs_noon / 254,
    }
    keys = list(normalized.keys())
    vals = list(normalized.values())
    c_bar = ['#FFD93D' if v > 0.7 else '#52b788' if v > 0.4 else '#FF6B6B' for v in vals]
    ax2.barh(keys, vals, color=c_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_title('Input Feature Intensity (Normalized)', color='white', fontweight='bold')
    ax2.set_xlabel('Normalized Value', color='#d1d5db')
    ax2.tick_params(colors='#9ca3af')
    for spine in ax2.spines.values(): spine.set_edgecolor('#374151')
    ax2.axvline(0.5, color='#374151', linewidth=1, linestyle='--')
    ax2.grid(True, axis='x', alpha=0.3, color='#1f2937')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── All Cover Types Reference ──────────────────────────────────
st.markdown("---")
st.markdown("### 🌿 Forest Cover Type Reference Guide")
cols = st.columns(7)
for i, col in enumerate(cols):
    with col:
        color = COVER_COLORS[i]
        icon = COVER_ICONS[i]
        name = COVER_NAMES[i]
        is_pred = (i == pred_class)
        border = f"border: 2px solid white;" if is_pred else ""
        bg = f"background: {color};" if is_pred else f"background: {color}44;"
        st.markdown(f"""
        <div style="{bg} {border} border-radius: 12px; padding: 0.8rem 0.4rem; text-align: center;">
            <div style="font-size: 1.8rem;">{icon}</div>
            <div style="color: {'white' if is_pred else '#ccc'}; font-size: 0.75rem; 
                        font-weight: {'800' if is_pred else '400'}; margin-top: 0.3rem;">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.85rem; padding: 1.5rem; margin-top: 1rem;">
    🌲 Forest Cover Classifier · Task 3: Multi-class Classification · ML Internship · Built with Streamlit & Scikit-learn/XGBoost
</div>
""", unsafe_allow_html=True)
