import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ForestIQ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = st.query_params.get("page", "classify")

def go_to(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  STYLES — SegmentIQ Glassmorphism System
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-a:        #0D0B1A;
    --bg-b:        #130F24;
    --bg-c:        #180D2E;
    --violet:      #8B5CF6;
    --violet2:     #A78BFA;
    --violet3:     #C4B5FD;
    --violet-dim:  rgba(139,92,246,0.15);
    --violet-brd:  rgba(139,92,246,0.3);
    --pink:        #EC4899;
    --pink-dim:    rgba(236,72,153,0.12);
    --glass-bg:    rgba(255,255,255,0.04);
    --glass-bg2:   rgba(255,255,255,0.07);
    --glass-brd:   rgba(255,255,255,0.1);
    --glass-brd2:  rgba(255,255,255,0.16);
    --text:        #F1EEFF;
    --text2:       #B8AED8;
    --text3:       #7C6FA0;
    --text4:       #3D3560;
    /* Cover type accent colors */
    --c0: #34D399; --c0d: rgba(52,211,153,0.15);  --c0b: rgba(52,211,153,0.3);
    --c1: #60A5FA; --c1d: rgba(96,165,250,0.15);  --c1b: rgba(96,165,250,0.3);
    --c2: #FBBF24; --c2d: rgba(251,191,36,0.15);  --c2b: rgba(251,191,36,0.3);
    --c3: #F97316; --c3d: rgba(249,115,22,0.15);  --c3b: rgba(249,115,22,0.3);
    --c4: #A78BFA; --c4d: rgba(167,139,250,0.15); --c4b: rgba(167,139,250,0.3);
    --c5: #F87171; --c5d: rgba(248,113,113,0.15); --c5b: rgba(248,113,113,0.3);
    --c6: #E879F9; --c6d: rgba(232,121,249,0.15); --c6b: rgba(232,121,249,0.3);
}

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--bg-a) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% -10%, rgba(139,92,246,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(236,72,153,0.1) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(139,92,246,0.05) 0%, transparent 70%) !important;
    background-attachment: fixed !important;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAV ── */
.gnav {
    background: rgba(13,11,26,0.7);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--glass-brd);
    padding: 0 2.5rem;
    display: flex; align-items: stretch;
    min-height: 58px;
    position: sticky; top: 0; z-index: 999;
}
.g-brand {
    display: flex; align-items: center; gap: 13px;
    padding-right: 28px; border-right: 1px solid var(--glass-brd);
    margin-right: 8px; min-width: 200px;
}
.g-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--violet), var(--pink));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 16px rgba(139,92,246,0.5);
    font-size: 16px; color: white; font-weight: 700;
    flex-shrink: 0;
}
.g-wordmark {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.02em;
}
.g-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.52rem; color: var(--text3);
    letter-spacing: 0.08em; margin-top: 1px;
}
.g-nav-links { display: flex; align-items: stretch; flex: 1; padding: 0 0.5rem; }
.g-nav-item {
    display: flex; align-items: center; gap: 7px;
    padding: 0 18px; font-size: 0.79rem; font-weight: 500;
    color: var(--text3); border-bottom: 2px solid transparent;
    text-decoration: none; cursor: pointer; user-select: none;
    transition: color .15s, border-color .15s; letter-spacing: 0.01em;
}
.g-nav-item:hover { color: var(--text2); border-bottom-color: rgba(139,92,246,0.4); }
.g-nav-item.active { color: var(--violet2); border-bottom-color: var(--violet); font-weight: 600; }
.g-nav-pill {
    background: var(--violet-dim); border: 1px solid var(--violet-brd);
    padding: 2px 8px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.52rem; color: var(--violet2); letter-spacing: 0.06em;
}
.g-nav-right {
    display: flex; align-items: center;
    border-left: 1px solid var(--glass-brd); padding-left: 20px; margin-left: 8px;
}
.live-badge {
    display: flex; align-items: center; gap: 6px;
    background: var(--glass-bg); border: 1px solid var(--glass-brd);
    padding: 5px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: var(--text2); letter-spacing: 0.06em;
}
.live-dot {
    width: 6px; height: 6px; background: #34D399; border-radius: 50%;
    box-shadow: 0 0 6px #34D399;
    animation: glow-pulse 2s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%,100% { opacity:1; box-shadow: 0 0 6px #34D399; }
    50%      { opacity:.6; box-shadow: 0 0 12px #34D399; }
}

/* ── PAGE HEADER ── */
.g-page-header {
    padding: 2.4rem 2.5rem 2rem;
    display: flex; align-items: flex-end; justify-content: space-between;
    border-bottom: 1px solid var(--glass-brd);
    background: linear-gradient(180deg, rgba(139,92,246,0.05) 0%, transparent 100%);
}
.g-page-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.04em; line-height: 1.05;
}
.g-page-title span {
    background: linear-gradient(135deg, var(--violet2), var(--pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.g-page-desc { font-size: 0.8rem; font-weight: 300; color: var(--text2); max-width: 340px; text-align: right; line-height: 1.7; }
.g-page-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--text3); margin-top: 4px; text-align: right; letter-spacing: 0.06em; }

/* ── SHELL ── */
.shell { max-width: 1280px; margin: 0 auto; padding: 2rem 2.5rem 4rem; }

/* ── SECTION LABEL ── */
.g-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 500;
    color: var(--text3); text-transform: uppercase; letter-spacing: 0.2em;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 10px;
}
.g-label::after { content:''; flex:1; height:1px; background: var(--glass-brd); }

/* ── GLASS CARD ── */
.gcard {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-brd);
    border-radius: 16px; padding: 1.5rem;
    position: relative; overflow: hidden;
    transition: border-color .2s, box-shadow .2s;
}
.gcard::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.gcard:hover {
    border-color: var(--glass-brd2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(139,92,246,0.08);
}

/* ── CHART CARD LABELS ── */
.ct-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 2px; }
.ct-title { font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text2); margin-bottom: 0.9rem; letter-spacing: -0.02em; }

/* ── KPI CARD ── */
.kpi-g {
    background: var(--glass-bg); backdrop-filter: blur(16px);
    border: 1px solid var(--glass-brd); border-radius: 16px;
    padding: 1.4rem 1.5rem; position: relative; overflow: hidden;
}
.kpi-g::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.kpi-accent { position: absolute; top: 0; left: 0; bottom: 0; width: 3px; border-radius: 16px 0 0 16px; }
.kpi-val { font-family: 'Sora', sans-serif; font-size: 2.6rem; font-weight: 700; color: var(--text); letter-spacing: -0.05em; line-height: 1; margin-top: 4px; }
.kpi-name { font-size: 0.72rem; font-weight: 600; color: var(--text2); margin-top: 7px; }
.kpi-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--text3); margin-top: 3px; line-height: 1.6; }

/* ── PREDICTION RESULT CARD ── */
.result-g {
    background: var(--glass-bg); backdrop-filter: blur(20px);
    border: 1px solid var(--glass-brd); border-radius: 16px;
    padding: 1.75rem 1.5rem; position: relative; overflow: hidden;
}
.result-g::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.result-icon { font-size: 3.5rem; line-height: 1; margin-bottom: 8px; }
.result-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--violet2); text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 4px; }
.result-name { font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em; line-height: 1.1; margin: 6px 0 4px; }
.result-desc { font-size: 0.72rem; color: var(--text3); line-height: 1.5; margin-bottom: 12px; }
.conf-block { border-radius: 12px; padding: 0.9rem 1rem; margin-top: 12px; }
.conf-label { font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 3px; }
.conf-val { font-family: 'Sora', sans-serif; font-size: 2.4rem; font-weight: 700; letter-spacing: -0.05em; line-height: 1; }

/* ── COVER TYPE GRID CARDS ── */
.cover-g {
    background: var(--glass-bg); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-brd); border-radius: 14px; padding: 1rem 0.6rem;
    text-align: center; position: relative; overflow: hidden;
    transition: border-color .2s;
}
.cover-g.selected { border-width: 1px; box-shadow: 0 0 20px rgba(139,92,246,0.15); }
.cover-g::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.cover-icon { font-size: 2rem; margin-bottom: 6px; }
.cover-name { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--text2); line-height: 1.3; }
.cover-name.selected { color: var(--text); font-weight: 600; }

/* ── ELEVATION ZONE STRIPS ── */
.elev-zone {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px; border-radius: 8px; margin-bottom: 4px;
    background: var(--glass-bg); border: 1px solid var(--glass-brd);
    transition: all .15s;
}
.elev-zone.active { background: var(--violet-dim); border-color: var(--violet-brd); }
.elev-name { font-size: 0.72rem; font-weight: 400; color: var(--text3); }
.elev-name.active { color: var(--violet2); font-weight: 600; }
.elev-range { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--text4); }
.elev-range.active { color: var(--violet3); }

/* ── INSIGHT PILLS ── */
.ins-g {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px; border-radius: 10px; margin-bottom: 6px;
    background: var(--glass-bg); border: 1px solid var(--glass-brd);
}
.ins-g.ok   { border-color: rgba(52,211,153,0.3);  background: rgba(52,211,153,0.06); }
.ins-g.warn { border-color: rgba(251,191,36,0.3);  background: rgba(251,191,36,0.06); }
.ins-g.info { border-color: rgba(139,92,246,0.3);  background: rgba(139,92,246,0.07); }
.ins-ico { font-size: 12px; flex-shrink:0; margin-top:1px; }
.ins-t   { font-size: 0.72rem; font-weight: 600; color: var(--text); }
.ins-b   { font-size: 0.65rem; color: var(--text2); margin-top: 1px; line-height: 1.5; }

/* ── INLINE METRICS STRIP ── */
.g-strip {
    display: flex; background: var(--glass-bg); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-brd); border-radius: 12px; overflow: hidden; margin-bottom: 12px;
}
.g-strip-item { flex: 1; padding: 0.9rem 1rem; border-right: 1px solid var(--glass-brd); }
.g-strip-item:last-child { border-right: none; }
.gsi-v { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em; }
.gsi-l { font-family: 'JetBrains Mono', monospace; font-size: 0.56rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }

/* ── SLIDERS / SELECTS ── */
.stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important; font-weight: 400 !important;
    color: var(--text3) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
.stSelectbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important; font-weight: 400 !important;
    color: var(--text3) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
.stSelectbox > div > div {
    background: var(--glass-bg) !important; border: 1px solid var(--glass-brd) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important; font-size: 0.82rem !important;
    backdrop-filter: blur(12px) !important;
}

/* ── FOOTER ── */
.g-footer {
    border-top: 1px solid var(--glass-brd);
    padding: 1.4rem 0 0.5rem; margin-top: 3.5rem;
    display: flex; align-items: center; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    color: var(--text3); letter-spacing: 0.06em;
}

.rule { height: 1px; background: var(--glass-brd); margin: 2rem 0; }
div[data-testid="stHorizontalBlock"] button { display: none !important; }
::-webkit-scrollbar { width: 5px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ── DATA / CONSTANTS ───────────────────────────────────────────
COVER_NAMES  = {0:'Spruce/Fir', 1:'Lodgepole Pine', 2:'Ponderosa Pine',
                3:'Cottonwood/Willow', 4:'Aspen', 5:'Douglas-fir', 6:'Krummholz'}
COVER_ICONS  = {0:'🌲', 1:'🌴', 2:'🌳', 3:'🌿', 4:'🍃', 5:'🌾', 6:'⛰️'}
COVER_DESC   = {
    0:'High elevation cold climate zones, typically above 8,000 ft',
    1:'Most common forest type, broad elevation range',
    2:'Lower elevation warm zones, fire-dependent ecosystem',
    3:'Riparian zones near streams and rivers',
    4:'Mid-elevation mixed forests with aspen groves',
    5:'Moderate elevation well-drained slopes',
    6:'Subalpine treeline, harsh wind-exposed ridges'
}
COVER_COLORS = ['#34D399','#60A5FA','#FBBF24','#F97316','#A78BFA','#F87171','#E879F9']
COVER_DIMS   = [
    'rgba(52,211,153,0.15)','rgba(96,165,250,0.15)','rgba(251,191,36,0.15)',
    'rgba(249,115,22,0.15)','rgba(167,139,250,0.15)','rgba(248,113,113,0.15)','rgba(232,121,249,0.15)'
]
COVER_BRDS   = [
    'rgba(52,211,153,0.3)','rgba(96,165,250,0.3)','rgba(251,191,36,0.3)',
    'rgba(249,115,22,0.3)','rgba(167,139,250,0.3)','rgba(248,113,113,0.3)','rgba(232,121,249,0.3)'
]

# ── MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    if os.path.exists('forest_model.pkl'):
        model  = joblib.load('forest_model.pkl')
        scaler = joblib.load('forest_scaler.pkl')
        with open('forest_features.json') as f:
            features = json.load(f)
        return model, scaler, features

    np.random.seed(42); n = 5000; n_feats = 54
    X    = np.random.randn(n, n_feats)
    X[:,0] = X[:,0] * 500 + 2500
    y    = ((X[:,0] - 1800) / 500).astype(int).clip(0, 6)
    sc   = StandardScaler(); Xs = sc.fit_transform(X)
    mdl  = RandomForestClassifier(n_estimators=50, random_state=42)
    mdl.fit(Xs, y)
    feats = (['Elevation','Aspect','Slope','H_Dist_Hydro','V_Dist_Hydro',
              'H_Dist_Roads','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','H_Dist_Fire']
             + [f'Wilderness_{i}' for i in range(4)] + [f'Soil_{i}' for i in range(40)])
    return mdl, sc, feats

model, scaler, feature_names = get_model()

def CC():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Manrope', color='#7C6FA0', size=10.5)
    )

GRID = 'rgba(255,255,255,0.06)'
TICK = dict(size=9, family='JetBrains Mono', color='#7C6FA0')
AX   = dict(size=9, color='#7C6FA0', family='JetBrains Mono')

# ── NAV ───────────────────────────────────────────────────────
tabs = [("classify","Classify","01"), ("explore","Explore","02"),
        ("reference","Reference","03")]
nav_html = "".join(
    f'<a class="g-nav-item {"active" if page==k else ""}" href="?page={k}" target="_self">'
    f'{lbl} <span class="g-nav-pill">{n}</span></a>'
    for k, lbl, n in tabs
)
st.markdown(f"""
<div class="gnav">
  <div class="g-brand">
    <div class="g-logo">◈</div>
    <div>
      <div class="g-wordmark">ForestIQ</div>
      <div class="g-tagline">RANDOM FOREST · 7 CLASSES</div>
    </div>
  </div>
  <div class="g-nav-links">{nav_html}</div>
  <div class="g-nav-right">
    <div class="live-badge"><span class="live-dot"></span>LIVE CLASSIFIER</div>
  </div>
</div>
""", unsafe_allow_html=True)

_nc = st.columns(len(tabs))
for _c, (_k, _l, _n) in zip(_nc, tabs):
    with _c:
        if st.button(_l, key=f"nav_{_k}"):
            go_to(_k)

# ── PAGE HEADERS ──────────────────────────────────────────────
page_headers = {
    "classify":  ("Forest","Classifier",  "Predict cover type from cartographic inputs", "RandomForest · 54 features · 7 classes"),
    "explore":   ("Feature","Explorer",   "Sensitivity curves · elevation zones",         "8 normalised input dimensions"),
    "reference": ("Cover Type","Guide",   "All 7 forest cover types with descriptions",   "Ecological zones · cartographic ranges"),
}
ht1, ht2, hdesc, hmeta = page_headers[page]
st.markdown(f"""
<div class="g-page-header">
  <div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--violet2);text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;opacity:.8;">{ht1}</div>
    <div class="g-page-title">{ht2}</div>
  </div>
  <div>
    <div class="g-page-desc">{hdesc}</div>
    <div class="g-page-meta">{hmeta}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SHARED INPUTS (inline, no sidebar) ────────────────────────
def input_panel(prefix=""):
    st.markdown('<div class="g-label">Topographic Parameters</div>', unsafe_allow_html=True)
    t1,t2,t3,t4,t5 = st.columns(5)
    with t1: elevation = st.slider("Elevation (m)",    1800, 3900, 2800, key=f"{prefix}elev")
    with t2: aspect    = st.slider("Aspect (°)",          0,  360,  180, key=f"{prefix}asp")
    with t3: slope     = st.slider("Slope (°)",            0,   52,   14, key=f"{prefix}slp")
    with t4: h_hydro   = st.slider("H-Dist Water (m)",    0, 1400,  300, key=f"{prefix}hh")
    with t5: v_hydro   = st.slider("V-Dist Water (m)",  -150,  600,   30, key=f"{prefix}vh")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Environmental Parameters</div>', unsafe_allow_html=True)
    e1,e2,e3,e4,e5,e6 = st.columns(6)
    with e1: h_roads  = st.slider("H-Dist Roads (m)",  0, 7000, 1200, key=f"{prefix}hr")
    with e2: h_fire   = st.slider("H-Dist Fire (m)",   0, 7000, 1700, key=f"{prefix}hf")
    with e3: hs_9am   = st.slider("Hillshade 9am",     0,  254,  212, key=f"{prefix}h9")
    with e4: hs_noon  = st.slider("Hillshade Noon",    0,  254,  220, key=f"{prefix}hn")
    with e5: hs_3pm   = st.slider("Hillshade 3pm",     0,  254,  142, key=f"{prefix}h3")
    with e6: wilderness = st.selectbox("Wilderness Area",
        ["Rawah","Neota","Comanche Peak","Cache la Poudre"], key=f"{prefix}wld")
    st.markdown('</div>', unsafe_allow_html=True)
    return elevation, aspect, slope, h_hydro, v_hydro, h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness

def build_and_predict(elevation, aspect, slope, h_hydro, v_hydro,
                       h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness):
    w_idx = ["Rawah","Neota","Comanche Peak","Cache la Poudre"].index(wilderness)
    w_arr = [0,0,0,0]; w_arr[w_idx] = 1
    s_arr = [0]*40; s_arr[int(elevation/100)%40] = 1
    base  = [elevation,aspect,slope,h_hydro,v_hydro,h_roads,hs_9am,hs_noon,hs_3pm,h_fire]
    full  = base + w_arr + s_arr
    X     = np.array([full[:len(feature_names)]])
    if X.shape[1] < len(feature_names):
        X = np.hstack([X, np.zeros((1, len(feature_names)-X.shape[1]))])
    Xs    = scaler.transform(X[:, :scaler.n_features_in_])
    pred  = int(model.predict(Xs)[0])
    try:    probs = model.predict_proba(Xs)[0]
    except: probs = np.zeros(7); probs[pred] = 1.0
    if len(probs) < 7: probs = np.pad(probs, (0, 7-len(probs)))
    return pred, probs


# ══════════════════════════════════════════════════════════════
#  01 — CLASSIFY
# ══════════════════════════════════════════════════════════════
if page == "classify":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    elevation, aspect, slope, h_hydro, v_hydro, h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness = input_panel("cl_")
    pred_class, probs = build_and_predict(elevation, aspect, slope, h_hydro, v_hydro,
                                           h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness)

    col, gcol, gbg, gbd = (COVER_COLORS[pred_class], COVER_COLORS[pred_class],
                            COVER_DIMS[pred_class], COVER_BRDS[pred_class])

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Classification Result</div>', unsafe_allow_html=True)

    c_res, c_chart, c_context = st.columns([1, 2.2, 1.3], gap="medium")

    with c_res:
        confidence = float(probs[pred_class]) * 100
        st.markdown(f"""
        <div class="result-g" style="border-color:{gbd};box-shadow:0 0 32px {gbg};">
          <div class="result-icon">{COVER_ICONS[pred_class]}</div>
          <div class="result-eyebrow">Predicted Type</div>
          <div class="result-name" style="color:{col};">{COVER_NAMES[pred_class]}</div>
          <div class="result-desc">{COVER_DESC[pred_class]}</div>
          <div class="conf-block" style="background:{gbg};border:1px solid {gbd};">
            <div class="conf-label" style="color:{col};">Confidence</div>
            <div class="conf-val" style="color:{col};">{confidence:.1f}%</div>
          </div>
          <div style="margin-top:14px;">
        """, unsafe_allow_html=True)
        for i in range(7):
            on = i == pred_class
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:5px 8px;border-radius:8px;margin-bottom:2px;
                        background:{''+COVER_DIMS[i] if on else 'transparent'};
                        border:1px solid {''+COVER_BRDS[i] if on else 'transparent'};">
              <span style="font-size:.73rem;color:{''+COVER_COLORS[i] if on else 'var(--text3)'};
                           font-weight:{'600' if on else '400'};display:flex;align-items:center;gap:6px;">
                <span style="width:7px;height:7px;border-radius:50%;background:{COVER_COLORS[i]};
                             box-shadow:0 0 5px {COVER_COLORS[i]};display:inline-block;"></span>
                {COVER_NAMES[i][:16]}
              </span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--text3);">
                {probs[i]:.1%}
              </span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with c_chart:
        # Probability chart
        st.markdown('<div class="gcard"><div class="ct-eyebrow">01 — Class Probabilities</div><div class="ct-title">Prediction Confidence · All 7 Cover Types</div>', unsafe_allow_html=True)
        bar_c = [COVER_COLORS[i] for i in range(7)]
        opacs = [0.9 if i == pred_class else 0.25 for i in range(7)]
        fig_prob = go.Figure(go.Bar(
            y=[COVER_NAMES[i] for i in range(7)],
            x=[probs[i] for i in range(7)],
            orientation='h',
            marker=dict(color=bar_c, opacity=opacs, line=dict(width=0)),
            text=[f'{probs[i]:.1%}' for i in range(7)],
            textposition='outside',
            textfont=dict(size=9.5, family='JetBrains Mono', color='#7C6FA0'),
            hovertemplate='%{y}: %{x:.1%}<extra></extra>',
            showlegend=False, width=0.6,
        ))
        fig_prob.update_layout(
            **CC(), margin=dict(l=0, r=52, t=4, b=4), height=260,
            xaxis=dict(range=[0,1.3], gridcolor=GRID, zeroline=False,
                       tickformat='.0%', tickfont=TICK, showline=True, linecolor=GRID),
            yaxis=dict(gridcolor='rgba(0,0,0,0)',
                       tickfont=dict(size=11, family='JetBrains Mono', color='#B8AED8')),
        )
        st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)

        # Input feature radar
        st.markdown('<div class="gcard"><div class="ct-eyebrow">02 — Input Feature Profile</div><div class="ct-title">Normalised Dimensions</div>', unsafe_allow_html=True)
        norm = {
            'Elevation':     (elevation-1800)/(3900-1800),
            'Aspect':        aspect/360,
            'Slope':         slope/52,
            'H-Hydro':       h_hydro/1400,
            'V-Hydro':       (v_hydro+150)/750,
            'H-Roads':       h_roads/7000,
            'Hillshade 9am': hs_9am/254,
            'Hillshade Noon':hs_noon/254,
        }
        labs = list(norm.keys()); vals = list(norm.values())
        bar_c2 = ['#34D399' if v>=.7 else '#FBBF24' if v>=.4 else '#F87171' for v in vals]
        fig_feat = go.Figure(go.Bar(
            y=labs, x=vals, orientation='h',
            marker=dict(color=bar_c2, opacity=0.82, line=dict(width=0)),
            text=[f'{v:.0%}' for v in vals], textposition='outside',
            textfont=dict(size=9.5, family='JetBrains Mono', color='#7C6FA0'),
            hovertemplate='%{y}: %{x:.1%}<extra></extra>', showlegend=False, width=0.55,
        ))
        fig_feat.update_layout(
            **CC(), margin=dict(l=0, r=52, t=4, b=4), height=220,
            xaxis=dict(range=[0,1.3], gridcolor=GRID, zeroline=False,
                       tickformat='.0%', tickfont=TICK, showline=True, linecolor=GRID),
            yaxis=dict(gridcolor='rgba(0,0,0,0)',
                       tickfont=dict(size=11, family='JetBrains Mono', color='#B8AED8')),
        )
        st.plotly_chart(fig_feat, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c_context:
        st.markdown('<div class="g-label">Elevation Zone</div>', unsafe_allow_html=True)
        elev_zones = [
            (1800, 2200, "Low Montane"),
            (2200, 2800, "Mid Montane"),
            (2800, 3200, "Upper Montane"),
            (3200, 3600, "Subalpine"),
            (3600, 3900, "Alpine"),
        ]
        for lo, hi, label in elev_zones:
            active = lo <= elevation < hi
            st.markdown(f"""
            <div class="elev-zone {"active" if active else ""}">
              <span class="elev-name {"active" if active else ""}">{"→ " if active else ""}{label}</span>
              <span class="elev-range {"active" if active else ""}">{lo}–{hi}m</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

        # Contextual insights
        tips = []
        if elevation > 3200:
            tips.append(("warn","◈","High Elevation","Subalpine zone — expect Krummholz or Spruce/Fir dominance."))
        elif elevation > 2800:
            tips.append(("ok","✓","Upper Montane","Prime Lodgepole Pine territory."))
        else:
            tips.append(("ok","✓","Mid Montane","Broad species mix, Ponderosa likely."))

        if slope > 30:
            tips.append(("warn","→","Steep Slope","High erosion risk, affects soil type distribution."))
        if h_hydro < 150:
            tips.append(("info","◉","Near Water","Riparian influence — favors Cottonwood/Willow."))
        if hs_9am < 100 or hs_noon < 150:
            tips.append(("warn","→","Low Hillshade","North-facing aspect — cold shaded microclimate."))

        for sev, ico, title, body in tips[:4]:
            st.markdown(f"""
            <div class="ins-g {sev}">
              <span class="ins-ico">{ico}</span>
              <div><div class="ins-t">{title}</div><div class="ins-b">{body}</div></div>
            </div>""", unsafe_allow_html=True)

        # Wilderness info
        w_data = {
            "Rawah":            ("Large wilderness, diverse elevations", "#8B5CF6"),
            "Neota":            ("Small, high-elevation meadows", "#60A5FA"),
            "Comanche Peak":    ("Complex terrain, mixed forest", "#34D399"),
            "Cache la Poudre":  ("Lower elevation, riparian areas", "#FBBF24"),
        }
        wdesc, wcol = w_data.get(wilderness, ("Wilderness area", "#8B5CF6"))
        st.markdown(f"""
        <div style="margin-top:8px;background:var(--violet-dim);border:1px solid var(--violet-brd);
                    border-radius:12px;padding:10px 12px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--violet2);
                      text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;">Wilderness Area</div>
          <div style="font-size:.85rem;font-weight:600;color:var(--text);">{wilderness}</div>
          <div style="font-size:.68rem;color:var(--text3);margin-top:3px;">{wdesc}</div>
        </div>""", unsafe_allow_html=True)

    # Cover type grid
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">All 7 Cover Types</div>', unsafe_allow_html=True)
    g7 = st.columns(7, gap="small")
    for i, col_w in enumerate(g7):
        is_pred = (i == pred_class)
        with col_w:
            st.markdown(f"""
            <div class="cover-g {"selected" if is_pred else ""}"
                 style="border-color:{''+COVER_BRDS[i] if is_pred else 'var(--glass-brd)'};
                        background:{''+COVER_DIMS[i] if is_pred else 'var(--glass-bg)'};">
              <div class="cover-icon">{COVER_ICONS[i]}</div>
              <div class="cover-name {"selected" if is_pred else ""}"
                   style="color:{''+COVER_COLORS[i] if is_pred else 'var(--text3)'};">
                {COVER_NAMES[i]}
              </div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;
                          color:{''+COVER_COLORS[i] if is_pred else 'var(--text4)'};margin-top:5px;">
                {probs[i]:.1%}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="g-footer">
      <div style="display:flex;gap:16px;align-items:center;">
        <span style="color:var(--violet2);">✦</span>
        <span>Random Forest Classifier</span>
        <span style="color:var(--glass-brd)">·</span>
        <span>54 features · 7 cover types · cartographic data</span>
      </div>
      <div>@abdel2ty</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  02 — EXPLORE
# ══════════════════════════════════════════════════════════════
elif page == "explore":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    elevation, aspect, slope, h_hydro, v_hydro, h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness = input_panel("ex_")
    pred_class, probs = build_and_predict(elevation, aspect, slope, h_hydro, v_hydro,
                                           h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness)

    # Metrics strip
    st.markdown(f"""
    <div class="g-strip" style="margin-top:1rem;">
      <div class="g-strip-item"><div class="gsi-v">{COVER_ICONS[pred_class]} {COVER_NAMES[pred_class]}</div><div class="gsi-l">Predicted Type</div></div>
      <div class="g-strip-item"><div class="gsi-v" style="color:{COVER_COLORS[pred_class]};">{probs[pred_class]:.0%}</div><div class="gsi-l">Confidence</div></div>
      <div class="g-strip-item"><div class="gsi-v">{elevation}m</div><div class="gsi-l">Elevation</div></div>
      <div class="g-strip-item"><div class="gsi-v">{slope}°</div><div class="gsi-l">Slope</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Elevation Sensitivity</div>', unsafe_allow_html=True)

    st.markdown('<div class="gcard"><div class="ct-eyebrow">Cover Type vs Elevation</div><div class="ct-title">Predicted Class as Elevation Varies · All Other Inputs Fixed</div>', unsafe_allow_html=True)
    elev_range = np.arange(1800, 3901, 30)
    elev_preds = []
    for e in elev_range:
        p, _ = build_and_predict(int(e), aspect, slope, h_hydro, v_hydro,
                                  h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness)
        elev_preds.append(p)

    fig_elev = go.Figure()
    for i in range(7):
        mask = np.array(elev_preds) == i
        if mask.any():
            fig_elev.add_trace(go.Scatter(
                x=elev_range[mask], y=np.ones(mask.sum())*i,
                mode='markers',
                marker=dict(color=COVER_COLORS[i], size=10, opacity=0.85, symbol='square',
                            line=dict(color='rgba(0,0,0,.4)', width=1)),
                name=COVER_NAMES[i],
                hovertemplate=f'{COVER_NAMES[i]}<br>%{{x}}m<extra></extra>'
            ))
    fig_elev.add_vline(x=elevation, line=dict(color='#EC4899', width=1.5, dash='dot'),
        annotation_text="  baseline", annotation_font=dict(size=9, family='JetBrains Mono', color='#EC4899'))
    fig_elev.update_layout(
        **CC(), margin=dict(l=0,r=0,t=0,b=0), height=220,
        xaxis=dict(title="Elevation (m)", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title_font=AX, showline=True, linecolor=GRID),
        yaxis=dict(title="Cover Type ID", gridcolor=GRID, zeroline=False, dtick=1, tickfont=TICK,
                   title_font=AX, range=[-0.5, 6.5]),
        legend=dict(font=dict(size=9, family='JetBrains Mono'), bgcolor='rgba(13,11,26,.8)',
                    bordercolor=GRID, borderwidth=1, orientation='h', y=-0.3)
    )
    st.plotly_chart(fig_elev, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Input Feature Sensitivity Curves</div>', unsafe_allow_html=True)

    r1 = st.columns(2, gap="medium")
    r2 = st.columns(2, gap="medium")

    def sweep_curve(param_idx, rng, cur_val, label, x_label, color):
        xs = np.arange(rng[0], rng[1]+1, max(1,(rng[1]-rng[0])//50))
        params_base = [elevation, aspect, slope, h_hydro, v_hydro, h_roads, h_fire, hs_9am, hs_noon, hs_3pm, wilderness]
        ys = []
        for v in xs:
            p_args = list(params_base); p_args[param_idx] = int(v)
            c, pr = build_and_predict(*p_args)
            ys.append(pr[pred_class] * 100)
        r = int(color[1:3],16); g2 = int(color[3:5],16); b = int(color[5:7],16)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=2.5, shape='spline'),
            fill='tozeroy', fillcolor=f'rgba({r},{g2},{b},0.08)',
            hovertemplate=f'{x_label}: %{{x}} → %{{y:.1f}}%<extra></extra>', showlegend=False
        ))
        fig.add_vline(x=cur_val, line=dict(color='#EC4899', width=1.5, dash='dot'))
        fig.update_layout(
            **CC(), margin=dict(l=0,r=16,t=4,b=4), height=200,
            xaxis=dict(title=x_label, gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Confidence %", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
        )
        return fig

    sweep_defs = [
        (0, (1800,3900), elevation, "Elevation Sweep",   "Elevation (m)", "#8B5CF6"),
        (2, (0,52),      slope,     "Slope Sweep",        "Slope (°)",     "#60A5FA"),
        (3, (0,1400),    h_hydro,   "H-Hydro Sweep",      "Distance (m)",  "#FBBF24"),
        (5, (0,7000),    h_roads,   "H-Roads Sweep",      "Distance (m)",  "#EC4899"),
    ]
    for col_w, (pidx, rng, cur, lbl, xl, col_c) in zip([r1[0],r1[1],r2[0],r2[1]], sweep_defs):
        with col_w:
            st.markdown(f'<div class="gcard"><div class="ct-eyebrow">Sensitivity</div><div class="ct-title">{lbl}</div>', unsafe_allow_html=True)
            st.plotly_chart(sweep_curve(pidx, rng, cur, lbl, xl, col_c),
                            use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="g-footer">
      <div style="display:flex;gap:16px;align-items:center;">
        <span style="color:var(--violet2);">✦</span>
        <span>Feature Explorer</span>
        <span style="color:var(--glass-brd)">·</span>
        <span>Elevation sweep · 4 sensitivity curves</span>
      </div>
      <div>@abdel2ty</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  03 — REFERENCE
# ══════════════════════════════════════════════════════════════
elif page == "reference":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown('<div class="g-label">Cover Type Overview</div>', unsafe_allow_html=True)

    kc = st.columns(4, gap="small")
    for col_w, (val, lbl, sub) in zip(kc, [
        ("7",  "Cover Classes",  "Multi-class classification"),
        ("54", "Input Features", "Topographic + cartographic"),
        ("4",  "Wilderness Areas","Roosevelt National Forest"),
        ("40", "Soil Types",     "USFS ecological mapping"),
    ]):
        with col_w:
            st.markdown(f"""
            <div class="kpi-g">
              <div class="kpi-accent" style="background:linear-gradient(180deg,var(--violet),transparent);"></div>
              <div class="kpi-val" style="padding-left:8px;">{val}</div>
              <div class="kpi-name" style="padding-left:8px;">{lbl}</div>
              <div class="kpi-meta" style="padding-left:8px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Forest Cover Types · Detailed Reference</div>', unsafe_allow_html=True)

    for row_start in range(0, 7, 4):
        row_items = list(range(row_start, min(row_start+4, 7)))
        cols_w = st.columns(len(row_items), gap="medium")
        for col_w, i in zip(cols_w, row_items):
            with col_w:
                st.markdown(f"""
                <div class="gcard" style="border-color:{COVER_BRDS[i]};">
                  <div style="font-size:2.8rem;margin-bottom:8px;">{COVER_ICONS[i]}</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;
                              color:{COVER_COLORS[i]};text-transform:uppercase;letter-spacing:.14em;
                              margin-bottom:3px;opacity:.8;">Type {i}</div>
                  <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;
                              color:{COVER_COLORS[i]};margin-bottom:6px;letter-spacing:-.02em;">
                    {COVER_NAMES[i]}
                  </div>
                  <div style="height:2px;background:linear-gradient(90deg,{COVER_COLORS[i]},transparent);
                              border-radius:2px;margin-bottom:10px;opacity:.5;"></div>
                  <div style="font-size:.73rem;color:var(--text2);line-height:1.55;">{COVER_DESC[i]}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Feature Reference</div>', unsafe_allow_html=True)

    feat_ref = [
        ("Elevation",          "1800 – 3900m", "Numeric",  "Primary predictor. Determines temperature and moisture."),
        ("Aspect",             "0 – 360°",     "Numeric",  "Compass direction the slope faces. Affects sun exposure."),
        ("Slope",              "0 – 52°",      "Numeric",  "Steepness of terrain. Affects soil drainage and erosion."),
        ("H-Dist to Hydrology","0 – 1400m",    "Numeric",  "Horizontal distance to nearest water feature."),
        ("V-Dist to Hydrology","-150 – 600m",  "Numeric",  "Vertical distance to nearest water feature."),
        ("H-Dist to Roads",    "0 – 7000m",    "Numeric",  "Distance from road network. Proxy for human influence."),
        ("H-Dist to Fire",     "0 – 7000m",    "Numeric",  "Distance to historical fire ignition points."),
        ("Hillshade (3×)",     "0 – 254",      "Numeric",  "Shadow index at 9am, Noon, and 3pm."),
        ("Wilderness Area",    "4 types",      "Binary",   "One-hot encoded wilderness designation."),
        ("Soil Type",          "40 types",     "Binary",   "One-hot encoded USFS soil classification."),
    ]

    rows = ""
    for name, rng, typ, desc in feat_ref:
        chip_c   = "#8B5CF6" if typ=="Binary" else "#60A5FA"
        chip_bg  = "rgba(139,92,246,0.12)" if typ=="Binary" else "rgba(96,165,250,0.12)"
        chip_brd = "rgba(139,92,246,0.3)" if typ=="Binary" else "rgba(96,165,250,0.3)"
        rows += f"""<tr>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--text);">{name}</td>
          <td style="font-family:'JetBrains Mono',monospace;">{rng}</td>
          <td><span style="display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;
                           background:{chip_bg};color:{chip_c};border:1px solid {chip_brd};
                           font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.05em;">{typ}</span></td>
          <td style="color:var(--text2);">{desc}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:var(--glass-bg);backdrop-filter:blur(12px);border:1px solid var(--glass-brd);border-radius:14px;overflow:hidden;">
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr>
          <th style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:12px 16px;text-align:left;border-bottom:1px solid var(--glass-brd);background:rgba(255,255,255,.03);">Feature</th>
          <th style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:12px 16px;text-align:left;border-bottom:1px solid var(--glass-brd);background:rgba(255,255,255,.03);">Range</th>
          <th style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:12px 16px;text-align:left;border-bottom:1px solid var(--glass-brd);background:rgba(255,255,255,.03);">Type</th>
          <th style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:12px 16px;text-align:left;border-bottom:1px solid var(--glass-brd);background:rgba(255,255,255,.03);">Description</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="g-footer">
      <div style="display:flex;gap:16px;align-items:center;">
        <span style="color:var(--violet2);">✦</span>
        <span>ForestIQ · Multi-class Classification</span>
        <span style="color:var(--glass-brd)">·</span>
        <span>Random Forest · Scikit-learn</span>
      </div>
      <div>@abdel2ty</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
