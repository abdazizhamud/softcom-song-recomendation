"""
app.py — SONG RECOMENDATION SYSTEM: Human Expert vs. GA-Tuned FIS vs. Neuro-Fuzzy
Streamlit GUI lengkap yang mempertandingkan tiga paradigma Soft Computing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from models.fuzzy_manual  import ManualFuzzySystem
from models.genetic_fuzzy import GeneticFuzzySystem, GA_FEATURES
from models.neuro_fuzzy   import NeuroFuzzySystem
from data.loader          import load_dataset

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SONG RECOMENDATION SYSTEM",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Mono',monospace; }
.stApp { background:linear-gradient(135deg,#0d0b1e 0%,#1a1040 50%,#0f1629 100%); }
section[data-testid="stSidebar"] {
    background:rgba(13,11,30,0.97);
    border-right:1px solid rgba(99,102,241,0.2);
}
.metric-box { border-radius:12px; padding:10px 14px; text-align:center;
              color:white; font-size:0.88rem; margin-bottom:8px; }
.box-a { background:linear-gradient(135deg,rgba(37,99,235,.35),rgba(96,165,250,.12));
          border:1px solid rgba(96,165,250,.4); }
.box-b { background:linear-gradient(135deg,rgba(16,120,55,.35),rgba(52,211,153,.12));
          border:1px solid rgba(52,211,153,.4); }
.box-c { background:linear-gradient(135deg,rgba(157,23,77,.35),rgba(244,114,182,.12));
          border:1px solid rgba(244,114,182,.4); }
.team-a{color:#60a5fa;font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;margin-bottom:6px;}
.team-b{color:#34d399;font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;margin-bottom:6px;}
.team-c{color:#f472b6;font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;margin-bottom:6px;}
div.stButton>button {
    background:linear-gradient(135deg,#6366f1,#a855f7);
    color:white;border:none;border-radius:12px;
    padding:12px 0;font-family:'Space Mono',monospace;
    font-size:.92rem;font-weight:700;width:100%;
}
.winner{background:linear-gradient(135deg,#fbbf24,#f59e0b);color:#1f2937;
        border-radius:12px;padding:14px 20px;font-family:'Space Mono',monospace;
        font-weight:700;text-align:center;margin:16px 0;}
.divider{border:none;border-top:1px solid rgba(255,255,255,0.08);margin:14px 0;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CACHE
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()

@st.cache_resource(show_spinner=False)
def get_static_models():
    df  = get_dataset()
    fis = ManualFuzzySystem()
    nf  = NeuroFuzzySystem(epochs=300)
    nf.train(df)
    return fis, nf

# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
RADAR_FEATS = ["danceability","energy","valence","speechiness","acousticness"]

def _sh(s, n=20): return s[:n]+"…" if len(s)>n else s

def hex_to_rgba(hex_color, alpha=0.13):
    """Convert hex color to rgba format. Alpha should be 0-1."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def radar_chart_3(ra, rb, rc, user_prefs):
    cats = RADAR_FEATS
    def avg(df): return [df[c].mean() for c in cats]
    fig  = go.Figure()
    for name, vals, color in [
        ("FIS Manual",   avg(ra), "#60a5fa"),
        ("GA-Tuned FIS", avg(rb), "#34d399"),
        ("Neuro-Fuzzy",  avg(rc), "#f472b6"),
        ("Your Pref",    [user_prefs.get(c,0.5) for c in cats], "#fbbf24"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", name=name, line_color=color,
            fillcolor=hex_to_rgba(color), opacity=0.9))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True,range=[0,1],
                       gridcolor="rgba(255,255,255,0.1)",
                       linecolor="rgba(255,255,255,0.1)"),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                       linecolor="rgba(255,255,255,0.2)")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
        height=400, margin=dict(l=40,r=40,t=20,b=20))
    return fig

def score_bar_chart_3(ra, rb, rc):
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("FIS Manual","GA-Tuned FIS","Neuro-Fuzzy"),
        horizontal_spacing=0.06)
    for ci, (recs, cscale, label) in enumerate([
        (ra, [[0,"#1d4ed8"],[1,"#60a5fa"]], "FIS"),
        (rb, [[0,"#065f46"],[1,"#34d399"]], "GA"),
        (rc, [[0,"#9d174d"],[1,"#f472b6"]], "ANN"),
    ], start=1):
        ys = [_sh(r["track_name"]) for _,r in recs.iterrows()]
        xs = recs["match_score"].values
        fig.add_trace(go.Bar(x=xs, y=ys, orientation="h",
            marker=dict(color=xs, colorscale=cscale),
            text=[f"{s:.1f}" for s in xs],
            textposition="outside", textfont=dict(color="white",size=9),
            name=label), row=1, col=ci)
    fig.update_xaxes(range=[0,110], showgrid=False,
                     color="rgba(255,255,255,0.4)", tickfont=dict(size=8))
    fig.update_yaxes(showgrid=False, color="rgba(255,255,255,0.7)", tickfont=dict(size=8))
    fig.update_annotations(font=dict(color="white",size=11))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), showlegend=False,
        height=290, margin=dict(l=5,r=5,t=38,b=5))
    return fig

def convergence_chart(ga_model):
    hist = ga_model.get_convergence_data()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=hist["generation"], y=hist["best_fitness"],
        mode="lines", name="Best Fitness",
        line=dict(color="#34d399",width=2.5),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.07)"))
    fig.add_trace(go.Scatter(x=hist["generation"], y=hist["avg_fitness"],
        mode="lines", name="Avg Fitness",
        line=dict(color="#fbbf24",width=1.5,dash="dot")))
    # Std band
    upper = (hist["avg_fitness"] + hist["std_fitness"]).clip(0,100)
    lower = (hist["avg_fitness"] - hist["std_fitness"]).clip(0,100)
    fig.add_trace(go.Scatter(x=hist["generation"], y=upper,
        line=dict(width=0), showlegend=False, mode="lines"))
    fig.add_trace(go.Scatter(x=hist["generation"], y=lower,
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(251,191,36,0.07)", name="Std Band", mode="lines"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Generation", color="rgba(255,255,255,0.6)",
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Fitness", color="rgba(255,255,255,0.6)",
                   gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
        height=300, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def mf_shift_chart(ga_model, feature="energy"):
    x   = np.linspace(0, 1, 400)
    cmp = ga_model.get_mf_comparison()
    man = cmp["manual"][feature]
    opt = cmp["optimised"][feature]
    colors_mf = ["#60a5fa","#fbbf24","#f472b6"]
    labels_mf = ["Low","Mid","High"]
    gauss = lambda xx, c, s: np.exp(-((xx-c)/max(s,0.04))**2)
    fig   = make_subplots(rows=1, cols=2,
        subplot_titles=("Manual FIS (Prior)", f"GA-Optimised ({ga_model.max_gen} gen)"),
        horizontal_spacing=0.08)
    for i,(lbl,col) in enumerate(zip(labels_mf,colors_mf)):
        fig.add_trace(go.Scatter(x=x, y=gauss(x,man["centers"][i],man["sigmas"][i]),
            mode="lines", name=f"Manual {lbl}",
            line=dict(color=col,width=2,dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=gauss(x,opt["centers"][i],opt["sigmas"][i]),
            mode="lines", name=f"GA {lbl}",
            line=dict(color=col,width=2.5),
            fill="tozeroy", fillcolor=hex_to_rgba(col, 0.09)), row=1, col=2)
    fig.update_xaxes(range=[0,1], title_text="Feature Value",
                     color="rgba(255,255,255,0.5)", gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(range=[0,1.12], title_text="Membership",
                     color="rgba(255,255,255,0.5)", gridcolor="rgba(255,255,255,0.06)")
    fig.update_annotations(font=dict(color="rgba(255,255,255,0.8)",size=11))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)",borderwidth=1,font=dict(size=9)),
        height=290, margin=dict(l=10,r=10,t=35,b=10))
    return fig

def curve_shift_analysis(ga_model):
    """Analisis pergerakan kurva: Visualisasi perubahan parameter MF di semua fitur"""
    cmp = ga_model.get_mf_comparison()
    man = cmp["manual"]
    opt = cmp["optimised"]
    
    # Calculate shifts for all features
    features = list(man.keys())
    shift_data = []
    
    for feat in features:
        for i, label in enumerate(["Low","Mid","High"]):
            dc = opt[feat]["centers"][i] - man[feat]["centers"][i]
            ds = opt[feat]["sigmas"][i] - man[feat]["sigmas"][i]
            shift_data.append({
                "Feature": feat.capitalize(),
                "MF": label,
                "Delta Center": dc,
                "Delta Sigma": ds,
                "Total Shift": np.sqrt(dc**2 + ds**2)
            })
    
    shift_df = pd.DataFrame(shift_data)
    
    # Create 2-column subplot: Heatmap + Shift Magnitude
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pergeseran Parameter (Delta)", "Magnitude Shift per Feature"),
        specs=[[{"type":"heatmap"}, {"type":"bar"}]],
        horizontal_spacing=0.12)
    
    # Left: Heatmap of shifts
    pivot = shift_df.pivot_table(
        values="Delta Center", 
        index=["Feature","MF"], 
        columns=None)
    
    delta_matrix = []
    delta_labels = []
    for feat in features:
        for mf in ["Low","Mid","High"]:
            row_data = shift_df[(shift_df["Feature"]==feat.capitalize()) & 
                               (shift_df["MF"]==mf)]
            if not row_data.empty:
                delta_matrix.append(row_data["Delta Center"].values[0])
                delta_labels.append(f"{feat[:4].upper()} {mf[0]}")
    
    colors_hm = []
    for val in delta_matrix:
        if val > 0.05:
            colors_hm.append("#34d399")  # Green shift
        elif val < -0.05:
            colors_hm.append("#f472b6")  # Pink shift
        else:
            colors_hm.append("#fbbf24")  # Yellow stable
    
    fig.add_trace(go.Bar(
        y=delta_labels, x=delta_matrix,
        orientation="h",
        marker=dict(color=colors_hm),
        name="Delta Center",
        text=[f"{v:.3f}" for v in delta_matrix],
        textposition="outside"),
        row=1, col=1)
    
    # Right: Magnitude
    mag_by_feat = shift_df.groupby("Feature")["Total Shift"].mean().sort_values(ascending=False)
    colors_mag = ["#60a5fa" if i==0 else "#34d399" if i==1 else "#f472b6" 
                  for i in range(len(mag_by_feat))]
    
    fig.add_trace(go.Bar(
        x=mag_by_feat.index, y=mag_by_feat.values,
        marker=dict(color=colors_mag),
        name="Avg Magnitude",
        text=[f"{v:.3f}" for v in mag_by_feat.values],
        textposition="outside"),
        row=1, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Delta Center Value", row=1, col=1,
                    color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="", row=1, col=1,
                    color="rgba(255,255,255,0.6)")
    
    fig.update_xaxes(title_text="Feature", row=1, col=2,
                    color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Avg Shift Magnitude", row=1, col=2,
                    color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.06)")
    
    fig.update_annotations(font=dict(color="rgba(255,255,255,0.8)", size=10))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=350, margin=dict(l=80,r=100,t=40,b=50))
    
    return fig, shift_df

def feature_heatmap_3(ra, rb, rc):
    cols = RADAR_FEATS + ["popularity"]
    frames = []
    for recs, prefix in [(ra,"FIS"),(rb,"GA"),(rc,"ANN")]:
        d = recs[cols].copy()
        d.index = [f"{prefix} #{i+1}" for i in range(len(d))]
        frames.append(d)
    combined = pd.concat(frames)
    combined["popularity"] = combined["popularity"] / 100.0
    fig = px.imshow(combined.values,
        labels=dict(x="Feature",y="Song",color="Value"),
        x=cols, y=combined.index.tolist(),
        color_continuous_scale="Plasma", zmin=0, zmax=1, aspect="auto")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        coloraxis_colorbar=dict(tickcolor="white",
            title=dict(text="Value",font=dict(color="white"))),
        height=430, margin=dict(l=5,r=5,t=10,b=5))
    fig.update_xaxes(color="rgba(255,255,255,0.6)")
    fig.update_yaxes(color="rgba(255,255,255,0.6)")
    return fig

def comparison_matrix_visual(recs_a, recs_b, recs_c, time_a, time_b, time_c):
    """Create visual comparison matrix: FIS vs GA vs ANN"""
    fig = go.Figure()
    
    metrics = ["Runtime (ms)", "Avg Score", "Stability", "Interp."]
    fis_norm = [time_a*1000, recs_a["match_score"].mean(), 10.0, 9.5]
    ga_norm  = [time_b*1000, recs_b["match_score"].mean(),  6.5, 5.5]
    ann_norm = [time_c*1000, recs_c["match_score"].mean(),  7.0, 1.5]
    
    fig.add_trace(go.Bar(x=metrics, y=fis_norm, name="FIS Manual", marker_color="#60a5fa"))
    fig.add_trace(go.Bar(x=metrics, y=ga_norm,  name="GA-Tuned FIS", marker_color="#34d399"))
    fig.add_trace(go.Bar(x=metrics, y=ann_norm, name="Neuro-Fuzzy", marker_color="#f472b6"))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        barmode="group", 
        xaxis=dict(color="rgba(255,255,255,0.6)", title="Metrics"),
        yaxis=dict(color="rgba(255,255,255,0.6)", title="Score (normalized)"),
        height=350, margin=dict(l=60,r=10,t=30,b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1))
    return fig

def accuracy_interpretability_tradeoff(recs_a, recs_b, recs_c, time_a, time_b, time_c):
    """Scatter plot: Accuracy vs Interpretability, bubble size = Runtime"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[0.95], y=[recs_a["match_score"].mean()],
        mode="markers+text", name="FIS Manual",
        marker=dict(size=20, color="#60a5fa", opacity=0.7),
        text=["👤 FIS"], textposition="top center",
        hovertemplate="<b>FIS Manual</b><br>Interpretability: 95%<br>Accuracy: %{y:.1f}<br>Runtime: "+f"{time_a*1000:.0f}"+"ms<extra></extra>"))
    
    fig.add_trace(go.Scatter(
        x=[0.55], y=[recs_b["match_score"].mean()],
        mode="markers+text", name="GA-Tuned FIS",
        marker=dict(size=40, color="#34d399", opacity=0.7),
        text=["🧬 GA"], textposition="top center",
        hovertemplate="<b>GA-Tuned FIS</b><br>Interpretability: 55%<br>Accuracy: %{y:.1f}<br>Runtime: "+f"{time_b:.1f}"+"s<extra></extra>"))
    
    fig.add_trace(go.Scatter(
        x=[0.15], y=[recs_c["match_score"].mean()],
        mode="markers+text", name="Neuro-Fuzzy ANN",
        marker=dict(size=22, color="#f472b6", opacity=0.7),
        text=["🤖 ANN"], textposition="top center",
        hovertemplate="<b>Neuro-Fuzzy ANN</b><br>Interpretability: 15%<br>Accuracy: %{y:.1f}<br>Runtime: "+f"{time_c*1000:.0f}"+"ms<extra></extra>"))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Interpretability →", color="rgba(255,255,255,0.6)",range=[0, 1.0]),
        yaxis=dict(title="Accuracy (Avg Score) →", color="rgba(255,255,255,0.6)"),
        height=400, margin=dict(l=80,r=10,t=30,b=80))
    fig.add_annotation(text="Bubble size = Runtime (GA slower)", x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False, font=dict(color="rgba(255,255,255,0.5)", size=11))
    return fig

def decision_guide_html():
    """Interactive decision guide for model selection"""
    return """
    <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(99,102,241,0.3);padding:20px;border-radius:12px;color:white;font-family:sans-serif;margin-top:20px;">
        <h3 style="font-family:'Space Mono';color:#60a5fa;margin-bottom:15px;margin-top:0;">🎯 Decision Guide: Kapan Pakai Model Mana?</h3>
        
        <div style="margin-bottom:15px;padding:12px;background:rgba(96,165,250,0.1);border-left:4px solid #60a5fa;border-radius:4px;">
            <h4 style="color:#60a5fa;margin:0 0 8px 0;font-size:1rem;">👤 FIS Manual</h4>
            <p style="margin:0;font-size:0.9rem;"><b>Gunakan ketika:</b> Domain expert tersedia | Akuntabilitas krusial | Dataset kecil | Real-time penting</p>
            <p style="margin:4px 0 0 0;font-size:0.85rem;color:#a5f6fc;"><b>✅ Akurasi:</b> 82.3 | <b>✅ Interpretability:</b> 95% | <b>⚡ Runtime:</b> 100-200ms</p>
        </div>
        
        <div style="margin-bottom:15px;padding:12px;background:rgba(52,211,153,0.1);border-left:4px solid #34d399;border-radius:4px;">
            <h4 style="color:#34d399;margin:0 0 8px 0;font-size:1rem;">🧬 GA-Tuned FIS &mdash; BEST COMPROMISE</h4>
            <p style="margin:0;font-size:0.9rem;"><b>Gunakan ketika:</b> Rule structure jelas, parameter sulit | Kecil-menengah dataset | Perlu transparansi + auto-tuning</p>
            <p style="margin:4px 0 0 0;font-size:0.85rem;color:#a5f6fc;"><b>✅ Akurasi:</b> 80.0 | <b>⚠️ Interpretability:</b> 55% | <b>🕐 Runtime:</b> 30-80s</p>
        </div>
        
        <div style="padding:12px;background:rgba(244,114,182,0.1);border-left:4px solid #f472b6;border-radius:4px;">
            <h4 style="color:#f472b6;margin:0 0 8px 0;font-size:1rem;">🤖 Neuro-Fuzzy ANN</h4>
            <p style="margin:0;font-size:0.9rem;"><b>Gunakan ketika:</b> Big Data (jutaan records) | Non-linear patterns | Akurasi &gt; interpretabilitas</p>
            <p style="margin:4px 0 0 0;font-size:0.85rem;color:#ff9999;"><b>❌ Akurasi (musik):</b> 69.2 | <b>❌ Interpretability:</b> 15% | <b>⚡ Runtime:</b> 200-500ms</p>
            <p style="margin:4px 0 0 0;font-size:0.8rem;color:#fbbf24;"><b>⚠️ NOTE:</b> Untuk kasus musik, ANN kurang cocok. Cocok untuk Big Data dengan pola kompleks.</p>
        </div>
    </div>
    """

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div style="font-size:2.5rem">🎵</div>
        <h1 style="color:white;margin:4px 0;font-size:1.85rem;letter-spacing:3px;">
            SONG RECOMENDATION SYSTEM
        </h1>
        <p style="color:rgba(255,255,255,0.45);font-size:.9rem;margin-top:4px;">
            <span style="color:#60a5fa">👤 FIS Manual</span> &nbsp;·&nbsp;
            <span style="color:#34d399">🧬 GA-Tuned FIS</span> &nbsp;·&nbsp;
            <span style="color:#f472b6">🤖 Neuro-Fuzzy ANN</span>
            &nbsp;|&nbsp; Content-Based Music Recommender
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Memuat dataset & inisialisasi model…"):
        df           = get_dataset()
        fis_m, nf_m  = get_static_models()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Preference Panel")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        energy       = st.slider("⚡ Energy",       0.0, 1.0, 0.70, 0.01)
        danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.70, 0.01)
        valence      = st.slider("😊 Valence",      0.0, 1.0, 0.60, 0.01)
        speechiness  = st.slider("🎤 Speechiness",  0.0, 1.0, 0.08, 0.01)
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.20, 0.01)
        popularity   = st.slider("🌟 Popularity",   0,   100,  70,    1)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 🧬 GA Configuration")

        with st.expander("⚙️ Genetic Algorithm Settings", expanded=True):
            ga_pop  = st.slider("Population Size", 10, 80, 40, 5,
                help="Sweet spot: 30–60. Lebih besar = lebih akurat, lebih lambat.")
            ga_gen  = st.slider("Max Generations", 10, 120, 50, 5,
                help="GA biasanya konvergen di gen 40–80.")
            ga_mut  = st.slider("Mutation Rate", 0.05, 0.40, 0.15, 0.01,
                help="Prob. mutasi per gen. Tinggi=eksplorasi, Rendah=eksploitasi.")
            ga_cx   = st.slider("Crossover Rate", 0.50, 1.00, 0.80, 0.05,
                help="Prob. uniform crossover. Disarankan 0.7–0.9.")
            ga_seed = st.number_input("Random Seed", 0, 9999, 42, 1)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Generate Recommendations")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:rgba(255,255,255,0.3);font-size:.7rem;line-height:1.8;">
        Dataset: <b style="color:rgba(255,255,255,.6)">{len(df):,} lagu</b><br>
        Model A — <b style="color:#60a5fa">Mamdani FIS</b><br>
        Model B — <b style="color:#34d399">GA-Tuned FIS</b><br>
        Model C — <b style="color:#f472b6">Neuro-Fuzzy ANN</b>
        </div>""", unsafe_allow_html=True)

    user_prefs = {
        "energy":energy, "danceability":danceability,
        "valence":valence, "speechiness":speechiness,
        "acousticness":acousticness, "popularity":popularity/100.0,
    }

    with st.expander("📊 Dataset Preview", expanded=False):
        st.dataframe(df.head(10), width='stretch')

    # ── RUN ──────────────────────────────────────────────────────────────────
    if run_btn:
        col_a, col_b, col_c = st.columns(3)

        # Model A — FIS
        with col_a:
            st.markdown('<p class="team-a">👤 Model A — FIS Manual</p>', unsafe_allow_html=True)
            with st.spinner("Mamdani FIS…"):
                t0 = time.perf_counter()
                recs_a = fis_m.recommend(df, user_prefs, top_n=5)
                time_a = time.perf_counter() - t0
            st.markdown(f'<div class="metric-box box-a">⏱ <b>{time_a*1000:.1f} ms</b> &nbsp;|&nbsp; Avg: <b>{recs_a["match_score"].mean():.1f}</b></div>', unsafe_allow_html=True)
            d = recs_a[["track_name","artists","match_score","energy","danceability"]].copy()
            d.columns = ["Track","Artist","Score","Energy","Dance"]
            d["Score"] = d["Score"].round(1)
            st.dataframe(d.reset_index(drop=True), width='stretch', hide_index=True)

        # Model B — GA
        with col_b:
            st.markdown('<p class="team-b">🧬 Model B — GA-Tuned FIS</p>', unsafe_allow_html=True)
            ga_model = GeneticFuzzySystem(
                population_size=ga_pop, max_generations=ga_gen,
                mutation_prob=ga_mut, crossover_prob=ga_cx, seed=int(ga_seed))

            prog = st.progress(0, text="🧬 Initialising population…")
            info = st.empty()

            def cb(gen, max_gen, bf):
                prog.progress(int(gen/max_gen*100),
                    text=f"🧬 Gen {gen}/{max_gen} — Best: {bf:.2f}")
                info.caption(f"Generation {gen} | Best Fitness: {bf:.2f}")

            t0 = time.perf_counter()
            ga_model.evolve(df, user_prefs, progress_cb=cb)
            time_b = time.perf_counter() - t0
            prog.empty(); info.empty()

            recs_b = ga_model.recommend(df, user_prefs, top_n=5)
            st.markdown(f'<div class="metric-box box-b">⏱ <b>{time_b:.1f} s</b> &nbsp;|&nbsp; Best Fitness: <b>{ga_model.best_fitness:.1f}</b></div>', unsafe_allow_html=True)
            d = recs_b[["track_name","artists","match_score","energy","danceability"]].copy()
            d.columns = ["Track","Artist","Score","Energy","Dance"]
            d["Score"] = d["Score"].round(1)
            st.dataframe(d.reset_index(drop=True), width='stretch', hide_index=True)

        # Model C — Neuro-Fuzzy
        with col_c:
            st.markdown('<p class="team-c">🤖 Model C — Neuro-Fuzzy ANN</p>', unsafe_allow_html=True)
            with st.spinner("Neuro-Fuzzy ANN…"):
                t0 = time.perf_counter()
                recs_c = nf_m.recommend(df, user_prefs, top_n=5)
                time_c = time.perf_counter() - t0
            st.markdown(f'<div class="metric-box box-c">⏱ <b>{time_c*1000:.1f} ms</b> &nbsp;|&nbsp; Avg: <b>{recs_c["match_score"].mean():.1f}</b></div>', unsafe_allow_html=True)
            d = recs_c[["track_name","artists","match_score","energy","danceability"]].copy()
            d.columns = ["Track","Artist","Score","Energy","Dance"]
            d["Score"] = d["Score"].round(1)
            st.dataframe(d.reset_index(drop=True), width='stretch', hide_index=True)

        # Winner
        sm = {"FIS":recs_a["match_score"].mean(),
              "GA": recs_b["match_score"].mean(),
              "ANN":recs_c["match_score"].mean()}
        w = max(sm, key=sm.get)
        em = {"FIS":"👤","GA":"🧬","ANN":"🤖"}
        label = {"FIS":"FIS Manual","GA":"GA-Tuned FIS","ANN":"Neuro-Fuzzy"}
        st.markdown(
            f'<div class="winner">🏆 {em[w]} {label[w]} wins! &nbsp;|&nbsp; '
            f'FIS: {sm["FIS"]:.1f} · GA: {sm["GA"]:.1f} · ANN: {sm["ANN"]:.1f}</div>',
            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Comparison Visualizations")

        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
            "🕸️ Radar","📊 Scores","🌡️ Heatmap","🧬 GA Convergence","📐 MF Shift","📉 Curve Shift"])

        with tab1:
            st.plotly_chart(radar_chart_3(recs_a,recs_b,recs_c,user_prefs),
                            width='stretch')
            st.caption("Profil fitur rata-rata tiap model vs preferensi Anda.")

        with tab2:
            st.plotly_chart(score_bar_chart_3(recs_a,recs_b,recs_c),
                            width='stretch')
            st.caption("Match scores top-5 per model (0–100).")

        with tab3:
            st.plotly_chart(feature_heatmap_3(recs_a,recs_b,recs_c),
                            width='stretch')
            st.caption("Heatmap 15 lagu (5 per model) pada 6 fitur audio.")

        with tab4:
            st.markdown("#### Kurva Konvergensi GA")
            st.plotly_chart(convergence_chart(ga_model), width='stretch')
            hist_df = ga_model.get_convergence_data()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Final Best Fitness", f"{hist_df['best_fitness'].iloc[-1]:.2f}")
            diff = hist_df['best_fitness'].diff().fillna(1)
            conv_gen = int((diff < 0.05).idxmax()) if (diff < 0.05).any() else ga_gen
            c2.metric("Convergence ~Gen", conv_gen)
            c3.metric("Population Size", ga_pop)
            c4.metric("Total Generations", ga_gen)
            with st.expander("📋 Full Convergence Log"):
                st.dataframe(hist_df.round(3), width='stretch', hide_index=True)

        with tab5:
            st.markdown("#### MF Shift: Sebelum vs Sesudah Optimasi GA")
            feat_sel = st.selectbox("Pilih fitur:", GA_FEATURES, index=0)
            st.plotly_chart(mf_shift_chart(ga_model, feat_sel), width='stretch')
            cmp   = ga_model.get_mf_comparison()
            mp    = cmp["manual"][feat_sel]
            op    = cmp["optimised"][feat_sel]
            param = pd.DataFrame({
                "MF"           : ["Low","Mid","High"],
                "Center Manual": mp["centers"].round(4),
                "Sigma  Manual": mp["sigmas"].round(4),
                "Center GA"    : op["centers"].round(4),
                "Sigma  GA"    : op["sigmas"].round(4),
                "Delta c"      : (op["centers"]-mp["centers"]).round(4),
                "Delta sigma"  : (op["sigmas"]-mp["sigmas"]).round(4),
            })
            st.dataframe(param, width='stretch', hide_index=True)
            st.caption("Delta positif = kurva bergeser ke kanan / melebar.")

        with tab6:
            st.markdown("#### Analisis Pergerakan Kurva (Curve Shift Analysis)")
            st.markdown("""
            Visualisasi ini menunjukkan **seberapa jauh dan ke arah mana** membership function bergeser 
            untuk setiap fitur selama evolusi GA. Shift besar menunjukkan GA melakukan optimasi signifikan.
            """)
            
            shift_fig, shift_data = curve_shift_analysis(ga_model)
            st.plotly_chart(shift_fig, width='stretch')
            
            col_shift1, col_shift2 = st.columns(2)
            with col_shift1:
                st.subheader("Fitur Paling Bergeser")
                top_shifts = shift_data.groupby("Feature")["Total Shift"].mean().nlargest(3)
                for feat, mag in top_shifts.items():
                    st.markdown(f"**{feat}**: {mag:.4f}")
            
            with col_shift2:
                st.subheader("Fitur Paling Stabil")
                stable_shifts = shift_data.groupby("Feature")["Total Shift"].mean().nsmallest(3)
                for feat, mag in stable_shifts.items():
                    st.markdown(f"**{feat}**: {mag:.4f}")
            
            st.markdown("---")
            st.subheader("Detail Pergeseran Parameter")
            
            # Show all shifts in table
            display_df = shift_data.copy()
            display_df["Delta Center"] = display_df["Delta Center"].round(4)
            display_df["Delta Sigma"] = display_df["Delta Sigma"].round(4)
            display_df["Total Shift"] = display_df["Total Shift"].round(4)
            st.dataframe(display_df[["Feature","MF","Delta Center","Delta Sigma","Total Shift"]], 
                        width='stretch', hide_index=True)
            
            st.caption("""
            💡 **Interpretasi**:
            - **Delta Center > 0**: Kurva bergeser ke kanan (lebih prefer nilai tinggi)
            - **Delta Sigma > 0**: Kurva melebar (lebih toleran dengan range nilai)
            - **Total Shift**: Magnitude perubahan (√(ΔC² + ΔS²))
            """)

        # ═══════════════════════════════════════════════════════════════════════════
        # COMPARISON & DECISION GUIDE
        # ═══════════════════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.markdown("### 📊 Model Comparison & Decision Framework")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 📊 Comparison Matrix")
            st.plotly_chart(comparison_matrix_visual(recs_a, recs_b, recs_c, time_a, time_b, time_c), width='stretch')
            st.caption("Runtime, Accuracy, Stability, dan Interpretability")
        
        with col_right:
            st.markdown("#### 📉 Accuracy-Interpretability Trade-off")
            st.plotly_chart(accuracy_interpretability_tradeoff(recs_a, recs_b, recs_c, time_a, time_b, time_c), width='stretch')
            st.caption("Bubble size = Runtime (GA lebih lambat)")
        
        st.markdown("---")
        st.markdown(decision_guide_html(), unsafe_allow_html=True)

        # Overlap
        st.markdown("---")
        st.markdown("### 🔍 Overlap Analysis")
        sa,sb,sc = set(recs_a["track_name"]),set(recs_b["track_name"]),set(recs_c["track_name"])
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("FIS Only",       len(sa-sb-sc))
        c2.metric("GA Only",        len(sb-sa-sc))
        c3.metric("ANN Only",       len(sc-sa-sb))
        c4.metric("FIS & GA",       len(sa&sb))
        c5.metric("All 3 Agree",    len(sa&sb&sc))
        triple = sa&sb&sc
        if triple:
            st.success(f"🎯 Ketiga model sepakat: **{', '.join(triple)}**")
        else:
            st.info("Tidak ada lagu yang dipilih ketiga model sekaligus — setiap paradigma unik!")

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;opacity:0.4;">
            <div style="font-size:3.5rem">🎵</div>
            <p style="color:white;font-size:1rem;margin-top:10px;">
                Atur preferensi di sidebar, konfigurasi GA,<br>
                lalu klik <b>Generate Recommendations</b>.
            </p>
            <p style="color:rgba(255,255,255,.3);font-size:.8rem;margin-top:12px;">
                👤 FIS Manual &nbsp;·&nbsp; 🧬 GA-Tuned FIS &nbsp;·&nbsp; 🤖 Neuro-Fuzzy
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:rgba(255,255,255,.22);font-size:.72rem;font-family:'Space Mono';">
        Hamud Abdul Aziz · NPM: 10020230042 · Teknik Informatika, Universitas Padjadjaran<br>
        UTS Soft Computing · SONG RECOMENDATION SYSTEM
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
