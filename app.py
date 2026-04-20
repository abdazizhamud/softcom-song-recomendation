"""
app.py — The Intelligence Battle: Human Expert vs. Evolutionary Tuning & Neuro-Fuzzy
Streamlit GUI yang menggabungkan FIS Manual (Mamdani) dan Neuro-Fuzzy (ANN-based)
untuk sistem rekomendasi lagu berbasis konten.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")

# ── Matplotlib / Plotly ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Model backend ────────────────────────────────────────────────────────────
from models.fuzzy_manual import ManualFuzzySystem
from models.neuro_fuzzy import NeuroFuzzySystem
from data.loader import load_dataset

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="The Intelligence Battle 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

/* Dark gradient background */
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}

/* Metric box */
.metric-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(168,85,247,0.3));
    border: 1px solid rgba(168,85,247,0.4);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    color: white;
}

/* Team A / Team B headers */
.team-a { color: #60a5fa; font-family: 'Space Mono', monospace; font-size: 1.1rem; }
.team-b { color: #f472b6; font-family: 'Space Mono', monospace; font-size: 1.1rem; }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
}
.badge-a { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid #60a5fa; }
.badge-b { background: rgba(244,114,182,0.2); color: #f472b6; border: 1px solid #f472b6; }

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    width: 100%;
    transition: all 0.3s;
    cursor: pointer;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.5);
}

/* Slider label */
.slider-label {
    color: rgba(255,255,255,0.7);
    font-size: 0.85rem;
    margin-bottom: 4px;
}

/* Table override */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Divider */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 24px 0; }

/* Winner banner */
.winner { 
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: #1f2937;
    border-radius: 12px;
    padding: 12px 20px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS  (cached)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()

@st.cache_resource(show_spinner=False)
def get_models(df):
    fuzzy = ManualFuzzySystem()
    nf    = NeuroFuzzySystem()
    nf.train(df)
    return fuzzy, nf


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
FEATURES = ["danceability", "energy", "valence", "speechiness", "acousticness"]

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert hex color to rgba string with opacity."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def radar_chart(recs_a: pd.DataFrame, recs_b: pd.DataFrame, user_prefs: dict):
    """Radar chart comparing mean feature profile of both recommendation sets."""
    cats = FEATURES
    N    = len(cats)

    def avg(df):
        return [df[c].mean() for c in cats]

    vals_a    = avg(recs_a)
    vals_b    = avg(recs_b)
    vals_user = [user_prefs.get(c, 0.5) for c in cats]

    fig = go.Figure()
    for name, vals, color, fill in [
        ("👤 Human Expert (FIS)",    vals_a,    "#60a5fa", "toself"),
        ("🤖 Neuro-Fuzzy (ANN)",      vals_b,    "#f472b6", "toself"),
        ("🎯 Your Preference",        vals_user, "#fbbf24", "toself"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill=fill,
            name=name,
            line_color=color,
            fillcolor=hex_to_rgba(color, alpha=0.15),
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.1)",
                            linecolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             linecolor="rgba(255,255,255,0.2)"),
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400,
    )
    return fig


def score_bar_chart(recs_a: pd.DataFrame, recs_b: pd.DataFrame):
    """Bar chart comparing recommendation scores side by side."""
    labels_a = [f"{r['track_name'][:22]}…" if len(r['track_name']) > 22
                else r['track_name'] for _, r in recs_a.iterrows()]
    labels_b = [f"{r['track_name'][:22]}…" if len(r['track_name']) > 22
                else r['track_name'] for _, r in recs_b.iterrows()]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("👤 Human Expert (FIS)", "🤖 Neuro-Fuzzy (ANN)"),
                        horizontal_spacing=0.08)

    fig.add_trace(go.Bar(
        x=recs_a["match_score"].values,
        y=labels_a,
        orientation="h",
        marker=dict(
            color=recs_a["match_score"].values,
            colorscale=[[0, "#1d4ed8"], [1, "#60a5fa"]],
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=[f"{s:.1f}" for s in recs_a["match_score"].values],
        textposition="outside",
        textfont=dict(color="white"),
        name="FIS Score",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=recs_b["match_score"].values,
        y=labels_b,
        orientation="h",
        marker=dict(
            color=recs_b["match_score"].values,
            colorscale=[[0, "#9d174d"], [1, "#f472b6"]],
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=[f"{s:.1f}" for s in recs_b["match_score"].values],
        textposition="outside",
        textfont=dict(color="white"),
        name="ANN Score",
    ), row=1, col=2)

    fig.update_xaxes(range=[0, 105], showgrid=False,
                     color="rgba(255,255,255,0.5)")
    fig.update_yaxes(showgrid=False, color="rgba(255,255,255,0.7)")
    fig.update_annotations(font=dict(color="white", size=13))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def feature_heatmap(recs_a: pd.DataFrame, recs_b: pd.DataFrame):
    """Heatmap of features for both recommendation sets."""
    cols = FEATURES + ["popularity"]
    da = recs_a[cols].copy()
    db = recs_b[cols].copy()
    da.index = [f"FIS #{i+1}" for i in range(len(da))]
    db.index = [f"ANN #{i+1}" for i in range(len(db))]
    combined = pd.concat([da, db])

    # Normalize popularity to 0-1
    combined["popularity"] = combined["popularity"] / 100.0

    fig = px.imshow(
        combined.values,
        labels=dict(x="Feature", y="Song", color="Value"),
        x=cols,
        y=combined.index.tolist(),
        color_continuous_scale="Viridis",
        zmin=0, zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        coloraxis_colorbar=dict(tickcolor="white", title=dict(text="Value", font=dict(color="white"))),
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(color="rgba(255,255,255,0.7)")
    fig.update_yaxes(color="rgba(255,255,255,0.7)")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:3rem">🎵</div>
        <h1 style="color:white; margin:0; font-size:2rem; letter-spacing:2px;">
            THE INTELLIGENCE BATTLE
        </h1>
        <p style="color:rgba(255,255,255,0.5); font-family:'DM Sans'; margin-top:6px;">
            Human Expert FIS  ·  vs  ·  Neuro-Fuzzy ANN  |  Music Recommender
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data & models ───────────────────────────────────────────────────
    with st.spinner("🔄 Loading dataset & initialising models…"):
        df = get_dataset()
        fuzzy_model, nf_model = get_models(df)

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Preference Panel")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        energy       = st.slider("⚡ Energy",       0.0, 1.0, 0.7, 0.01)
        danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.7, 0.01)
        valence      = st.slider("😊 Valence (Mood)", 0.0, 1.0, 0.6, 0.01)
        speechiness  = st.slider("🎤 Speechiness",  0.0, 1.0, 0.1, 0.01)
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.2, 0.01)
        popularity   = st.slider("🌟 Popularity",   0,   100,  70,    1)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Generate Recommendations")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:rgba(255,255,255,0.4); font-size:0.75rem;">
        Dataset: <b style="color:rgba(255,255,255,0.7)">{len(df):,} songs</b><br>
        Model A: <b style="color:#60a5fa">Mamdani FIS</b><br>
        Model B: <b style="color:#f472b6">Neuro-Fuzzy (ANN)</b>
        </div>
        """, unsafe_allow_html=True)

    user_prefs = {
        "energy":       energy,
        "danceability": danceability,
        "valence":      valence,
        "speechiness":  speechiness,
        "acousticness": acousticness,
        "popularity":   popularity / 100.0,
    }

    # ── Dataset Preview ──────────────────────────────────────────────────────
    with st.expander("📊 Dataset Preview (first 10 rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # ── Run recommendations ──────────────────────────────────────────────────
    if run_btn:
        col_a, col_b = st.columns(2)

        # ── Model A: FIS ─────────────────────────────────────────────────────
        with col_a:
            st.markdown('<p class="team-a">👤 Model A — Human Expert FIS</p>',
                        unsafe_allow_html=True)
            with st.spinner("Running Mamdani FIS…"):
                t0 = time.perf_counter()
                recs_a = fuzzy_model.recommend(df, user_prefs, top_n=5)
                time_a = time.perf_counter() - t0

            st.markdown(f"""
            <div class="metric-box" style="background:linear-gradient(135deg,rgba(37,99,235,0.3),rgba(96,165,250,0.2));">
                ⏱️ Execution Time: <b>{time_a*1000:.1f} ms</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Table
            display_a = recs_a[["track_name", "artists", "match_score",
                                  "energy", "danceability", "valence"]].copy()
            display_a.columns = ["Track", "Artist", "Score", "Energy", "Dance", "Valence"]
            display_a["Score"] = display_a["Score"].round(1)
            st.dataframe(display_a.reset_index(drop=True), use_container_width=True,
                         hide_index=True)

        # ── Model B: Neuro-Fuzzy ──────────────────────────────────────────────
        with col_b:
            st.markdown('<p class="team-b">🤖 Model B — Neuro-Fuzzy (ANN)</p>',
                        unsafe_allow_html=True)
            with st.spinner("Running Neuro-Fuzzy ANN…"):
                t0 = time.perf_counter()
                recs_b = nf_model.recommend(df, user_prefs, top_n=5)
                time_b = time.perf_counter() - t0

            st.markdown(f"""
            <div class="metric-box" style="background:linear-gradient(135deg,rgba(157,23,77,0.3),rgba(244,114,182,0.2));">
                ⏱️ Execution Time: <b>{time_b*1000:.1f} ms</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            display_b = recs_b[["track_name", "artists", "match_score",
                                  "energy", "danceability", "valence"]].copy()
            display_b.columns = ["Track", "Artist", "Score", "Energy", "Dance", "Valence"]
            display_b["Score"] = display_b["Score"].round(1)
            st.dataframe(display_b.reset_index(drop=True), use_container_width=True,
                         hide_index=True)

        # ── Winner banner ─────────────────────────────────────────────────────
        avg_a = recs_a["match_score"].mean()
        avg_b = recs_b["match_score"].mean()
        if avg_a > avg_b:
            winner = "👤 Human Expert FIS wins this round!"
        elif avg_b > avg_a:
            winner = "🤖 Neuro-Fuzzy ANN wins this round!"
        else:
            winner = "🤝 It's a tie!"

        st.markdown(f'<div class="winner">🏆 {winner} &nbsp;|&nbsp; Avg Score FIS: {avg_a:.1f} · ANN: {avg_b:.1f}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Visualizations ────────────────────────────────────────────────────
        st.markdown("### 📈 Comparison Visualizations")

        tab1, tab2, tab3 = st.tabs(["🕸️ Radar Chart", "📊 Score Bars", "🌡️ Feature Heatmap"])

        with tab1:
            st.plotly_chart(radar_chart(recs_a, recs_b, user_prefs),
                            use_container_width=True)
            st.caption("Radar chart: mean feature profile of recommended songs vs your preference")

        with tab2:
            st.plotly_chart(score_bar_chart(recs_a, recs_b),
                            use_container_width=True)
            st.caption("Match scores for top-5 songs from each model (0–100)")

        with tab3:
            st.plotly_chart(feature_heatmap(recs_a, recs_b),
                            use_container_width=True)
            st.caption("Heatmap of audio features across all 10 recommended songs")

        # ── Overlap analysis ──────────────────────────────────────────────────
        st.markdown("### 🔍 Overlap Analysis")
        set_a = set(recs_a["track_name"].values)
        set_b = set(recs_b["track_name"].values)
        overlap = set_a & set_b
        c1, c2, c3 = st.columns(3)
        c1.metric("FIS Unique Songs", len(set_a - set_b))
        c2.metric("Shared Songs", len(overlap))
        c3.metric("ANN Unique Songs", len(set_b - set_a))
        if overlap:
            st.success(f"Both models agree on: **{', '.join(overlap)}**")
        else:
            st.info("No overlap — the two models produced completely different recommendations!")

    else:
        # ── Landing state ─────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; opacity:0.5;">
            <div style="font-size:4rem">🎵</div>
            <p style="color:white; font-size:1.1rem;">
                Set your preferences in the sidebar and click<br>
                <b>Generate Recommendations</b> to start the battle.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; font-family:'Space Mono';">
        Hamud Abdul Aziz — Teknik Informatika, Universitas Padjadjaran<br>
        UTS Soft Computing · The Intelligence Battle
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
