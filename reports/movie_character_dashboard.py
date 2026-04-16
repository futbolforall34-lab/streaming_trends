from pathlib import Path
from itertools import combinations
from src.utils.paths import PROJECT_ROOT

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Movie Character Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef4f8 0%, #dff3f1 100%);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .dashboard-title {
        color: #1f2937;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        color: #4b5563;
        margin-bottom: 1.2rem;
    }

    .white-card {
        background: rgba(255,255,255,0.98);
        border-radius: 22px;
        padding: 18px 20px 16px 20px;
        box-shadow: 0 10px 22px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .mini-kpi-label {
        color: #6b7280;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .mini-kpi-value {
        color: #111827;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.2rem;
    }

    .section-title {
        color: #111827;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .role-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: linear-gradient(90deg, #14b8a6 0%, #8b5cf6 100%);
        color: white;
        margin-bottom: 0.6rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.84rem;
    }

    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-title">🎬 Movie Character Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Narrative roles, interaction networks, emotional profiles, and emotional evolution across movies.</div>',
    unsafe_allow_html=True
)


# =========================================================
# CONSTANTS
# =========================================================
EMOTION_COLS = ["anger", "disgust", "fear", "sadness", "neutral", "surprise", "joy"]
DEFAULT_ARC_EMOTIONS = ["anger", "sadness", "joy", "neutral"]

CLUSTER_LABELS = {
    0: "Supporting Social Layer",
    1: "Peripheral Characters",
    2: "Core Narrative Drivers",
    3: "Emotionally Intense Characters",
    4: "Positive / Distinct Outliers"
}

CLUSTER_DESCRIPTIONS = {
    0: "Supporting characters who enrich the social structure of the story without dominating it.",
    1: "Minor or peripheral characters with limited narrative impact and low structural influence.",
    2: "Main characters with high narrative importance and strong connectivity in the interaction network.",
    3: "Conflict-driven or emotionally intense characters, often associated with tension or authority.",
    4: "Characters with unusual or distinctive emotional profiles compared with the rest of the cast."
}

CLUSTER_COLORS = {
    0: "#2DD4BF",
    1: "#60A5FA",
    2: "#A78BFA",
    3: "#F97316",
    4: "#22C55E"
}


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    processed_dir = PROJECT_ROOT / "data" / "processed"

    df_characters = pd.read_parquet(processed_dir / "final_character_dataset.parquet")
    df_dialogue_emotions = pd.read_parquet(processed_dir / "dialogue_emotion_dataset.parquet")

    return df_characters, df_dialogue_emotions


df_characters, df_dialogue_emotions = load_data()

if df_characters.empty or df_dialogue_emotions.empty:
    st.error("One or more processed datasets are empty.")
    st.stop()


# =========================================================
# HELPERS
# =========================================================
def cluster_label(x):
    if pd.isna(x):
        return "Unknown"
    return CLUSTER_LABELS.get(int(x), f"Cluster {int(x)}")


def cluster_description(x):
    if pd.isna(x):
        return "No description available."
    return CLUSTER_DESCRIPTIONS.get(int(x), "No description available.")


def dominant_emotion(row):
    vals = {emo: row.get(emo, np.nan) for emo in EMOTION_COLS}
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not vals:
        return "Unknown"
    return max(vals, key=vals.get)


def build_movie_edges(dialogue_df: pd.DataFrame) -> pd.DataFrame:
    if dialogue_df.empty:
        return pd.DataFrame(columns=["char_1", "char_2", "weight"])

    scene_groups = (
        dialogue_df.groupby(["movie_id", "segment_id"])["character_name"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="characters_present")
    )

    edge_rows = []
    for _, row in scene_groups.iterrows():
        chars = row["characters_present"]
        if len(chars) < 2:
            continue
        for c1, c2 in combinations(chars, 2):
            edge_rows.append((c1, c2))

    if not edge_rows:
        return pd.DataFrame(columns=["char_1", "char_2", "weight"])

    edges = pd.DataFrame(edge_rows, columns=["char_1", "char_2"])
    edges = (
        edges.groupby(["char_1", "char_2"])
        .size()
        .reset_index(name="weight")
        .sort_values("weight", ascending=False)
    )
    return edges


def make_network_figure(edges_weighted: pd.DataFrame, char_df_movie: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if edges_weighted.empty:
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            height=560,
            annotations=[dict(
                text="No interaction network available for this movie.",
                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
            )]
        )
        return fig

    G = nx.Graph()

    for _, row in edges_weighted.iterrows():
        G.add_edge(row["char_1"], row["char_2"], weight=row["weight"])

    for _, row in char_df_movie.iterrows():
        if row["character_name"] not in G.nodes:
            G.add_node(row["character_name"])

    pos = nx.spring_layout(G, seed=42, k=0.9)

    edge_x = []
    edge_y = []
    for u, v, _ in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="rgba(140,140,140,0.35)"),
            hoverinfo="none",
            showlegend=False
        )
    )

    lookup = char_df_movie.set_index("character_name").to_dict("index")

    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        meta = lookup.get(node, {})
        imp = float(meta.get("importance_score", 8))
        cluster = meta.get("cluster", np.nan)
        wdeg = float(meta.get("weighted_degree", 0))
        rank = meta.get("rank_in_movie", np.nan)

        node_x.append(x)
        node_y.append(y)
        node_size.append(max(18, min(55, imp / 5)))
        node_color.append(CLUSTER_COLORS.get(int(cluster), "#94A3B8") if pd.notna(cluster) else "#94A3B8")
        node_text.append(
            f"<b>{node}</b><br>"
            f"Importance: {imp:.2f}<br>"
            f"Weighted degree: {wdeg:.0f}<br>"
            f"Rank in movie: {rank}<br>"
            f"Role: {cluster_label(cluster)}"
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color="white"),
                opacity=0.92
            ),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False
        )
    )

    fig.update_layout(
        title="Interaction Network",
        template="plotly_white",
        margin=dict(l=10, r=10, t=45, b=10),
        height=560,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def make_radar_figure(row: pd.Series) -> go.Figure:
    vals = [float(row[e]) if pd.notna(row[e]) else 0.0 for e in EMOTION_COLS]
    vals_closed = vals + [vals[0]]
    labels_closed = EMOTION_COLS + [EMOTION_COLS[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=labels_closed,
        fill="toself",
        name=row["character_name"],
        line=dict(color="#14B8A6", width=3)
    ))

    fig.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(0.4, max(vals) + 0.05)])
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
        title=f"Emotional Profile — {row['character_name']}"
    )
    return fig


def make_arc_figure(dialogue_df: pd.DataFrame, character_name: str, emotions: list[str]) -> go.Figure:
    df_arc = (
        dialogue_df[dialogue_df["character_name"] == character_name]
        .groupby("scene_id")[EMOTION_COLS]
        .mean()
        .reset_index()
        .sort_values("scene_id")
    )

    fig = go.Figure()

    if df_arc.empty:
        fig.update_layout(
            template="plotly_white",
            title=f"Emotional Arc — {character_name}",
            annotations=[dict(
                text="No emotional arc available.",
                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
            )],
            height=420
        )
        return fig

    df_arc["time_step"] = range(len(df_arc))

    for emo in emotions:
        df_arc[emo] = df_arc[emo].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_arc["time_step"],
            y=df_arc[emo],
            mode="lines",
            name=emo,
            line=dict(width=3)
        ))

    fig.update_layout(
        template="plotly_white",
        title=f"Emotional Arc — {character_name}",
        xaxis_title="Narrative progression",
        yaxis_title="Emotion intensity",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def make_comparison_figure(dialogue_df: pd.DataFrame, characters: list[str], emotion_name: str) -> go.Figure:
    fig = go.Figure()

    for char in characters:
        df_arc = (
            dialogue_df[dialogue_df["character_name"] == char]
            .groupby("scene_id")[EMOTION_COLS]
            .mean()
            .reset_index()
            .sort_values("scene_id")
        )

        if df_arc.empty:
            continue

        df_arc["time_step"] = range(len(df_arc))
        df_arc[emotion_name] = df_arc[emotion_name].rolling(window=5, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=df_arc["time_step"],
            y=df_arc[emotion_name],
            mode="lines",
            name=char,
            line=dict(width=3)
        ))

    fig.update_layout(
        template="plotly_white",
        title=f"{emotion_name.capitalize()} Comparison",
        xaxis_title="Narrative progression",
        yaxis_title="Emotion intensity",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# =========================================================
# SESSION STATE
# =========================================================
movie_options = sorted(df_characters["movie_id"].dropna().unique().tolist())

if not movie_options:
    st.error("No movie_id values were found in final_character_dataset.parquet.")
    st.stop()

if "applied_movie" not in st.session_state:
    st.session_state.applied_movie = None

if "applied_character" not in st.session_state:
    st.session_state.applied_character = None

if "applied_compare" not in st.session_state:
    st.session_state.applied_compare = None

if "applied_arc_emotions" not in st.session_state:
    st.session_state.applied_arc_emotions = DEFAULT_ARC_EMOTIONS

if "applied_comparison_emotion" not in st.session_state:
    st.session_state.applied_comparison_emotion = "anger"


# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Filters")

with st.sidebar.form("filters_form"):
    selected_movie_temp = st.selectbox(
        "Movie",
        options=movie_options,
        index=None,
        placeholder="Type to search a movie..."
    )

    if selected_movie_temp is not None:
        df_movie_chars_temp = df_characters[df_characters["movie_id"] == selected_movie_temp].copy()
        character_options_temp = (
            df_movie_chars_temp.sort_values("importance_score", ascending=False)["character_name"].tolist()
        )
    else:
        character_options_temp = []

    selected_character_temp = st.selectbox(
        "Character",
        options=character_options_temp,
        index=None,
        placeholder="Select a movie first..."
    )

    compare_candidates_temp = (
        [c for c in character_options_temp if c != selected_character_temp]
        if selected_character_temp is not None else []
    )

    selected_compare_temp = st.selectbox(
        "Compare With",
        options=compare_candidates_temp,
        index=None,
        placeholder="Select a character first..."
    )

    selected_arc_emotions_temp = st.multiselect(
        "Emotions in arc",
        EMOTION_COLS,
        default=st.session_state.applied_arc_emotions
    )

    selected_comparison_emotion_temp = st.selectbox(
        "Comparison emotion",
        EMOTION_COLS,
        index=EMOTION_COLS.index(st.session_state.applied_comparison_emotion)
        if st.session_state.applied_comparison_emotion in EMOTION_COLS else 0
    )

    col_apply, col_reset = st.columns(2)
    apply_clicked = col_apply.form_submit_button("Apply Filters", use_container_width=True)
    reset_clicked = col_reset.form_submit_button("Reset Filters", use_container_width=True)

if reset_clicked:
    st.session_state.applied_movie = None
    st.session_state.applied_character = None
    st.session_state.applied_compare = None
    st.session_state.applied_arc_emotions = DEFAULT_ARC_EMOTIONS
    st.session_state.applied_comparison_emotion = "anger"
    st.rerun()

if apply_clicked:
    st.session_state.applied_movie = selected_movie_temp
    st.session_state.applied_character = selected_character_temp
    st.session_state.applied_compare = selected_compare_temp
    st.session_state.applied_arc_emotions = selected_arc_emotions_temp
    st.session_state.applied_comparison_emotion = selected_comparison_emotion_temp
    st.rerun()


# =========================================================
# ACTIVE FILTERS
# =========================================================
selected_movie = st.session_state.applied_movie

if selected_movie is None:
    st.info("Select a movie from the sidebar and click 'Apply Filters' to load the dashboard.")
    st.stop()

df_movie_chars = df_characters[df_characters["movie_id"] == selected_movie].copy()
df_movie_dialog = df_dialogue_emotions[df_dialogue_emotions["movie_id"] == selected_movie].copy()

if df_movie_chars.empty:
    st.warning(
        "No character-level data was found for the selected movie. "
        "Check whether your processed parquet files contain that movie."
    )
    st.stop()

df_movie_chars["cluster_label"] = df_movie_chars["cluster"].apply(cluster_label)
df_movie_chars["dominant_emotion"] = df_movie_chars.apply(dominant_emotion, axis=1)

character_options = (
    df_movie_chars.sort_values("importance_score", ascending=False)["character_name"].tolist()
)

if st.session_state.applied_character is None or st.session_state.applied_character not in character_options:
    selected_character = character_options[0]
else:
    selected_character = st.session_state.applied_character

compare_candidates = [c for c in character_options if c != selected_character]

if not compare_candidates:
    selected_compare = selected_character
elif st.session_state.applied_compare is None or st.session_state.applied_compare not in compare_candidates:
    selected_compare = compare_candidates[0]
else:
    selected_compare = st.session_state.applied_compare

selected_arc_emotions = st.session_state.applied_arc_emotions
comparison_emotion = st.session_state.applied_comparison_emotion

if not selected_arc_emotions:
    selected_arc_emotions = DEFAULT_ARC_EMOTIONS


# =========================================================
# KPIs
# =========================================================
main_row = df_movie_chars.sort_values("importance_score", ascending=False).iloc[0]
dominant_movie_emotion = (
    df_movie_chars[EMOTION_COLS].mean().sort_values(ascending=False).index[0]
)

n_characters = int(df_movie_chars["character_name"].nunique())
n_scenes = int(df_movie_dialog["scene_id"].nunique()) if not df_movie_dialog.empty else 0

top_kpi_cols = st.columns([1.5, 1.0, 1.0, 1.0, 1.0], gap="medium")

with top_kpi_cols[0]:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-kpi-label">Selected Movie</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-kpi-value" style="font-size:1.25rem;">{selected_movie}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Each movie loads its own interaction network, emotional profiles, clusters, and arcs.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with top_kpi_cols[1]:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-kpi-label">Characters</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-kpi-value">{n_characters}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top_kpi_cols[2]:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-kpi-label">Scenes</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-kpi-value">{n_scenes}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top_kpi_cols[3]:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-kpi-label">Main Character</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-kpi-value" style="font-size:1.25rem;">{main_row["character_name"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top_kpi_cols[4]:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-kpi-label">Dominant Emotion</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mini-kpi-value" style="font-size:1.25rem;">{dominant_movie_emotion.capitalize()}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ROW 2: NETWORK + CHARACTER PANEL + CLUSTER BAR
# =========================================================
row2_col1, row2_col2, row2_col3 = st.columns([1.8, 0.9, 0.9], gap="medium")

with row2_col1:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    edges_weighted = build_movie_edges(df_movie_dialog)
    st.plotly_chart(make_network_figure(edges_weighted, df_movie_chars), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with row2_col2:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Character Snapshot</div>', unsafe_allow_html=True)

    selected_row = df_movie_chars[df_movie_chars["character_name"] == selected_character].iloc[0]

    st.markdown(f'<div class="role-chip">{cluster_label(selected_row["cluster"])}</div>', unsafe_allow_html=True)
    st.write(cluster_description(selected_row["cluster"]))

    c1, c2 = st.columns(2)
    c1.metric("Importance", f'{selected_row["importance_score"]:.1f}')
    c2.metric("Rank", int(selected_row["rank_in_movie"]))

    c3, c4 = st.columns(2)
    c3.metric("Weighted Degree", int(selected_row["weighted_degree"]))
    c4.metric("Betweenness", f'{selected_row["betweenness"]:.3f}')

    st.markdown("**Dominant emotion**")
    st.write(dominant_emotion(selected_row).capitalize())

    st.markdown("**Top signals**")
    top_emos = (
        pd.Series({emo: selected_row[emo] for emo in EMOTION_COLS})
        .sort_values(ascending=False)
        .head(3)
    )
    for emo, val in top_emos.items():
        st.write(f"- {emo.capitalize()}: {val:.3f}")

    st.markdown('</div>', unsafe_allow_html=True)

with row2_col3:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Narrative Role Distribution</div>', unsafe_allow_html=True)

    cluster_counts = (
        df_movie_chars["cluster"].value_counts().sort_index()
        .rename_axis("cluster").reset_index(name="count")
    )
    cluster_counts["cluster_label"] = cluster_counts["cluster"].apply(cluster_label)

    fig_cluster = px.bar(
        cluster_counts,
        x="cluster_label",
        y="count",
        color="cluster_label",
        template="plotly_white",
        title=""
    )
    fig_cluster.update_layout(
        showlegend=False,
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="Characters"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ROW 3: RADAR + ARC
# =========================================================
row3_col1, row3_col2 = st.columns([1.0, 1.6], gap="medium")

with row3_col1:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.plotly_chart(make_radar_figure(selected_row), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with row3_col2:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.plotly_chart(
        make_arc_figure(df_movie_dialog, selected_character, selected_arc_emotions),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ROW 4: COMPARISON + TABLE
# =========================================================
row4_col1, row4_col2 = st.columns([1.2, 1.4], gap="medium")

with row4_col1:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.plotly_chart(
        make_comparison_figure(
            df_movie_dialog,
            [selected_character, selected_compare],
            comparison_emotion
        ),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with row4_col2:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Character Table</div>', unsafe_allow_html=True)

    table_cols = [
        "character_name",
        "importance_score",
        "rank_in_movie",
        "degree_centrality",
        "weighted_degree",
        "betweenness",
        "cluster_label",
        "dominant_emotion"
    ]

    display_df = (
        df_movie_chars[table_cols]
        .sort_values("importance_score", ascending=False)
        .rename(columns={"cluster_label": "role"})
    )

    st.dataframe(display_df, use_container_width=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)