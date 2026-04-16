from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alert_engine import AlertEngine
from stream_simulator import YouTubeTrendingStream, load_category_map
from window_engine import SlidingWindowProcessor, TumblingWindowProcessor


DATA_PATH = ROOT / "data" / "raw"
PALETTE = {"MX": "#B71020", "US": "#0056C2"}
SEV_COLORS = {
    "LOW": "#A3A3A3",
    "MEDIUM": "#FF8A47",
    "HIGH": "#0056C2",
    "CRITICAL": "#B71020",
}


st.set_page_config(
    page_title="Media Pulse | Trending Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(183,16,32,0.08), transparent 24%),
                radial-gradient(circle at top right, rgba(0,86,194,0.07), transparent 20%),
                linear-gradient(180deg, #f9f7f6 0%, #f3f2f1 100%);
            color: #2d2f2f;
        }
        .block-container {
            max-width: 1440px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stVerticalBlock"]:has(> .card-shell) {
            margin-bottom: 0;
        }
        .topbar {
            position: sticky;
            top: 0;
            z-index: 30;
            margin: -1rem 0 1.5rem 0;
            padding: 1rem 0.25rem 1rem 0.25rem;
            background: rgba(246, 246, 246, 0.74);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        .brand {
            font-size: 1.7rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #b71020;
        }
        .hero-copy {
            padding-top: 2.5rem;
            animation: riseIn 0.9s ease-out;
        }
        .live-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: #dfe8ff;
            color: #00439b;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 1.2rem;
        }
        .dot {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            background: #0056c2;
            box-shadow: 0 0 0 rgba(0,86,194,0.5);
            animation: pulseDot 1.8s infinite;
        }
        .hero-title {
            font-size: clamp(2.6rem, 5vw, 4.6rem);
            line-height: 0.98;
            letter-spacing: -0.05em;
            font-weight: 800;
            margin: 0;
            color: #202223;
        }
        .hero-title em {
            color: #b71020;
            font-style: italic;
        }
        .hero-sub {
            margin-top: 1.35rem;
            max-width: 46rem;
            color: #5a5c5c;
            font-size: 1.1rem;
            line-height: 1.7;
        }
        .hero-visual {
            position: relative;
            min-height: 32rem;
            border-radius: 2rem;
            overflow: hidden;
            background:
                radial-gradient(circle at 80% 18%, rgba(255,122,122,0.45), transparent 16%),
                radial-gradient(circle at 25% 75%, rgba(0,86,194,0.35), transparent 20%),
                linear-gradient(145deg, #07111a 0%, #132733 60%, #0e1014 100%);
            box-shadow: 0 22px 60px rgba(0,0,0,0.18);
            border: 1px solid rgba(255,255,255,0.08);
            animation: floatCard 6s ease-in-out infinite;
        }
        .hero-grid {
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
            background-size: 32px 32px;
            mask-image: linear-gradient(180deg, rgba(255,255,255,0.65), transparent 85%);
        }
        .hero-wave, .hero-wave-2 {
            position: absolute;
            left: -8%;
            right: -8%;
            height: 7px;
            border-radius: 999px;
            filter: blur(0.1px);
            transform-origin: center;
        }
        .hero-wave {
            top: 34%;
            background: linear-gradient(90deg, rgba(183,16,32,0.15), rgba(255,118,111,0.95), rgba(0,86,194,0.12));
            box-shadow: 0 0 24px rgba(255,118,111,0.28);
            transform: rotate(8deg);
        }
        .hero-wave-2 {
            top: 47%;
            background: linear-gradient(90deg, rgba(0,86,194,0.18), rgba(255,118,111,0.75), rgba(0,86,194,0.22));
            box-shadow: 0 0 22px rgba(0,86,194,0.2);
            transform: rotate(-9deg);
        }
        .hero-overlay {
            position: absolute;
            left: 1.4rem;
            right: 1.4rem;
            bottom: 1.4rem;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.15rem 1.2rem;
            border-radius: 1.25rem;
            background: rgba(246,246,246,0.9);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.4);
        }
        .hero-kicker {
            color: #767777;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
        }
        .hero-metric {
            font-size: 2rem;
            font-weight: 800;
            color: #b71020;
        }
        .hero-meta strong {
            display: block;
            font-size: 0.95rem;
            color: #2d2f2f;
        }
        .hero-meta span {
            font-size: 0.72rem;
            color: #767777;
        }
        .control-shell,
        .card-shell,
        .timeline-shell,
        .panel-shell {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(172,173,173,0.14);
            box-shadow: 0 10px 26px rgba(18,24,34,0.045);
        }
        .control-shell {
            border-radius: 1.35rem;
            padding: 0.7rem;
            margin: 1.5rem 0 1rem 0;
        }
        .card-shell {
            border-radius: 1.8rem;
            padding: 1.5rem;
            height: 100%;
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
            animation: riseIn 0.7s ease-out;
        }
        .card-shell:hover,
        .timeline-shell:hover,
        .panel-shell:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 34px rgba(18,24,34,0.07);
            border-color: rgba(183,16,32,0.14);
        }
        .timeline-shell,
        .panel-shell {
            border-radius: 2rem;
            padding: 1.8rem;
            animation: riseIn 0.8s ease-out;
        }
        .mini-label {
            color: #767777;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2.35rem;
            line-height: 1;
            font-weight: 800;
            color: #202223;
        }
        .metric-row {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
        }
        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.35rem 0.55rem;
            border-radius: 999px;
            background: rgba(0,86,194,0.08);
            color: #0056c2;
        }
        .metric-chip.danger {
            background: rgba(183,16,32,0.1);
            color: #b71020;
        }
        .metric-chip.neutral {
            background: rgba(118,119,119,0.1);
            color: #5a5c5c;
        }
        .section-title {
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0;
            color: #202223;
        }
        .section-sub {
            margin-top: 0.35rem;
            color: #6c6f6f;
            font-size: 0.92rem;
        }
        .manifest-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }
        .manifest-head,
        .manifest-row {
            display: grid;
            grid-template-columns: 1.1fr 2.6fr 1.2fr 0.55fr;
            align-items: center;
            gap: 1rem;
        }
        .manifest-head {
            padding: 0 1rem 0.6rem 1rem;
            color: #8a8c8c;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }
        .manifest-row {
            padding: 1rem;
            border-radius: 1.15rem;
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(172,173,173,0.12);
            margin-bottom: 0.65rem;
            transition: transform 180ms ease, border-color 180ms ease;
        }
        .manifest-row:hover {
            transform: translateX(3px);
            border-color: rgba(183,16,32,0.16);
        }
        .manifest-target {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-weight: 700;
            color: #202223;
        }
        .manifest-icon {
            width: 2rem;
            height: 2rem;
            border-radius: 0.7rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #ececec;
            color: #4f5252;
            font-size: 0.95rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
        }
        .pill-critical { background: rgba(183,16,32,0.12); color: #b71020; }
        .pill-high { background: rgba(0,86,194,0.12); color: #0056c2; }
        .pill-medium { background: rgba(255,138,71,0.16); color: #a34d15; }
        .pill-low { background: rgba(90,92,92,0.12); color: #5a5c5c; }
        .panel-note {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(172,173,173,0.14);
            color: #6c6f6f;
            font-size: 0.92rem;
            line-height: 1.7;
        }
        .panel-bullet {
            display: flex;
            gap: 0.8rem;
            margin-top: 1rem;
            align-items: flex-start;
        }
        .panel-line {
            width: 0.2rem;
            min-width: 0.2rem;
            height: 2.4rem;
            border-radius: 999px;
            background: linear-gradient(180deg, #b71020, #ff766f);
        }
        .download-wrap button,
        .stDownloadButton button {
            background: transparent !important;
            color: #b71020 !important;
            border: none !important;
            padding: 0 !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }
        .stSelectbox label,
        .stRadio label,
        .stSlider label {
            color: #767777 !important;
            text-transform: uppercase;
            font-size: 0.7rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.12em;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] input,
        .stRadio > div,
        .stSlider > div[data-baseweb="slider"] {
            background: #ffffff !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: 0.95rem !important;
            border: 1px solid rgba(172,173,173,0.18) !important;
            min-height: 3rem !important;
            box-shadow: none !important;
        }
        .stPlotlyChart {
            animation: riseIn 0.7s ease-out;
        }
        @keyframes pulseDot {
            0% { box-shadow: 0 0 0 0 rgba(0,86,194,0.35); }
            70% { box-shadow: 0 0 0 10px rgba(0,86,194,0); }
            100% { box-shadow: 0 0 0 0 rgba(0,86,194,0); }
        }
        @keyframes floatCard {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }
        @keyframes riseIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def enrich_df(df: pd.DataFrame, cat_map: dict[int, str], region: str) -> pd.DataFrame:
    enriched = df.copy()
    enriched["region"] = region
    enriched["category_name"] = enriched["categoryId"].map(cat_map).fillna("Unknown")
    enriched["trending_date"] = pd.to_datetime(enriched["trending_date"]).dt.tz_localize(None)
    enriched["publishedAt"] = pd.to_datetime(enriched["publishedAt"]).dt.tz_localize(None)
    enriched["days_to_trending"] = (
        enriched["trending_date"].dt.date - enriched["publishedAt"].dt.date
    ).apply(lambda delta: max(0, delta.days) if pd.notna(delta) else 0)
    return enriched


def build_tumbling_windows(df: pd.DataFrame, cat_map: dict[int, str], region: str) -> pd.DataFrame:
    stream = YouTubeTrendingStream(df, cat_map, region=region)
    processor = TumblingWindowProcessor(window_size_days=7, region=region)
    for batch_date, batch in stream.daily_batches():
        processor.add_batch(batch_date, batch)
    processor.flush()
    return processor.to_dataframe()


@st.cache_data(show_spinner=False)
def load_pipeline() -> dict[str, pd.DataFrame | dict[int, str]]:
    mx_raw = pd.read_csv(
        DATA_PATH / "MX_youtube_trending_data.csv",
        parse_dates=["publishedAt", "trending_date"],
    )
    us_raw = pd.read_csv(
        DATA_PATH / "US_youtube_trending_data.csv",
        parse_dates=["publishedAt", "trending_date"],
    )
    cat_mx = load_category_map(str(DATA_PATH / "MX_category_id.json"))
    cat_us = load_category_map(str(DATA_PATH / "US_category_id.json"))

    mx = enrich_df(mx_raw, cat_mx, "MX")
    us = enrich_df(us_raw, cat_us, "US")

    sw_mx = SlidingWindowProcessor(mx, window_days=7)
    sw_us = SlidingWindowProcessor(us, window_days=7)

    cat_vel_mx = sw_mx.category_velocity()
    cat_vel_us = sw_us.category_velocity()
    cat_views_mx = sw_mx.category_avg_views()
    cat_views_us = sw_us.category_avg_views()
    ch_freq_mx = sw_mx.channel_frequency(top_n=50)
    ch_freq_us = sw_us.channel_frequency(top_n=50)

    engine = AlertEngine(z_threshold=2.0, baseline_days=30, cooldown_days=3)
    engine.run_backtest(cat_vel_mx, entity_type="category", metric="trending_velocity", region="MX")
    engine.run_backtest(cat_views_mx, entity_type="category", metric="avg_views", region="MX")
    engine.run_backtest(cat_vel_us, entity_type="category", metric="trending_velocity", region="US")
    engine.run_backtest(cat_views_us, entity_type="category", metric="avg_views", region="US")
    engine.run_backtest(ch_freq_mx, entity_type="channel", metric="trending_frequency", region="MX")
    engine.run_backtest(ch_freq_us, entity_type="channel", metric="trending_frequency", region="US")

    alerts_df = engine.alerts_df()
    alerts_df["alert_date"] = pd.to_datetime(alerts_df["alert_date"])
    alerts_df["display_time"] = alerts_df["alert_date"].dt.strftime("%Y-%m-%d")

    tw_mx = build_tumbling_windows(mx, cat_mx, "MX")
    tw_us = build_tumbling_windows(us, cat_us, "US")

    return {
        "mx": mx,
        "us": us,
        "cat_mx": cat_mx,
        "cat_us": cat_us,
        "cat_vel_mx": cat_vel_mx,
        "cat_vel_us": cat_vel_us,
        "cat_views_mx": cat_views_mx,
        "cat_views_us": cat_views_us,
        "ch_freq_mx": ch_freq_mx,
        "ch_freq_us": ch_freq_us,
        "alerts_df": alerts_df,
        "tw_mx": tw_mx,
        "tw_us": tw_us,
    }


def style_figure(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2d2f2f", family="Manrope, Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        tickfont=dict(color="#767777"),
        title=None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(118,119,119,0.12)",
        zeroline=False,
        tickfont=dict(color="#767777"),
    )
    return fig


def resolve_regions(region_mode: str) -> list[str]:
    return {"MX + US": ["MX", "US"], "Mexico (MX)": ["MX"], "United States (US)": ["US"]}[region_mode]


def resolve_severities(mode: str) -> list[str]:
    mapping = {
        "All Severities": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "Critical Only": ["CRITICAL"],
        "High + Critical": ["HIGH", "CRITICAL"],
        "Medium+": ["MEDIUM", "HIGH", "CRITICAL"],
    }
    return mapping[mode]


def resolve_start_date(max_date: pd.Timestamp, time_mode: str) -> pd.Timestamp:
    offsets = {
        "Full History": 3650,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 30 Weeks": 210,
        "Last 52 Weeks": 364,
    }
    return max_date - pd.Timedelta(days=offsets[time_mode])


def filter_alerts(alerts_df: pd.DataFrame, regions: list[str], severities: list[str], start_date: pd.Timestamp) -> pd.DataFrame:
    return alerts_df[
        alerts_df["region"].isin(regions)
        & alerts_df["severity"].isin(severities)
        & (alerts_df["alert_date"] >= start_date)
    ].copy()


def top_velocity_snapshot(pipeline: dict, regions: list[str], start_date: pd.Timestamp, top_n: int = 4) -> list[dict]:
    rows: list[dict] = []
    multi_region = len(regions) > 1
    for region in regions:
        vel_df = pipeline[f"cat_vel_{region.lower()}"]
        vel_df = vel_df[vel_df.index >= start_date]
        for category in vel_df.columns:
            series = vel_df[category].dropna()
            if len(series) < 8:
                continue
            current = float(series.iloc[-1])
            baseline = float(series.iloc[:-1].tail(30).mean()) if len(series) > 1 else current
            if abs(baseline) < 1e-9:
                continue
            pct_change = ((current - baseline) / abs(baseline)) * 100
            rows.append(
                {
                    "label": f"{category} ({region})" if multi_region else category,
                    "category": category,
                    "region": region,
                    "pct_change": pct_change,
                    "current": current,
                    "baseline": baseline,
                }
            )

    rows = sorted(rows, key=lambda row: row["pct_change"], reverse=True)
    positive = [row for row in rows if row["pct_change"] > 0]
    return (positive or rows)[:top_n]


def render_velocity_bars(snapshot: list[dict]) -> str:
    if not snapshot:
        return "<p style='color:#767777;'>No velocity data for this window.</p>"
    max_pct = max(max(item["pct_change"], 1) for item in snapshot)
    blocks = []
    for idx, item in enumerate(snapshot):
        width = max(10, min(100, (max(item["pct_change"], 0) / max_pct) * 100))
        color = "#B71020" if idx == 0 else "#0056C2" if idx < 3 else "#acadad"
        gradient = (
            "linear-gradient(90deg, #b71020, #ff766f)"
            if idx == 0
            else "linear-gradient(90deg, #0056c2, #7baeff)"
            if idx < 3
            else "linear-gradient(90deg, #8f9393, #c2c5c5)"
        )
        blocks.append(
            f"""
            <div style="margin-bottom:1.55rem;">
                <div style="display:flex;justify-content:space-between;gap:1rem;margin-bottom:0.45rem;">
                    <span style="font-size:0.96rem;font-weight:800;color:#202223;">{item['label']}</span>
                    <span style="font-size:0.9rem;font-weight:800;color:{color};">{item['pct_change']:+.0f}%</span>
                </div>
                <div style="height:0.9rem;background:#ececec;border-radius:999px;overflow:hidden;">
                    <div style="height:100%;width:{width:.0f}%;background:{gradient};border-radius:999px;box-shadow:0 0 20px rgba(183,16,32,0.12);"></div>
                </div>
            </div>
            """
        )
    return "".join(blocks)


def severity_donut(alerts_df: pd.DataFrame) -> go.Figure:
    sev_counts = alerts_df["severity"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=sev_counts.index,
                values=sev_counts.values,
                hole=0.66,
                marker=dict(colors=[SEV_COLORS[s] for s in sev_counts.index], line=dict(color="#ffffff", width=3)),
                sort=False,
                direction="clockwise",
                textinfo="none",
            )
        ]
    )
    total = int(sev_counts.sum())
    critical_share = round((sev_counts.get("CRITICAL", 0) / total) * 100) if total else 0
    fig.add_annotation(text=f"<b style='font-size:30px'>{critical_share}%</b><br><span style='font-size:11px;color:#767777'>CRITICAL MASS</span>", showarrow=False)
    return style_figure(fig, 310)


def timeline_chart(alerts_df: pd.DataFrame) -> go.Figure:
    if alerts_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No alerts in selected window", showarrow=False, font=dict(size=16, color="#767777"))
        return style_figure(fig, 300)

    timeline = alerts_df.groupby("alert_date").agg(count=("alert_id", "count"), peak_z=("z_score", "max")).reset_index()
    peak_idx = timeline["count"].idxmax()
    peak_row = timeline.loc[peak_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["alert_date"],
            y=timeline["count"],
            mode="lines+markers",
            line=dict(color="#1d2db6", width=3, shape="spline", smoothing=1.25),
            marker=dict(size=timeline["peak_z"].clip(2, 7) * 2.4, color="#5d98f3", line=dict(color="#dfe8ff", width=6)),
            hovertemplate="%{x|%Y-%m-%d}<br>Alerts=%{y}<extra></extra>",
            name="Signal flow",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_row["alert_date"]],
            y=[peak_row["count"]],
            mode="markers",
            marker=dict(size=26, color="#b71020", line=dict(color="#f4b7bc", width=10)),
            name="Peak",
            hovertemplate="Peak<br>%{x|%Y-%m-%d}<br>Alerts=%{y}<extra></extra>",
        )
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    return style_figure(fig, 320)


def replay_chart(pipeline: dict, region: str, start_date: pd.Timestamp) -> tuple[go.Figure, pd.DataFrame]:
    vel_df = pipeline[f"cat_vel_{region.lower()}"]
    vel_df = vel_df[vel_df.index >= start_date]

    engine = AlertEngine(z_threshold=2.0, baseline_days=30, cooldown_days=3)
    engine.run_backtest(vel_df, entity_type="category", metric="trending_velocity", region=region)
    replay_alerts = engine.alerts_df()
    replay_alerts["alert_date"] = pd.to_datetime(replay_alerts["alert_date"])

    fig = go.Figure()
    top_categories = vel_df.mean().sort_values(ascending=False).head(3).index.tolist()
    palette = ["#b71020", "#0056c2", "#ff8a47"]
    for idx, category in enumerate(top_categories):
        fig.add_trace(
            go.Scatter(
                x=vel_df.index,
                y=vel_df[category],
                mode="lines",
                name=category,
                line=dict(width=2.4, color=palette[idx % len(palette)]),
                hovertemplate="%{x|%Y-%m-%d}<br>" + category + "<br>Velocity=%{y:.2f}<extra></extra>",
            )
        )

    if not replay_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=replay_alerts["alert_date"],
                y=replay_alerts["current_value"],
                mode="markers",
                name="Alerts",
                marker=dict(
                    size=replay_alerts["z_score"].clip(2, 8) * 2.3,
                    color=replay_alerts["severity"].map(SEV_COLORS),
                    line=dict(color="#ffffff", width=1.4),
                ),
                text=replay_alerts["entity_name"] + " | " + replay_alerts["severity"],
                hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>Value=%{y:.2f}<extra></extra>",
            )
        )

    fig.update_yaxes(title="Velocity", title_font=dict(color="#767777"))
    return style_figure(fig, 250), replay_alerts.sort_values("alert_date", ascending=False)


def metric_card(label: str, value: str, chip: str, chip_class: str = "") -> str:
    chip_html = f'<div class="metric-chip {chip_class}">{chip}</div>' if chip else ""
    return f"""
    <div class="card-shell">
        <div class="mini-label">{label}</div>
        <div class="metric-row">
            <div class="metric-value">{value}</div>
            {chip_html}
        </div>
    </div>
    """


def render_manifest_rows(latest_alerts: pd.DataFrame) -> str:
    if latest_alerts.empty:
        return '<div class="manifest-row"><div>No alerts</div><div>No signals in the selected window.</div><div></div><div></div></div>'

    icon_map = {"category": "●", "channel": "◉"}
    rows = []
    for _, row in latest_alerts.head(8).iterrows():
        sev = row["severity"].lower()
        rows.append(
            f"""
            <div class="manifest-row">
                <div style="font-size:0.9rem;font-weight:700;color:#444;">{row['alert_date'].strftime('%Y-%m-%d')}</div>
                <div class="manifest-target">
                    <div class="manifest-icon">{icon_map.get(row['entity_type'], '•')}</div>
                    <div>
                        <div>{row['entity_name']}</div>
                        <div style="font-size:0.74rem;color:#8a8c8c;font-weight:600;">{row['entity_type'].title()} | {row['region']} | {row['metric']}</div>
                    </div>
                </div>
                <div><span class="status-pill pill-{sev}">{row['severity']}</span></div>
                <div style="text-align:right;color:#9a9c9c;font-size:1.1rem;">›</div>
            </div>
            """
        )
    return "".join(rows)


def topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">Media Pulse</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_block(peak_item: dict | None, latest_alert_date: str) -> None:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero-copy">
                <div class="live-chip"><span class="dot"></span> LIVE SIGNAL DETECTION</div>
                <h1 class="hero-title">YouTube <em>Trending Signals</em><br>Report</h1>
                <p class="hero-sub">
                    Deciphering the kinetic energy of content movements with real pipeline data.
                    This dashboard monitors category velocity, highlights statistical outliers and surfaces
                    the moments where momentum becomes an operational signal.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        peak_value = f"{peak_item['pct_change']:+.0f}%" if peak_item else "-"
        peak_label = peak_item["label"] if peak_item else "No dominant signal"
        st.markdown(
            f"""
            <div class="hero-visual">
                <div class="hero-grid"></div>
                <div class="hero-wave"></div>
                <div class="hero-wave-2"></div>
                <div class="hero-overlay">
                    <div>
                        <div class="hero-kicker">Peak Momentum</div>
                        <div class="hero-metric">{peak_value}</div>
                    </div>
                    <div class="hero-meta" style="text-align:right;">
                        <div class="hero-kicker">Live Category</div>
                        <strong>{peak_label}</strong>
                        <span>Updated {latest_alert_date}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def control_bar(max_date: pd.Timestamp) -> tuple[list[str], list[str], pd.Timestamp, str]:
    st.markdown('<div class="control-shell">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.1, 1.2], gap="small")
    with c1:
        region_mode = st.selectbox("Region", ["MX + US", "Mexico (MX)", "United States (US)"], index=0)
    with c2:
        severity_mode = st.selectbox("Severity", ["All Severities", "Medium+", "High + Critical", "Critical Only"], index=0)
    with c3:
        time_mode = st.selectbox("Time Window", ["Last 30 Weeks", "Last 90 Days", "Last 30 Days", "Last 52 Weeks", "Full History"], index=0)
    with c4:
        replay_region = st.selectbox("Replay Region", ["MX", "US"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    return resolve_regions(region_mode), resolve_severities(severity_mode), resolve_start_date(max_date, time_mode), replay_region


def main() -> None:
    inject_styles()
    topbar()

    with st.spinner("Loading live signal engine from historical stream..."):
        pipeline = load_pipeline()

    max_date = max(pipeline["mx"]["trending_date"].max(), pipeline["us"]["trending_date"].max())
    regions, severities, start_date, replay_region = control_bar(max_date)
    filtered_alerts = filter_alerts(pipeline["alerts_df"], regions, severities, start_date)
    snapshot = top_velocity_snapshot(pipeline, regions, start_date, top_n=4)
    peak_item = snapshot[0] if snapshot else None
    latest_alert_date = filtered_alerts["alert_date"].max().strftime("%Y-%m-%d") if not filtered_alerts.empty else max_date.strftime("%Y-%m-%d")

    hero_block(peak_item, latest_alert_date)

    total_events = sum(len(pipeline[key]) for key in ["mx", "us"] if key.upper() in regions)
    critical_count = int(filtered_alerts["severity"].eq("CRITICAL").sum())
    latest_signal = peak_item["category"] if peak_item else "No signal"
    latest_delta = latest_alert_date

    k1, k2, k3 = st.columns(3, gap="large")
    with k1:
        st.markdown(metric_card("Events Analyzed", f"{total_events:,}", f"{len(regions)} region(s)"), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card("Critical Alerts", f"{critical_count:,}", "Active", "danger"), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card("Latest Signal", latest_signal, latest_delta, "neutral"), unsafe_allow_html=True)

    upper_left, upper_right = st.columns([1.85, 1], gap="large")
    with upper_left:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Category Velocity</h2><p class="section-sub">Normalized momentum across the most accelerated categories in the selected window.</p>', unsafe_allow_html=True)
        st.markdown(render_velocity_bars(snapshot), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with upper_right:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Severity Mix</h2><p class="section-sub">Signal concentration by impact level.</p>', unsafe_allow_html=True)
        st.plotly_chart(severity_donut(filtered_alerts), use_container_width=True, config={"displayModeBar": False})
        summary_col_1, summary_col_2 = st.columns(2)
        with summary_col_1:
            st.markdown(metric_card("Alert Signals", f"{len(filtered_alerts):,}", "Detects"), unsafe_allow_html=True)
        with summary_col_2:
            medium_plus = int(filtered_alerts["severity"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum())
            st.markdown(metric_card("Observation", f"{medium_plus:,}", "Escalations"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="timeline-shell">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:end;gap:1rem;flex-wrap:wrap;">', unsafe_allow_html=True)
    st.markdown('<div><div class="mini-label" style="display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;background:#ffd8d8;color:#b71020;margin-bottom:0.8rem;">TIMELINE PULSE</div><h2 class="section-title" style="font-size:2.2rem;">Alert Intelligence Stream</h2></div>', unsafe_allow_html=True)
    active_hours = f"{start_date.strftime('%Y-%m-%d')} -> {max_date.strftime('%Y-%m-%d')}"
    st.markdown(f'<div style="text-align:right;"><div class="mini-label">Active Signal Window</div><div style="font-weight:800;color:#202223;">{active_hours}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.plotly_chart(timeline_chart(filtered_alerts), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    left_panel, right_panel = st.columns([1.65, 1], gap="large")

    with left_panel:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        head_left, head_right = st.columns([1.4, 0.6])
        with head_left:
            st.markdown('<div class="manifest-title">Active Signal Manifest</div>', unsafe_allow_html=True)
        with head_right:
            export_df = filtered_alerts.sort_values(["alert_date", "z_score"], ascending=[False, False]).head(50)
            st.download_button(
                "Download CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="media_pulse_alerts.csv",
                mime="text/csv",
            )

        st.markdown(
            '<div class="manifest-head"><div>Timestamp</div><div>Target / Topic</div><div>Velocity</div><div>Status</div></div>',
            unsafe_allow_html=True,
        )
        latest_manifest = filtered_alerts.sort_values(["alert_date", "z_score"], ascending=[False, False])[
            ["alert_date", "entity_type", "entity_name", "region", "severity", "metric", "z_score"]
        ]
        st.markdown(render_manifest_rows(latest_manifest), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_panel:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title" style="font-size:1.4rem;">Trend Replay: Velocity Spike</h2>', unsafe_allow_html=True)
        replay_fig, replay_alerts = replay_chart(pipeline, replay_region, start_date)
        st.plotly_chart(replay_fig, use_container_width=True, config={"displayModeBar": False})
        lead_text = replay_alerts.iloc[0]["entity_name"] if not replay_alerts.empty else "No active replay spike"
        lead_severity = replay_alerts.iloc[0]["severity"] if not replay_alerts.empty else "-"
        st.markdown(
            f"""
            <div class="panel-bullet">
                <div class="panel-line"></div>
                <div>
                    <div style="font-weight:800;color:#202223;">Signal focus: {lead_text}</div>
                    <div style="font-size:0.72rem;color:#8a8c8c;font-weight:700;">Replay region {replay_region} | Highest visible severity: {lead_severity}</div>
                </div>
            </div>
            <div class="panel-note">
                This replay replaces the decorative video block with a real operational visual. It shows the live window of category velocity and overlays the alerts actually emitted by the engine.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
