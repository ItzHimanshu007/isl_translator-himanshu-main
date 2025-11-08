import streamlit as st
from typing import Iterable, List, Dict, Optional

BASE_STYLE = """
<style>
:root {
    --bg-dark: #05070f;
    --bg-panel: #0e1424;
    --border-muted: rgba(255, 255, 255, 0.08);
    --brand: #5e81ff;
    --brand-soft: rgba(94, 129, 255, 0.18);
    --text-muted: #a7b0c9;
}
body {
    background-color: var(--bg-dark);
    color: #f5f6fb;
    font-family: "Segoe UI", "Inter", sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 20%, rgba(94, 129, 255, 0.1), transparent 25%), var(--bg-dark);
}
.stAppHeader {
    background: transparent;
}
main .block-container {
    padding: 2rem 3rem 3rem 3rem;
}
[data-testid="stSidebar"] {
    background: #080b18;
    border-right: 1px solid var(--border-muted);
}
.stButton>button, .stDownloadButton>button {
    background: var(--brand-soft);
    color: #f5f6fb;
    border: 1px solid transparent;
    border-radius: 0.6rem;
    padding: 0.55rem 1rem;
    font-weight: 600;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--brand);
}
.neuronest-hero {
    background: linear-gradient(120deg, rgba(94, 129, 255, 0.18), rgba(9, 13, 32, 0.95));
    border: 1px solid var(--border-muted);
    border-radius: 1.2rem;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: var(--brand-soft);
    color: #dfe6ff;
    font-size: 0.75rem;
    margin-bottom: 0.6rem;
}
.neuronest-hero h1 {
    margin: 0;
    font-size: 2.4rem;
}
.neuronest-hero p {
    margin: 0.35rem 0 0 0;
    color: var(--text-muted);
}
.section-label {
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.5rem;
}
.section-label + h2 {
    margin-top: 0.15rem;
}
.section-caption {
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}
.stat-card, .info-card, .feature-card, .form-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-muted);
    border-radius: 1rem;
    padding: 1.2rem 1.4rem;
}
.stat-card h3 {
    margin: 0.35rem 0 0 0;
    font-size: 1.9rem;
}
.stat-card span {
    color: var(--text-muted);
    font-size: 0.9rem;
}
.stat-delta {
    color: #7ddba3;
    font-size: 0.85rem;
}
.feature-card h3 {
    margin: 0.2rem 0;
}
.feature-tags span {
    display: inline-block;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 0.6rem;
    padding: 0.1rem 0.45rem;
    font-size: 0.75rem;
    margin-right: 0.25rem;
    color: var(--text-muted);
}
.info-card h4 {
    margin: 0;
}
.info-card p {
    margin: 0.35rem 0 0 0;
    color: var(--text-muted);
}
.checklist-card {
    background: rgba(15, 20, 38, 0.8);
    border: 1px solid var(--border-muted);
    border-radius: 1rem;
    padding: 1.2rem 1.4rem;
}
.checklist-card li {
    margin-bottom: 0.35rem;
}
.form-card label {
    color: #c6cde3;
}
</style>
"""


def inject_base_styles():
    """Apply the shared Neuronest theme to the current Streamlit page."""
    st.markdown(BASE_STYLE, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, caption: Optional[str] = None, badge: Optional[str] = None):
    badge_html = f"<div class='hero-badge'>{badge}</div>" if badge else ""
    caption_html = f"<p>{caption}</p>" if caption else ""
    st.markdown(
        f"""
        <div class="neuronest-hero">
            {badge_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, caption: Optional[str] = None):
    st.markdown("<div class='section-label'>Neuronest</div>", unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if caption:
        st.markdown(f"<p class='section-caption'>{caption}</p>", unsafe_allow_html=True)


def render_stat_cards(stats: Iterable[Dict[str, str]]):
    stats_list = list(stats)
    cols = st.columns(len(stats_list))
    for col, stat in zip(cols, stats_list):
        delta_html = f"<div class='stat-delta'>{stat.get('delta','')}</div>" if stat.get("delta") else ""
        col.markdown(
            f"""
            <div class="stat-card">
                <span>{stat['label']}</span>
                <h3>{stat['value']}</h3>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_cards(features: List[Dict[str, str]]):
    cols = st.columns(len(features))
    for col, feature in zip(cols, features):
        with col:
            tags = "".join(f"<span>{tag}</span>" for tag in feature.get("tags", []))
            col.markdown(
                f"""
                <div class="feature-card">
                    <span class="section-label">{feature.get('label', 'Module')}</span>
                    <h3>{feature['title']}</h3>
                    <p>{feature['description']}</p>
                    <div class="feature-tags">{tags}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(feature["cta"], key=feature["title"]):
                st.switch_page(feature["page"])


def render_info_cards(cards: List[Dict[str, str]], columns: int = 3):
    cols = st.columns(columns)
    for idx, card in enumerate(cards):
        with cols[idx % columns]:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>{card['title']}</h4>
                    <p>{card['body']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_checklist(title: str, items: Iterable[str]):
    items_html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="checklist-card">
            <strong>{title}</strong>
            <ul>
                {items_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
