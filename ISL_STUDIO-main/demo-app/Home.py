import streamlit as st
from ui_components import (
    inject_base_styles,
    render_hero,
    render_section_heading,
    render_stat_cards,
    render_feature_cards,
    render_info_cards,
    render_checklist,
)

st.set_page_config(page_title="Neuronest ISL Studio", page_icon="Neuronest", layout="wide")
inject_base_styles()

FEATURES = [
    {
        "label": "Module",
        "title": "ISL to Text",
        "description": "Live MediaPipe capture and TensorFlow decoding for on-camera character detection.",
        "tags": ["Realtime", "Narration", "Camera"],
        "cta": "Launch recognizer",
        "page": "pages/isl_to_text.py",
    },
    {
        "label": "Module",
        "title": "ISL to Words",
        "description": "Phrase-level inference powered by the Roboflow API with multi-language lookups.",
        "tags": ["Roboflow", "Async queue", "Translations"],
        "cta": "Open phrase kit",
        "page": "pages/isl_to_words.py",
    },
    {
        "label": "Module",
        "title": "Text to ISL",
        "description": "Google Translate workspace to prep copy before scheduling new ISL sign packs.",
        "tags": ["Google Translate", "Planning", "Dataset"],
        "cta": "Prep text assets",
        "page": "pages/text_to_isl.py",
    },
    {
        "label": "Vision",
        "title": "Trust & Calibration",
        "description": "Consent, privacy, and calibration placeholders for the upcoming PWA release.",
        "tags": ["Consent", "Calibration", "Analytics"],
        "cta": "Open roadmap",
        "page": "pages/system_vision.py",
    },
]

STAT_CARDS = [
    {"label": "Active models", "value": "3", "delta": "2 vision · 1 NLP"},
    {"label": "Phrases covered", "value": "250+", "delta": "12 added this week"},
    {"label": "Latency target", "value": "<120 ms", "delta": "Lab benchmark"},
]

GUIDE_CARDS = [
    {"title": "Lighting", "body": "Face a soft light source and keep hands above mid-chest for best detection."},
    {"title": "Network", "body": "Minimum 10 Mbps upload recommended for smooth inference streaming."},
    {"title": "Team workflow", "body": "Log findings in Slack #isl-demo so the ML team can triage quickly."},
]

CHECKLIST = [
    "Position the camera at shoulder height with a neutral background.",
    "Warm up signs A–E to confirm MediaPipe is locking onto both hands.",
    "Switch to your preferred translation language before recording.",
    "Use the audio replay button to review pronunciation before sharing clips.",
]


def render_sidebar():
    with st.sidebar:
        st.header("Live demo tips")
        st.write("Keep this panel handy while you record:")
        st.markdown(
            "- Close unused browser tabs.\n"
            "- Use Chrome or Edge for WebRTC access.\n"
            "- Allow camera and microphone permissions when prompted."
        )
        st.divider()
        st.caption("Support: Slack #isl-demo · Email labs@neuronest.ai")


def home_page():
    render_sidebar()

    render_hero(
        "Neuronest ISL Experience Lab",
        "A focused workspace for rapid sign-language experiments, translations, and narration.",
        caption="Stack: Streamlit · MediaPipe · TensorFlow · Roboflow · Google Translate",
        badge="Preview build",
    )

    render_stat_cards(STAT_CARDS)
    render_section_heading("Prototype launcher", "Pick a workflow to explore or demo.")
    render_feature_cards(FEATURES)

    render_section_heading("Session guide", "Set up once, repeatable results all day.")
    render_info_cards(GUIDE_CARDS, columns=3)
    render_checklist("Quick checklist", CHECKLIST)

    st.caption("Questions or feedback? Share your notes in Slack #isl-demo so the team can iterate.")


def main():
    home_page()


if __name__ == "__main__":
    main()
