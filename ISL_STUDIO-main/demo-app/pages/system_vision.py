import streamlit as st
from ui_components import inject_base_styles, render_section_heading

st.set_page_config(page_title="Neuronest Vision Board", page_icon="Neuronest", layout="wide")
inject_base_styles()


def feature_checkbox(label: str, help_text: str = ""):
    st.checkbox(label, value=False, help=help_text, disabled=True)


def main():
    st.title("Neuronest Roadmap & Placeholder Controls")
    st.caption(
        "This page documents in-progress requirements. "
        "Disabled controls act as placeholders so stakeholders can preview the planned experience."
    )

    render_section_heading("Consent & Privacy", "Required before enabling live capture.")
    with st.container():
        feature_checkbox("User provided explicit camera consent", "UI modal will collect consent before enabling camera.")
        feature_checkbox("Opt-in to 'Improve the model' data sharing")
        feature_checkbox("TLS-enforced transport (auto-enabled in production)")
        st.text_area(
            "Plain-language data policy (draft)",
            "Neuronest captures video locally for interpretation only. "
            "No footage leaves your device unless you opt-in to share anonymized samples.",
            height=100,
            disabled=True,
        )

    render_section_heading("Calibration & Guidance", "30-second warmup sequence.")
    cal_col1, cal_col2 = st.columns(2)
    with cal_col1:
        feature_checkbox("Camera framing checklist", "Will verify shoulders/hands visibility.")
        feature_checkbox("Lighting score", "Real-time histogram + pass/fail.")
    with cal_col2:
        feature_checkbox("Handedness selection", "Left/Right auto-detect fallback.")
        feature_checkbox("Visibility alerts", "Warn when hands leave frame.")
    st.info("Calibration simulator coming soon. For now, follow the on-screen checklist before sessions.")

    render_section_heading("Session UX", "Split-screen layout plan.")
    ux_col1, ux_col2 = st.columns(2)
    with ux_col1:
        feature_checkbox("PiP overlay mode", "Shows captions in compact floating window.")
        feature_checkbox("Top-2 correction suggestions", "Already reserved in recognizer UI.")
    with ux_col2:
        feature_checkbox("Low-confidence badge + haptic cue", "Mobile vibration + desktop toast.")
        feature_checkbox("Adjustable transcript text size", "Toggle will appear beside transcript pane.")

    render_section_heading("Analytics & Remote Config", "For operators and admins.")
    feature_checkbox("Real-time latency dashboard export")
    feature_checkbox("Anonymous usage metrics (opt-in)")
    feature_checkbox("Remote threshold + phrase-pack config")
    st.write(
        "Once backend dashboard is wired, these toggles will control remote configs without redeploying the PWA."
    )

    render_section_heading("Accessibility & Localization", "WCAG + multi-language UI.")
    feature_checkbox("High-contrast mode")
    feature_checkbox("Screen reader announcements")
    feature_checkbox("Hindi UI localization")
    st.caption("Design tokens already support contrast adjustments; toggles will be activated after QA.")


if __name__ == "__main__":
    main()
