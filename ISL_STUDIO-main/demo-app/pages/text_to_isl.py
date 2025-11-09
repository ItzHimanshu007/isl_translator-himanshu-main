import streamlit as st
from googletrans import Translator
from ui_components import (
    inject_base_styles,
    render_hero,
    render_section_heading,
    render_info_cards,
    render_checklist,
)

st.set_page_config(page_title="Text to ISL · Neuronest", page_icon="Neuronest", layout="centered")
inject_base_styles()

PAGE_STYLE = """
<style>
.form-wrapper {
    background: var(--bg-panel);
    border: 1px solid var(--border-muted);
    border-radius: 1rem;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}
.placeholder-card {
    padding: 1.2rem 1.4rem;
    border-radius: 1rem;
    border: 1px dashed rgba(94, 129, 255, 0.5);
    background-color: rgba(94, 129, 255, 0.08);
}
.placeholder-card ol {
    padding-left: 1.2rem;
}
.small {
    font-size: 0.85rem;
    color: #cfd5ff;
}
</style>
"""
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
}
DEFAULT_TARGETS = ["Hindi", "Marathi", "Gujarati"]


def translate_text(text: str, src_code: str, target_langs: list[str]) -> list[dict]:
    """Translate text via Google Translate and return per-language metadata."""
    translator = Translator()
    results = []
    for lang in target_langs:
        dest_code = LANGUAGE_CODES[lang]
        try:
            translated = translator.translate(text, src=src_code, dest=dest_code)
            results.append(
                {
                    "language": lang,
                    "text": translated.text,
                    "pronunciation": translated.pronunciation or "",
                    "detected_src": translated.src,
                    "error": None,
                }
            )
        except Exception as err:
            results.append(
                {
                    "language": lang,
                    "text": "",
                    "pronunciation": "",
                    "detected_src": None,
                    "error": str(err),
                }
            )
    return results


def main():
    render_hero(
        "Text to ISL prep workspace",
        "Translate phrases, capture feedback, and queue the next sign pack.",
        caption="Powered by Google Translate · Output saved for dataset planning",
        badge="Workflow",
    )

    render_section_heading("Translation assistant", "Auto-detect source language or lock to a locale.")

    if "translation_rows" not in st.session_state:
        st.session_state.translation_rows = None
        st.session_state.detected_src = None

    st.markdown("<div class='form-wrapper'>", unsafe_allow_html=True)
    with st.form("google_translate_form"):
        user_text = st.text_area(
            "Message",
            placeholder="Example: I'd like a glass of water, please.",
            height=150,
        )
        source_options = ["Auto Detect"] + list(LANGUAGE_CODES.keys())
        source_choice = st.selectbox("Source language", source_options, index=0)
        source_code = "auto" if source_choice == "Auto Detect" else LANGUAGE_CODES[source_choice]
        target_langs = st.multiselect(
            "Translate to",
            options=list(LANGUAGE_CODES.keys()),
            default=DEFAULT_TARGETS,
        )
        submitted = st.form_submit_button("Translate with Google", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        cleaned_text = user_text.strip()
        if not cleaned_text:
            st.warning("Enter some text to translate.")
        elif not target_langs:
            st.warning("Select at least one target language.")
        else:
            rows = translate_text(cleaned_text, source_code, target_langs)
            st.session_state.translation_rows = rows
            detected = next((row["detected_src"] for row in rows if row["detected_src"]), None)
            st.session_state.detected_src = detected
            st.success("Translations updated.")

    if st.session_state.translation_rows:
        st.subheader("Google Translate output")
        if st.session_state.detected_src and st.session_state.detected_src != "auto":
            st.caption(f"Detected source language: {st.session_state.detected_src.upper()}")

        for row in st.session_state.translation_rows:
            st.markdown(f"**{row['language']}**")
            if row["error"]:
                st.error(row["error"])
            else:
                st.write(row["text"])
                if row["pronunciation"]:
                    st.caption(f"Pronunciation: {row['pronunciation']}")
        st.divider()

    col_request, col_home = st.columns([2, 1])
    with col_request:
        if st.button("Request ISL Sign Pack", type="primary", use_container_width=True):
            latest_text = user_text.strip()
            if latest_text:
                st.success(
                    "Thanks! Your phrase has been logged for dataset annotation. "
                    "We'll notify you in Slack once the sign pack is ready."
                )
            else:
                st.warning("Add a phrase first so we know what to plan.")
    with col_home:
        st.button("Back to Home", type="secondary", use_container_width=True, on_click=lambda: st.switch_page("Home.py"))

    st.divider()
    render_section_heading("Pipeline overview", "Where each approved phrase goes next.")
    st.markdown(
        """
        <div class="placeholder-card">
            <strong>What happens after you hit request?</strong>
            <ol>
                <li>Phrase is scored for frequency and accessibility priority.</li>
                <li>Motion-capture with ISL interpreters is scheduled.</li>
                <li>Segments are annotated, validated, and queued for inference.</li>
            </ol>
            <p class="small">Want to fast-track a phrase? Drop a note in Slack #isl-demo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_info_cards(
        [
            {"title": "Status tracker", "body": "Each submitted phrase is logged to Airtable with owner and ETA."},
            {"title": "Language coverage", "body": "Current focus: English plus eight Indian languages with phonetic hints."},
        ],
        columns=2,
    )
    render_checklist(
        "Best practices",
        [
            "Write one intent per submission for faster approval.",
            "Avoid slang or abbreviations unless you provide context footage.",
            "Confirm pronunciation in the translation output before sharing with the ML team.",
        ],
    )


if __name__ == "__main__":
    main()
