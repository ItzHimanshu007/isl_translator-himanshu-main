import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from pathlib import Path
from io import BytesIO
from base64 import b64encode
from typing import Union
from gtts import gTTS
from tensorflow.keras.models import load_model
from ui_components import (
    inject_base_styles,
    render_hero,
    render_section_heading,
    render_info_cards,
    render_checklist,
)

st.set_page_config(page_title="ISL to Words · Neuronest", page_icon="Neuronest", layout="wide")
inject_base_styles()

LANGUAGE_TRANSLATIONS = {
    'hello': {
        'English': 'Hello',
        'Hindi': 'Namaste',
        'Tamil': 'Vanakkam',
        'Telugu': 'Namaskaram',
        'Kannada': 'Namaskara',
        'Punjabi': 'Sat Sri Akal',
        'Gujarati': 'Namaskar',
        'Marathi': 'Namaskar',
        'Malayalam': 'Namaskaram'
    },
    'thankyou': {
        'English': 'Thank you',
        'Hindi': 'Dhanyavaad',
        'Tamil': 'Nandri',
        'Telugu': 'Dhanyavadhamulu',
        'Kannada': 'Dhanyavaadagalu',
        'Punjabi': 'Dhanwaad',
        'Gujarati': 'Aabhar',
        'Marathi': 'Dhanyavaad',
        'Malayalam': 'Nanni'
    },
    'goodmorning': {
        'English': 'Good morning',
        'Hindi': 'Shubh Prabhat',
        'Tamil': 'Kaalai Vanakkam',
        'Telugu': 'Shubhodhayam',
        'Kannada': 'Shubhodaya',
        'Punjabi': 'Sat Sri Akal Subah',
        'Gujarati': 'Suprabhat',
        'Marathi': 'Shubh Prabhat',
        'Malayalam': 'Suprabhatham'
    },
    'goodafternoon': {
        'English': 'Good afternoon',
        'Hindi': 'Shubh Dopahar',
        'Tamil': 'Madhyahnam Vanakkam',
        'Telugu': 'Madhyahnam Namaskaram',
        'Kannada': 'Madhyahna Namaskara',
        'Punjabi': 'Sat Sri Akal Dopher',
        'Gujarati': 'Shubh Aparahn',
        'Marathi': 'Shubh Dophar',
        'Malayalam': 'Shubha Madhyahnam'
    },
    'goodevening': {
        'English': 'Good evening',
        'Hindi': 'Shubh Sandhya',
        'Tamil': 'Maalai Vanakkam',
        'Telugu': 'Shubha Sayankalam',
        'Kannada': 'Shubha Sanje',
        'Punjabi': 'Sat Sri Akal Shaam',
        'Gujarati': 'Shubh Sandhya',
        'Marathi': 'Shubh Sandhya',
        'Malayalam': 'Shubha Sandhya'
    },
    'goodnight': {
        'English': 'Good night',
        'Hindi': 'Shubh Ratri',
        'Tamil': 'Iravu Vanakkam',
        'Telugu': 'Shubha Ratri',
        'Kannada': 'Shubha Ratri',
        'Punjabi': 'Shubh Raat',
        'Gujarati': 'Shubh Ratri',
        'Marathi': 'Shubh Ratri',
        'Malayalam': 'Shubha Ratri'
    },
    'howareyou': {
        'English': 'How are you?',
        'Hindi': 'Aap kaise hain?',
        'Tamil': 'Neenga epadi irukkinga?',
        'Telugu': 'Meeru ela unnaru?',
        'Kannada': 'Neenu hegiddiya?',
        'Punjabi': 'Tusi kiven ho?',
        'Gujarati': 'Tame kem cho?',
        'Marathi': 'Tumhi kase aahat?',
        'Malayalam': 'Ningal engane undu?'
    },
    'alright': {
        'English': 'Alright',
        'Hindi': 'Thik hai',
        'Tamil': 'Sari',
        'Telugu': 'Bagundi',
        'Kannada': 'Chennagide',
        'Punjabi': 'Theek aa',
        'Gujarati': 'Barabar',
        'Marathi': 'Thik aahe',
        'Malayalam': 'Sheri'
    },
    'pleased': {
        'English': 'Pleased to meet you',
        'Hindi': 'Aapse milkar khushi hui',
        'Tamil': 'Ungalai sandhithathil magizhchi',
        'Telugu': 'Mimmalni kalisi santosham',
        'Kannada': 'Nimmannu nodi santosha',
        'Punjabi': 'Tuhanu mil ke khushi hoi',
        'Gujarati': 'Tamne mali ne anand thayo',
        'Marathi': 'Tumhala bhetun anand zhala',
        'Malayalam': 'Ninne kanan sandosham'
    }
}


def normalize_key(text: str) -> str:
    return ''.join(ch.lower() for ch in text if ch.isalnum())


TRANSLATION_LOOKUP = {normalize_key(k): v for k, v in LANGUAGE_TRANSLATIONS.items()}

GUIDE_CARDS = [
    {"title": "Camera framing", "body": "Capture from the waist up so both hands remain visible to MediaPipe."},
    {"title": "Gesture pacing", "body": "Hold each phrase for ~2 seconds; the model needs 30 consecutive frames."},
    {"title": "Ambient setup", "body": "Bright, even lighting and a solid background improve confidence scores."},
]

CHECKLIST = [
    "Warm up signs for 10 seconds so the model calibrates to your movement.",
    "Switch on captions only after you see the webcam feed is stable.",
    "Use the translation panel to confirm each phrase before saving footage.",
]

TTS_LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Malayalam": "ml",
}

WORDS_SESSION_DEFAULTS = {
    "words_tts_enabled": True,
    "words_tts_language": "English",
    "words_tts_cache": {},
    "words_last_spoken_text": None,
    "words_last_audio": None,
    "words_tts_action": None,
    "words_transcript": [],
}


def ensure_words_session_defaults():
    for key, value in WORDS_SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else value


from typing import Union
def synthesize_phrase_audio(text: str, language: str) -> Union[BytesIO, None]:
    lang_code = TTS_LANGUAGE_CODES.get(language, "en")
    cache_key = (text, lang_code)
    cache = st.session_state.get("words_tts_cache", {})
    if cache_key in cache:
        return BytesIO(cache[cache_key])
    audio_buffer = BytesIO()
    try:
        gtts_obj = gTTS(text=text, lang=lang_code)
        gtts_obj.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        cache[cache_key] = audio_buffer.getvalue()
        st.session_state["words_tts_cache"] = cache
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as exc:
        st.warning(f"TTS error: {exc}")
        return None


def render_audio(audio_bytes: bytes, audio_container, autoplay_placeholder, autoplay: bool = True):
    if not audio_bytes:
        audio_container.empty()
        autoplay_placeholder.empty()
        return
    audio_container.audio(audio_bytes, format="audio/mp3")
    if autoplay:
        encoded = b64encode(audio_bytes).decode("utf-8")
        autoplay_placeholder.markdown(
            f"<audio autoplay style='display:none'>"
            f"<source src='data:audio/mp3;base64,{encoded}' type='audio/mp3'>"
            f"</audio>",
            unsafe_allow_html=True,
        )
    else:
        autoplay_placeholder.empty()


class LocalPhraseRecognizer:
    def __init__(self):
        self.actions = np.array([
            'Alright',
            'Good afternoon',
            'Good evening',
            'Good morning',
            'Good night',
            'Hello',
            'How are you',
            'Pleased',
            'Thank you'
        ])
        model_path = Path(__file__).resolve().parent.parent.parent / "isl_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find model at {model_path}")
        self.model = load_model(str(model_path))
        self.sequence = []
        self.predictions = []
        self.threshold = 0.7
        self.consensus_frames = 6
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = None
        self._build_holistic()

    def reset(self):
        self.sequence = []
        self.predictions = []

    def _build_holistic(self):
        if self.holistic:
            self.holistic.close()
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5
        )

    def ensure_active(self):
        if self.holistic is None:
            self._build_holistic()

    def cleanup(self):
        if self.holistic:
            self.holistic.close()
            self.holistic = None

    def set_sensitivity(self, threshold: float, consensus: int):
        self.threshold = float(np.clip(threshold, 0.3, 0.95))
        self.consensus_frames = max(3, int(consensus))

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        keypoints = np.concatenate([pose, face, lh, rh])[:63]
        return keypoints

    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def process_frame(self, frame):
        self.ensure_active()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        annotated = frame.copy()
        self.draw_landmarks(annotated, results)

        phrase = None
        confidence = None
        suggestion = None
        suggestion_conf = None
        alternatives = []
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]

            if len(self.sequence) == 30:
                input_seq = np.expand_dims(self.sequence, axis=0)
                res = self.model.predict(input_seq, verbose=0)[0]
                sorted_idx = np.argsort(res)[::-1]
                top_idx = int(sorted_idx[0])
                conf = float(res[top_idx])
                self.predictions.append(top_idx)
                self.predictions = self.predictions[-10:]

                suggestion = self.actions[top_idx]
                suggestion_conf = conf
                alternatives = [
                    (self.actions[int(idx)], float(res[idx]))
                    for idx in sorted_idx[: min(3, len(sorted_idx))]
                ]

                if conf > self.threshold and self.predictions.count(top_idx) >= self.consensus_frames:
                    phrase = suggestion
                    confidence = conf

        return annotated, phrase, confidence, suggestion, suggestion_conf, alternatives


def render_translation_panel(container, phrase, confidence, translations, languages):
    if not phrase:
        container.empty()
        return
    container.markdown(f"### Detected phrase: **{phrase}**")
    if confidence is not None:
        container.caption(f"Confidence: {confidence:.2f}")

    rows = []
    for lang in languages:
        value = translations.get(lang, phrase)
        rows.append(f"- **{lang}**: {value}")
    if rows:
        container.markdown("\n".join(rows))


def main():
    ensure_words_session_defaults()
    render_hero(
        "ISL to Words recognizer",
        "Run the uploaded phrase model locally for instant translations.",
        caption="Powered by MediaPipe + isl_model.keras (local)",
        badge="Local model",
    )
    render_info_cards(GUIDE_CARDS, columns=3)
    render_checklist("Before you record", CHECKLIST)
    render_section_heading("Session", "Choose translation languages and start the camera.")

    selected_languages = st.multiselect(
        "Select languages for translation output",
        list(TTS_LANGUAGE_CODES.keys()),
        default=['English', 'Hindi']
    )
    sensitivity_col, consensus_col = st.columns(2)
    confidence_threshold = sensitivity_col.slider("Detection confidence", 0.3, 0.9, value=0.65, step=0.05)
    consensus_frames = consensus_col.slider("Consensus frames", 3, 10, value=5, help="Number of consecutive frames that must agree before locking in a phrase.")
    tts_toggle_col, tts_lang_col = st.columns(2)
    tts_toggle_col.toggle("Auto speak", key="words_tts_enabled")
    tts_lang_col.selectbox("Narration language", list(TTS_LANGUAGE_CODES.keys()), key="words_tts_language")

    if "phrase_recognizer" not in st.session_state:
        st.session_state.phrase_recognizer = None

    if st.session_state.phrase_recognizer is None:
        try:
            st.session_state.phrase_recognizer = LocalPhraseRecognizer()
        except Exception as exc:
            st.error(f"Failed to load local phrase model: {exc}")
            return

    recognizer = st.session_state.phrase_recognizer
    recognizer.set_sensitivity(confidence_threshold, consensus_frames)

    frame_window = st.image([])
    translation_container = st.empty()
    low_conf_placeholder = st.empty()
    alternatives_container = st.empty()
    transcript_container = st.empty()
    status_placeholder = st.empty()
    audio_container = st.empty()
    autoplay_placeholder = st.empty()

    run = st.checkbox("Start camera", key="words_camera_checkbox")

    if run:
        recognizer.reset()
        recognizer.ensure_active()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Cannot access webcam. Please check permissions.")
            translation_container.empty()
            audio_container.empty()
            autoplay_placeholder.empty()
            return

        prev_frame_time = 0.0
        try:
            while st.session_state.get("words_camera_checkbox", False):
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                annotated, phrase, confidence, suggestion, suggestion_conf, alternatives = recognizer.process_frame(frame)
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")

                if phrase:
                    translations = TRANSLATION_LOOKUP.get(normalize_key(phrase), {})
                    render_translation_panel(translation_container, phrase, confidence, translations, selected_languages)
                    st.session_state.words_transcript.append({
                        "timestamp": time.time(),
                        "phrase": phrase,
                        "confidence": confidence or 0.0,
                        "spoken_language": st.session_state.words_tts_language,
                        "spoken_text": translations.get(st.session_state.words_tts_language, phrase),
                    })
                    if st.session_state.words_tts_enabled:
                        spoken_lang = st.session_state.words_tts_language
                        text_to_read = translations.get(spoken_lang, phrase)
                        marker = st.session_state.words_last_spoken_text
                        if text_to_read and marker != (phrase, spoken_lang):
                            audio_buffer = synthesize_phrase_audio(text_to_read, spoken_lang)
                            if audio_buffer:
                                audio_bytes = audio_buffer.getvalue()
                                st.session_state.words_last_audio = audio_bytes
                                st.session_state.words_last_spoken_text = (phrase, spoken_lang)
                                render_audio(audio_bytes, audio_container, autoplay_placeholder, autoplay=True)
                    if confidence is not None and confidence < confidence_threshold + 0.05:
                        low_conf_placeholder.warning("Low confidence. Adjust lighting or pause between signs.")
                    else:
                        low_conf_placeholder.empty()
                    if alternatives:
                        alt_lines = [f"{label} ({score:.2f})" for label, score in alternatives[:2]]
                        alternatives_container.markdown("**Alternatives:** " + " · ".join(alt_lines))
                elif suggestion:
                    translation_container.caption(f"Listening… best guess: {suggestion} ({suggestion_conf:.2f})")
                    low_conf_placeholder.info("Hold the sign steady to confirm recognition.")
                    if alternatives:
                        alt_lines = [f"{label} ({score:.2f})" for label, score in alternatives[:3]]
                        alternatives_container.markdown("**Alternatives:** " + " · ".join(alt_lines))
                    audio_container.empty()
                    autoplay_placeholder.empty()
                else:
                    translation_container.empty()
                    low_conf_placeholder.empty()
                    alternatives_container.empty()
                    audio_container.empty()
                    autoplay_placeholder.empty()

                new_frame_time = time.time()
                if prev_frame_time == 0:
                    fps = 0.0
                else:
                    fps = 1.0 / max(new_frame_time - prev_frame_time, 1e-6)
                prev_frame_time = new_frame_time
                status_placeholder.caption(f"FPS: {fps:.2f}")

                if not st.session_state.get("words_camera_checkbox", False):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
            translation_container.info("Camera stopped. Toggle the checkbox to run again.")
            audio_container.empty()
            autoplay_placeholder.empty()
    else:
        translation_container.info("Enable the checkbox to begin capturing phrases.")
        audio_container.empty()
        autoplay_placeholder.empty()

    with transcript_container:
        st.markdown("### Session transcript")
        if st.session_state.words_transcript:
            for entry in reversed(st.session_state.words_transcript[-50:]):
                timestamp = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
                st.markdown(f"- `{timestamp}` **{entry['phrase']}** (confidence {entry['confidence']:.2f})")
            transcript_text = "\n".join(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))} - {entry['phrase']} ({entry['confidence']:.2f})"
                for entry in st.session_state.words_transcript
            )
            st.download_button(
                "Download transcript",
                data=transcript_text,
                file_name="neuronest_isl_words_transcript.txt",
                mime="text/plain",
            )
        else:
            st.caption("No phrases captured yet. Start the camera to build a transcript.")

    st.sidebar.header("Narration controls")
    st.sidebar.write(f"Auto speak: {'On' if st.session_state.words_tts_enabled else 'Off'}")
    if st.sidebar.button("Replay last phrase", disabled=st.session_state.words_last_audio is None):
        st.session_state.words_tts_action = "replay"
    if st.sidebar.button("Clear TTS cache"):
        st.session_state.words_tts_cache = {}
        st.sidebar.success("Cleared cached audio clips.")

    if st.session_state.words_tts_action == "replay":
        audio_bytes = st.session_state.get("words_last_audio")
        if audio_bytes:
            render_audio(audio_bytes, audio_container, autoplay_placeholder, autoplay=True)
        else:
            st.sidebar.warning("No phrase audio available yet.")
        st.session_state.words_tts_action = None


if __name__ == "__main__":
    try:
        main()
    finally:
        if "phrase_recognizer" in st.session_state and st.session_state.phrase_recognizer:
            st.session_state.phrase_recognizer.cleanup()
