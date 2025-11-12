import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import copy
import itertools
import string
import pandas as pd
from pathlib import Path
from gtts import gTTS
from io import BytesIO
from base64 import b64encode
from typing import Optional
from ui_components import (
    inject_base_styles,
    render_hero,
    render_section_heading,
    render_info_cards,
    render_checklist,
)

st.set_page_config(page_title="ISL to Text · Neuronest", page_icon="Neuronest", layout="wide")

SPEECH_SPEED_MAP = {
    "Normal": False,
    "Slow": True
}

SESSION_DEFAULTS = {
    "camera": None,
    "last_spoken_text": None,
    "last_language": None,
    "tts_cache": {},
    "tts_enabled": True,
    "speech_speed": "Normal",
    "last_audio_bytes": None,
    "tts_action": None,
}

GUIDE_CARDS = [
    {"title": "Camera framing", "body": "Keep shoulders and both hands inside the frame. Center wrists to improve keypoints."},
    {"title": "Background", "body": "Use a plain wall or curtain. Contrasting colors help MediaPipe track fingertips."},
    {"title": "Audio monitor", "body": "Wear headphones if you plan to record screen share so the narration does not echo."},
]

SESSION_CHECKLIST = [
    "Enable Auto speak if you want narration for every new prediction.",
    "Select the spoken language before you start the camera.",
    "Review the translation pane to ensure the detected character is correct before saving footage.",
]


def ensure_session_defaults():
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(value, (dict, list, set)):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value


def render_audio_player(audio_bytes, audio_container, autoplay_placeholder, autoplay=True):
    """Render visible audio player plus optional hidden autoplay element."""
    if not audio_bytes:
        return
    audio_container.audio(audio_bytes, format="audio/mp3")
    if autoplay:
        audio_base64 = b64encode(audio_bytes).decode("utf-8")
        autoplay_placeholder.markdown(
            f"<audio autoplay style='display:none'>"
            f"<source src='data:audio/mp3;base64,{audio_base64}' type='audio/mp3'>"
            f"</audio>",
            unsafe_allow_html=True,
        )
    else:
        autoplay_placeholder.empty()

class SignLanguageApp:
    def __init__(self):
        self.tts_lang_map = {
            'English': 'en',
            'Hindi': 'hi',
            'Marathi': 'mr',
            'Gujarati': 'gu',
            'Bengali': 'bn',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Kannada': 'kn',
            'Malayalam': 'ml'
        }
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            # Higher model_complexity and confidence for improved landmark fidelity
            self.hands = self.mp_hands.Hands(
                model_complexity=1,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        except Exception as e:
            st.error(f"MediaPipe initialization error: {e}")
        
        # Load ML model
        self.model = None
        try:
            models_dir = Path(__file__).resolve().parent.parent / "models"
            model_path = models_dir / "model.h5"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = tf.keras.models.load_model(str(model_path))
            st.success(f"Model loaded successfully from {model_path}!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
        
        # Define alphabet for predictions
        self.alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)
        
        # Translation dictionary
        self.translations = {
            'English': {},
            'Hindi': {},
            'Marathi': {},
            'Gujarati': {},
            'Bengali': {},
            'Tamil': {},
            'Telugu': {},
            'Kannada': {},
            'Malayalam': {}
        }
        
        # Initialize translations for each letter/number
        for char in self.alphabet:
            for lang in self.translations.keys():
                self.translations[lang][char] = char  # You can replace with actual translations

        # Temporal smoothing + gating (tunable)
        self._prob_ema = None
        self._ema_alpha = 0.4
        self._conf_threshold = 0.60
        self._stable_required = 4
        self._top_history = []

    def set_inference_params(self, *, conf_threshold: float, stable_required: int, ema_alpha: float):
        self._conf_threshold = float(np.clip(conf_threshold, 0.3, 0.95))
        self._stable_required = max(1, int(stable_required))
        self._ema_alpha = float(np.clip(ema_alpha, 0.05, 0.95))

    def synthesize_speech(self, text: str, language: str, slow: bool = False):
        """Convert text to speech using gTTS and return audio bytes."""
        lang_code = self.tts_lang_map.get(language, 'en')
        cache_key = (text, lang_code, slow)
        cache = st.session_state.get('tts_cache', {})
        if cache_key in cache:
            return BytesIO(cache[cache_key]), None

        audio_buffer = BytesIO()
        try:
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.getvalue()
            cache[cache_key] = audio_bytes
            st.session_state.tts_cache = cache
            return BytesIO(audio_bytes), None
        except Exception as err:
            return None, str(err)

    def calc_landmark_list(self, image, landmarks):
        """Calculate landmark coordinates (use float precision, avoid early quantization)."""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for landmark in landmarks.landmark:
            landmark_x = float(np.clip(landmark.x * image_width, 0.0, image_width - 1.0))
            landmark_y = float(np.clip(landmark.y * image_height, 0.0, image_height - 1.0))
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list, handedness_label: Optional[str] = None):
        """Center on wrist, rotate to canonical orientation, mirror left to right, normalize to [-1,1]."""
        pts = np.array(landmark_list, dtype=np.float32)
        # Center at wrist (id 0)
        base_x, base_y = pts[0]
        pts[:, 0] -= base_x
        pts[:, 1] -= base_y
        # Orientation using wrist -> middle_mcp (id 9)
        v = pts[9]
        angle = np.arctan2(float(v[1]), float(v[0]))
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts @ rot.T
        # Mirror left to match right-hand canonical
        if handedness_label and str(handedness_label).lower().startswith('left'):
            pts[:, 0] = -pts[:, 0]
        # Flatten and normalize
        flat = pts.reshape(-1)
        max_value = float(np.max(np.abs(flat))) if flat.size else 1.0
        if max_value < 1e-6:
            max_value = 1.0
        flat = (flat / max_value).tolist()
        return flat

    def process_frame(self, frame, selected_language):
        """Process a single frame and return the processed frame and prediction"""
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        prediction = None
        if self.model is None:
            st.error("Model is not available. Please check the model file and restart the app.")
            return frame, None

        if results.multi_hand_landmarks:
            # Evaluate all detected hands, pick the one with highest confidence, then smooth
            best_prob = None
            best_landmarks = None
            best_idx = None
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                # Handedness label, if available
                handed = None
                try:
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handed = results.multi_handedness[idx].classification[0].label
                except Exception:
                    handed = None
                preprocessed_landmarks = self.pre_process_landmark(landmark_list, handedness_label=handed)
                df = pd.DataFrame(preprocessed_landmarks).transpose()
                prob = self.model.predict(df, verbose=0)[0]
                if best_prob is None or float(np.max(prob)) > float(np.max(best_prob)):
                    best_prob = prob
                    best_landmarks = hand_landmarks
                    best_idx = idx

                if best_prob is not None:
                    if self._prob_ema is None:
                        self._prob_ema = best_prob.copy()
                    else:
                        self._prob_ema = self._ema_alpha * best_prob + (1 - self._ema_alpha) * self._prob_ema

                top_idx = int(np.argmax(self._prob_ema))
                top_conf = float(self._prob_ema[top_idx])
                self._top_history.append(top_idx)
                self._top_history = self._top_history[-10:]

                predicted_char = None
                if top_conf >= self._conf_threshold and self._top_history.count(top_idx) >= self._stable_required:
                    predicted_char = self.alphabet[top_idx]
                    prediction = self.translations[selected_language].get(predicted_char, predicted_char)
                else:
                    prediction = None

                # Draw landmarks of the selected hand
                if best_landmarks is not None:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        best_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                overlay_text = (
                    f"Predicted: {predicted_char} (p={top_conf:.2f})" if predicted_char is not None else "Detecting..."
                )
                cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), prediction

def initialize_camera():
    """Initialize the camera capture"""
    return cv2.VideoCapture(0)

def release_camera():
    """Safely release the camera"""
    if 'camera' in st.session_state and st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    cv2.destroyAllWindows()
    if 'last_spoken_text' in st.session_state:
        st.session_state.last_spoken_text = None
    if 'last_audio_bytes' in st.session_state:
        st.session_state.last_audio_bytes = None
    if 'tts_action' in st.session_state:
        st.session_state.tts_action = None

def main():
    inject_base_styles()
    
    # Initialize the app
    app = SignLanguageApp()
    
    # Initialize session state defaults
    ensure_session_defaults()
    
    render_hero(
        "ISL to Text converter",
        "Live webcam recognition with inline narration for faster reviews.",
        caption="MediaPipe · TensorFlow · gTTS",
        badge="Realtime module",
    )
    render_info_cards(GUIDE_CARDS, columns=3)
    render_checklist("Before you record", SESSION_CHECKLIST)
    render_section_heading("Session controls", "Choose your translation, narration, and accuracy preferences.")
    
    # Language selection + TTS controls
    lang_col, tts_col, speed_col = st.columns([2, 1, 1])
    selected_language = lang_col.selectbox(
        "Select Language",
        options=list(app.translations.keys()),
        index=0,
        key='language_select'
    )
    tts_col.toggle("Auto speak", key='tts_enabled')
    speed_col.selectbox("Speech speed", options=list(SPEECH_SPEED_MAP.keys()), key='speech_speed')

    # Accuracy controls
    with st.expander("Recognition accuracy settings", expanded=False):
        preset_col, thresh_col, stable_col, ema_col = st.columns([1, 1, 1, 1])
        accuracy_first = preset_col.toggle("Accuracy first", value=False, help="Raises thresholds and stability for fewer false positives.")
        default_thresh = 0.75 if accuracy_first else 0.60
        default_stable = 6 if accuracy_first else 4
        default_ema = 0.25 if accuracy_first else 0.40
        conf_threshold = thresh_col.slider("Confidence", 0.50, 0.95, value=default_thresh, step=0.01)
        stable_required = int(stable_col.slider("Stable frames", 1, 10, value=default_stable, step=1))
        ema_alpha = ema_col.slider("Smoothing (EMA)", 0.10, 0.90, value=default_ema, step=0.05,
                                   help="Lower values = smoother but slower to react.")
        app.set_inference_params(conf_threshold=conf_threshold, stable_required=stable_required, ema_alpha=ema_alpha)

    # Camera permission gate (toggle)
    if 'camera_permission_granted' not in st.session_state:
        st.session_state.camera_permission_granted = False
    prev_perm = bool(st.session_state.camera_permission_granted)
    perm_col1, perm_col2 = st.columns([2, 1])
    with perm_col1:
        st.caption("Use the toggle to grant or revoke camera access.")
    with perm_col2:
        st.toggle("Camera access", key="camera_permission_granted")
    curr_perm = bool(st.session_state.camera_permission_granted)
    if curr_perm and not prev_perm:
        test_cam = cv2.VideoCapture(0)
        if test_cam is not None and test_cam.isOpened():
            st.success("Camera permission granted.")
            test_cam.release()
            cv2.destroyAllWindows()
        else:
            st.session_state.camera_permission_granted = False
            st.error("Unable to access camera. Check OS permissions and close other apps.")
    elif (not curr_perm) and prev_perm:
        # Revoke: stop any running capture
        if st.session_state.get('camera_checkbox'):
            st.session_state.camera_checkbox = False
        release_camera()

    if st.session_state.last_language != selected_language:
        st.session_state.last_language = selected_language
        st.session_state.last_spoken_text = None
        st.session_state.last_audio_bytes = None
    
    # Sidebar audio controls
    st.sidebar.header("Audio controls")
    st.sidebar.write(f"Auto speak: {'On' if st.session_state.tts_enabled else 'Off'}")
    replay_disabled = st.session_state.last_audio_bytes is None
    if st.sidebar.button("Replay latest speech", disabled=replay_disabled):
        st.session_state.tts_action = "replay"
    if st.sidebar.button("Clear TTS cache"):
        st.session_state.tts_cache = {}
        st.sidebar.success("Cleared cached audio clips.")

    # Placeholders for content regardless of camera state
    frame_window = st.empty()
    translation_container = st.empty()
    audio_container = st.empty()
    autoplay_placeholder = st.empty()

    # Camera control
    if st.checkbox('Start Camera', key='camera_checkbox', disabled=not st.session_state.camera_permission_granted):
        if st.session_state.camera is None:
            st.session_state.camera = initialize_camera()
        
        if st.session_state.camera and st.session_state.camera.isOpened():
            while True:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process frame and get prediction
                processed_frame, prediction = app.process_frame(frame, selected_language)
                
                # Display frame
                frame_window.image(processed_frame)
                
                # Display prediction
                if prediction:
                    translation_container.markdown(f"### Translation: {prediction}")
                    slow_flag = SPEECH_SPEED_MAP.get(st.session_state.speech_speed, False)

                    if st.session_state.tts_enabled and prediction != st.session_state.last_spoken_text:
                        audio_buffer, audio_error = app.synthesize_speech(
                            prediction,
                            selected_language,
                            slow=slow_flag
                        )
                        if audio_error:
                            st.warning(f"TTS error: {audio_error}")
                        elif audio_buffer:
                            audio_bytes = audio_buffer.getvalue()
                            st.session_state.last_audio_bytes = audio_bytes
                            st.session_state.last_spoken_text = prediction
                            render_audio_player(audio_bytes, audio_container, autoplay_placeholder, autoplay=True)
                else:
                    translation_container.empty()
                    audio_container.empty()
                    autoplay_placeholder.empty()
                    st.session_state.last_spoken_text = None
                    
                # Break if checkbox is unchecked
                if not st.session_state.camera_checkbox:
                    break
        else:
            st.error("Cannot open camera. Check connection and permissions.")
    else:
        # Release camera when checkbox is unchecked
        release_camera()

    # Handle manual replay or download actions outside of camera loop
    if st.session_state.tts_action == "replay":
        if st.session_state.last_audio_bytes:
            render_audio_player(
                st.session_state.last_audio_bytes,
                audio_container,
                autoplay_placeholder,
                autoplay=True
            )
        else:
            st.sidebar.warning("No audio available to replay yet.")
        st.session_state.tts_action = None

    # Back button
    if st.button("Back to Home"):
        release_camera()
        st.switch_page("Home.py")

if __name__ == "__main__":
    try:
        main()
    finally:
        release_camera()
