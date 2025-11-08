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
            self.hands = self.mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
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
        """Calculate landmark coordinates"""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
            
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        """Preprocess landmarks for model input"""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0]
        for point in temp_landmark_list:
            point[0] = point[0] - base_x
            point[1] = point[1] - base_y
        
        # Convert to one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalization
        max_value = max(map(abs, temp_landmark_list))
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        
        return temp_landmark_list

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
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                results.multi_handedness):
                # Calculate and preprocess landmarks
                landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                preprocessed_landmarks = self.pre_process_landmark(landmark_list)
                
                # Prepare data for prediction
                df = pd.DataFrame(preprocessed_landmarks).transpose()
                
                # Make prediction
                predictions = self.model.predict(df, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_char = self.alphabet[predicted_class]
                
                # Get translation
                prediction = self.translations[selected_language].get(predicted_char, predicted_char)
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add prediction text to frame
                cv2.putText(
                    frame,
                    f"Predicted: {predicted_char}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
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
    render_section_heading("Session controls", "Choose your translation and narration preferences.")
    
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
    if st.checkbox('Start Camera', key='camera_checkbox'):
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
