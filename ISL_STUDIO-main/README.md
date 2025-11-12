# Neuronest ISL Studio

Neuronest ISL Studio is a proof-of-concept platform for real-time Indian Sign Language (ISL) interpretation. It packages three interactive prototypes (ISL ➜ Text, ISL ➜ Words, Text ➜ ISL) plus a compliance/roadmap workspace into a single Streamlit Progressive Web App (PWA). Pre-trained TensorFlow models run locally via MediaPipe Holistic keypoints, while Google Translate and gTTS handle textual and audio output.

## Repository layout

```
.
├── demo-app/              # Streamlit UI + pages + shared components
├── server/                # Optional Flask micro-service starter
├── isl_model.keras        # Phrase-level Keras model (used by ISL ➜ Words)
├── demo-app/models/       # Character-level weights for ISL ➜ Text
├── data_preparation.py    # Dataset munging scripts
├── extract_keypoints.py   # MediaPipe landmark extraction utilities
├── train.py               # Model training entry point
└── model.py               # Legacy Tkinter demo ("Neuronest Studio")
```

## Tech stack

- **UI**: Streamlit 1.51+ with custom theming in `ui_components.py`
- **ML / CV**: TensorFlow 2.17, MediaPipe Holistic, OpenCV
- **Language / Audio**: Google Translate (`googletrans`), gTTS for narration
- **Utilities**: NumPy/Pandas for preprocessing, requests for optional HTTP calls

## Installation

Clone the repo and install the Streamlit app dependencies:

```bash
pip install -r demo-app/requirements.txt
```

The requirements file pins `numpy` to `<2` to stay compatible with TensorFlow and MediaPipe. If you plan to run the Flask backend sample as well, install `server/requirements.txt`.

## Running the Streamlit app

```bash
cd demo-app
streamlit run Home.py
```

Visit [http://localhost:8501](http://localhost:8501) and grant camera/microphone access. The launcher page lets you open:

| Module | Description |
| --- | --- |
| **ISL ➜ Text** (`pages/isl_to_text.py`) | Character recognition, translation dropdown, auto-speak, speech-speed control, sidebar replay buttons |
| **ISL ➜ Words** (`pages/isl_to_words.py`) | Phrase recognizer backed by `isl_model.keras`, confidence tuning, low-confidence badges, alternative suggestions, live transcript export, gTTS narration |
| **Text ➜ ISL** (`pages/text_to_isl.py`) | Google Translate assistant to queue phrases for future sign packs |
| **Trust & Calibration** (`pages/system_vision.py`) | Disabled controls capturing compliance and roadmap items (consent, calibration, analytics, accessibility) |

## Model & dataset notes

- The character model (`demo-app/models/model.h5`) covers digits 1–9 and A–Z gestures.
- The phrase model (`isl_model.keras`) was trained on sequences of MediaPipe Holistic keypoints (~30-frame windows) for a vocabulary of greetings, everyday phrases, and politeness markers (~500 candidates on the roadmap, ~9 currently translated).
- Training scripts (`data_preparation.py`, `extract_keypoints.py`, `train.py`) show how to build new datasets: capture video, extract landmarks, assemble sequences, and train LSTM/CNN models.
- The repository does **not** include the raw dataset; only the trained weights are provided. Update `LANGUAGE_TRANSLATIONS` dictionaries when you add new phrases.

## Backend starter (optional)

`server/app.py` is a minimal Flask template for teams that prefer to run inference remotely. Install requirements and start the service:

```bash
cd server
pip install -r requirements.txt
python app.py
```

Then point a client at `POST /api/process` (see `server/README.md` for a sample React hook). Running the server is optional—both ISL modules work fully offline.

## Roadmap & placeholders

The Streamlit UI already reserves space for requirements that are still being wired:

- **Consent & privacy**: camera opt-in modal, “Improve the model” toggle, TLS enforcement.
- **Calibration**: 30-second framing & lighting checklist, handedness detection, visibility alerts.
- **Analytics & remote config**: on-device latency counters, anonymous usage stats, remote threshold/phrase-pack tuning.
- **Accessibility**: high-contrast mode, screen-reader announcements, Hindi localization, overlay mode for video calls.

See `pages/system_vision.py` for the interactive checklist you can demo today.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `numpy._core.multiarray failed to import` | Ensure you installed `demo-app/requirements.txt` so TensorFlow, MediaPipe, and numpy stay compatible. |
| gTTS playback blocked | Some browsers require a user gesture before auto-play works. Use the sidebar replay button the first time. |
| Camera feed blank | Confirm permissions, close other apps that use the webcam, or change the camera index in the page code. |
| Google Translate errors | The `googletrans` package occasionally throttles requests; wait a few seconds and retry. |

## Improving accuracy

A few high‑leverage steps to lift accuracy and stability:

- Data quality/quantity: capture more balanced clips per phrase under varied lighting/backgrounds. Keep both hands fully visible and hold each phrase ~2s.
- Training pipeline: use `train.py` to retrain with standardization, class‑balancing, early stopping, and LR scheduling. Artifacts (`isl_model.keras`, `training_report.txt`, `confusion_matrix.png`, `label_map.json`) are saved at the repo root.
- Inference smoothing: the app now applies exponential moving average (EMA) over probabilities plus a short consensus window to reduce flicker and false positives.
- Threshold tuning: in the ISL ➜ Words page, adjust “Detection confidence” and “Consensus frames” to fit your setup.

To retrain:

```bash
python ISL_STUDIO-main/train.py \
  # Edit video_dir inside train.py to your dataset root
```

The dataset directory should be structured as:

```
<video_dir>/
  <class_0>/ clip1.mp4 clip2.mp4 ...
  <class_1>/ clip1.mp4 clip2.mp4 ...
  ...
```

## Contributing

1. Create a virtual environment and install dependencies.  
2. Keep model files out of version control if they exceed size limits.  
3. Run `streamlit run Home.py` locally when editing UI pages.  
4. Use the `system_vision` page to document any new compliance or UX requirements so stakeholders can preview them even before implementation.
