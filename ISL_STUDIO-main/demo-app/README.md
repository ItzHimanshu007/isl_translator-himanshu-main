# Neuronest ISL Studio – Streamlit Frontend

This folder contains the Streamlit application that powers the Neuronest ISL Studio demo. It exposes three prototype experiences:

1. **ISL ➜ Text** – character-level recognition with on-device narration.  
2. **ISL ➜ Words** – phrase-level recognition, multi-language translations, transcripts, and gTTS playback.  
3. **Text ➜ ISL** – Google Translate workspace to plan new sign packs before recording.  
4. **Trust & Calibration (vision)** – placeholder toggles describing consent, calibration, analytics, and accessibility requirements.

All pages share the design system defined in `ui_components.py`.

## Requirements

Create/activate a Python 3.10+ environment and install the dependencies listed in `demo-app/requirements.txt`:

```bash
pip install -r demo-app/requirements.txt
```

> **Tip:** Mediapipe and TensorFlow depend on `numpy<2`. The requirements file pins `numpy` accordingly, so use it instead of installing packages piecemeal.

## Running the app

```bash
cd demo-app
streamlit run Home.py
```

Navigate to [http://localhost:8501](http://localhost:8501). Grant camera and microphone permissions when Streamlit prompts you.

## Page summary

| Page | Path | Highlights |
| --- | --- | --- |
| Home | `Home.py` | Hero, KPI cards, module launcher, quick checklist |
| ISL ➜ Text | `pages/isl_to_text.py` | MediaPipe Hands + TensorFlow, language dropdown, auto-speak toggle, speed control, sidebar replay/clear buttons |
| ISL ➜ Words | `pages/isl_to_words.py` | Local `isl_model.keras`, confidence sliders, alternative suggestions, low-confidence hints, transcript export, gTTS narration |
| Text ➜ ISL | `pages/text_to_isl.py` | Google Translate (auto-detect), “Request sign pack” CTA, pipeline overview, best-practice checklist |
| Trust & Calibration | `pages/system_vision.py` | Disabled controls describing consent, calibration, analytics, accessibility, and remote-config goals |

## Development notes

- Model files live outside this folder (`isl_model.keras` and `demo-app/models/*.h5`). The pages resolve them via relative paths, so keep the repo structure intact.
- Auto-play audio uses embedded `<audio>` tags. Some browsers block auto-play until the user interacts with the page; use the sidebar replay button if needed.
- When editing styling, update `ui_components.py` so the theme stays consistent across pages.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `ImportError: numpy._core.multiarray failed to import` | Ensure `pip install -r demo-app/requirements.txt` was run so TensorFlow, Mediapipe, and numpy remain in sync. |
| Webcam feed black/blank | Confirm browser permission, pick a different camera index in the code if you have multiple devices, and ensure no other app is using the camera. |
| Google Translate throttling | The `googletrans` package scrapes the public endpoint. If it temporarily fails, wait a few seconds and retry. |
| Streamlit auto-reload loop | Disable `streamlit run`'s watcher (`streamlit run Home.py --server.fileWatcherType none`) if you are editing large binary files. |

## Folder structure

```
demo-app/
├── Home.py
├── ui_components.py
├── requirements.txt
├── README.md
└── pages/
    ├── isl_to_text.py
    ├── isl_to_words.py
    ├── text_to_isl.py
    └── system_vision.py
```
