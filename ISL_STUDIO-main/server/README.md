# Neuronest Server (Optional)

The Streamlit frontend ships with all models embedded, but some teams prefer to run inference or business logic behind an API. The `server/` folder provides a lightweight Flask + CORS starter that you can harden for production.

## Requirements

```bash
cd server
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
| --- | --- |
| `Flask` | Core web micro-framework |
| `Flask-CORS` | Simple cross-origin configuration so the Streamlit PWA can call the API |
| `python-dotenv` | (Optional) load environment variables from `.env` |
| `requests` | Example client call inside the server or utilities |

## Running locally

```bash
python app.py
```

By default the app listens on `http://127.0.0.1:5000`. Adjust host/port or enable TLS as needed.

## API surface

| Method | Path | Body | Response | Notes |
| --- | --- | --- | --- | --- |
| `POST` | `/api/process` | `{ "name": "Patrick" }` | `{ "message": "Hello, Patrick" }` | Sample echo endpoint demonstrating JSON parsing |

Open `app.py` to replace the placeholder logic with your own model loading, inference, or queue calls. The included React snippet (see below) demonstrates how to call the service from a web/mobile client.

```tsx
const API_URL = 'http://localhost:5000/api/process';
const sendData = async (name: string) => {
  const response = await axios.post(API_URL, { name });
  return response.data.message;
};
```

## Deployment hints

- **Environment variables**: set secrets (API keys, database DSNs) via `.env` or your orchestration platform; `python-dotenv` will populate `os.environ`.
- **CORS**: tighten the allowed origins once you know the exact Streamlit host or custom domain.
- **Scaling**: wrap Flask with a WSGI server such as Gunicorn or waitress for production traffic.
- **TLS**: always terminate HTTPS (either via reverse proxy or directly in your container) before transporting user data.

The Streamlit application does not require this server. Treat `server/` as a reference implementation if you need to expose remote inference or admin APIs alongside the UI.
