# Sarkari Saathi

A voice assistant that helps people find and claim government schemes, in their own language.

**Flow:** browser mic -> Sarvam speech-to-text -> retrieval over scheme documents -> Sarvam LLM (picks schemes, explains in simple words) -> Sarvam text-to-speech -> spoken answer + document checklist.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your key from https://dashboard.sarvam.ai in SARVAM_API_KEY
uvicorn main:app --reload
```

Open http://localhost:8000, allow the microphone, tap the mic and speak (or type / tap an example).
Microphone access works on `localhost` and on HTTPS only.

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app: `/api/ask` runs STT -> retrieve -> LLM -> TTS |
| `retrieval.py` | BM25 retrieval (Hindi + English), swappable for a vector store |
| `data/schemes.json` | Scheme knowledge base (15 schemes with eligibility, benefits, documents, how to apply) |
| `static/index.html` | Web UI: mic recording, chat, scheme cards, document checklist |

## API

`POST /api/ask` (multipart form): `audio` (file) **or** `text`, optional `language` (e.g. `hi-IN`).
Returns `transcript`, `language`, `answer`, `followup`, `schemes[]`, `documents[]`, `audios[]` (base64 WAV chunks).

`GET /api/health`, `GET /api/schemes`.

## Add more schemes or languages

- Append an object to `data/schemes.json` (same fields; put Hindi and English search words in `keywords`). Restart the server.
- Languages: pick any Sarvam-supported code in the UI dropdown; STT auto-detects and the LLM/TTS reply in the same language.
- To upgrade retrieval to embeddings, replace `SchemeIndex` in `retrieval.py` and keep `search(query, k)`.

## Notes

- If the LLM is unreachable the app falls back to a grounded answer built from retrieval; if TTS fails, the browser's own voice reads the reply.
- Scheme amounts and rules in `schemes.json` are a summary for the demo. **Verify against the official portals before launch**; rules and income limits change.
- Model names (`saarika:v2.5`, `sarvam-m`, `bulbul:v3`) are configurable in `.env` if Sarvam updates them.


## Local setup (macOS)

Run these commands from the folder containing `main.py` and `requirements.txt`:

```bash
cd ~/Downloads/sarkari-saathi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file in this folder using `.env.example` as a template, then add your own Sarvam API key. Never commit `.env` or put the key in frontend files.

Start the app:

```bash
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser. Keep the terminal running while testing.
