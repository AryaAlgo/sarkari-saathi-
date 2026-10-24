"""Sarkari Saathi - voice assistant for finding and claiming government schemes.

Pipeline: browser mic -> Sarvam STT -> BM25 retrieval over scheme docs
          -> Sarvam LLM (answer + pick schemes) -> Sarvam TTS -> browser.

Run:  uvicorn main:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from retrieval import SchemeIndex

load_dotenv()
log = logging.getLogger("sarkari")
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
API = "https://api.sarvam.ai"
KEY = os.getenv("SARVAM_API_KEY", "")
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-m")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "ritu")  # v3 voices; use anushka with bulbul:v2
DEFAULT_LANG = os.getenv("DEFAULT_LANGUAGE", "hi-IN")

SUPPORTED_LANGS = {
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN",
    "mr-IN", "gu-IN", "pa-IN", "od-IN", "en-IN",
}
LANG_NAMES = {
    "hi-IN": "Hindi", "bn-IN": "Bengali", "ta-IN": "Tamil", "te-IN": "Telugu",
    "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi",
    "gu-IN": "Gujarati", "pa-IN": "Punjabi", "od-IN": "Odia", "en-IN": "English",
}

index = SchemeIndex.from_file(BASE / "data" / "schemes.json")
app = FastAPI(title="Sarkari Saathi")


def headers() -> dict:
    return {"api-subscription-key": KEY}


# ---------------------------------------------------------------- Sarvam calls
async def speech_to_text(audio: bytes, filename: str, content_type: str) -> tuple[str, str]:
    """Returns (transcript, language_code)."""
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{API}/speech-to-text",
            headers=headers(),
            files={"file": (filename, audio, content_type)},
            data={"model": STT_MODEL, "language_code": "unknown"},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Speech-to-text failed: {r.text[:300]}")
    j = r.json()
    return (j.get("transcript") or "").strip(), j.get("language_code") or DEFAULT_LANG


SYSTEM_PROMPT = """You are Sarkari Saathi, a friendly helper who explains Indian government schemes to people with little formal education.

You get the person's spoken message and a list of CANDIDATE SCHEMES (the only facts you may use).
Do this:
1. Understand who the person is (occupation, state, land, family, age, children and their class, income, gender) from what they said.
2. Pick only the candidate schemes they MAY qualify for. Never invent schemes, amounts or documents. If key details are missing, still suggest likely schemes and say what you would need to confirm.
3. Write the spoken answer in {language}, in very simple everyday words, short sentences, no jargon, no bullet symbols, no markdown. Maximum 90 words. Always say final eligibility is confirmed on the official portal or at a CSC centre.

Reply with ONLY valid JSON, nothing else:
{{"answer": "<spoken answer in {language}>", "scheme_ids": ["<id>", "..."], "followup": "<one short question in {language} asking for the most useful missing detail, or empty string>"}}"""


def build_context(cands: list[dict]) -> str:
    parts = []
    for s in cands:
        parts.append(
            f"id: {s['id']}\nname: {s['name']}\nsummary: {s['summary']}\n"
            f"eligibility: {s['eligibility']}\nbenefit: {s['benefit']}\n"
            f"documents: {', '.join(s['documents'])}"
        )
    return "\n\n".join(parts)


def extract_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def ask_llm(transcript: str, lang: str, cands: list[dict]) -> dict | None:
    language = LANG_NAMES.get(lang, "Hindi")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {
            "role": "user",
            "content": f"CANDIDATE SCHEMES:\n{build_context(cands)}\n\nPERSON SAID:\n{transcript}",
        },
    ]
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{API}/v1/chat/completions",
                headers={**headers(), "Content-Type": "application/json"},
                json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 900},
            )
        if r.status_code != 200:
            log.warning("LLM error %s: %s", r.status_code, r.text[:300])
            return None
        content = r.json()["choices"][0]["message"]["content"]
        return extract_json(content)
    except Exception as e:  # network, parse, etc.
        log.warning("LLM call failed: %s", e)
        return None


def chunk_text(text: str, limit: int = 450) -> list[str]:
    sentences = re.split(r"(?<=[।.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > limit and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks[:4]


async def text_to_speech(text: str, lang: str) -> str | None:
    """Returns base64 WAV audio, or None on failure (UI falls back to browser voice)."""
    target = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{API}/text-to-speech",
                headers={**headers(), "Content-Type": "application/json"},
                json={
                    "text": text[:2400],
                    "language_code": target,
                    "speaker": TTS_SPEAKER,
                    "model": TTS_MODEL,
                },
            )
        if r.status_code != 200:
            log.warning("TTS error %s: %s", r.status_code, r.text[:300])
            return None
        audios = r.json().get("audios") or []
        # Multiple chunks are returned as separate base64 WAVs; the UI plays them in order.
        return audios if audios else None
    except Exception as e:
        log.warning("TTS call failed: %s", e)
        return None


# ---------------------------------------------------------------- fallback
def fallback_answer(cands: list[dict], lang: str) -> dict:
    """Used if the LLM is unreachable: still gives a useful, grounded reply."""
    top = cands[:3]
    if lang == "en-IN":
        names = ", ".join(s["name"] for s in top)
        ans = (f"Based on what you said, you may qualify for: {names}. "
               "Please carry the documents listed below. Final eligibility is confirmed on the official portal or at your nearest CSC centre.")
    else:
        names = ", ".join(s["name_hi"] for s in top)
        ans = (f"आपकी बात के हिसाब से आप इन योजनाओं के पात्र हो सकते हैं: {names}। "
               "नीचे लिखे कागज़ात साथ रखें। अंतिम पात्रता सरकारी पोर्टल या नज़दीकी CSC केंद्र पर पक्की होगी।")
    return {"answer": ans, "scheme_ids": [s["id"] for s in top], "followup": ""}


# ---------------------------------------------------------------- routes
@app.get("/api/health")
async def health():
    return {"ok": True, "sarvam_key_configured": bool(KEY), "schemes": len(index.schemes)}


@app.get("/api/schemes")
async def schemes():
    return index.schemes


@app.post("/api/ask")
async def ask(
    audio: UploadFile | None = File(None),
    text: str = Form(""),
    language: str = Form(""),
):
    if not KEY:
        raise HTTPException(500, "SARVAM_API_KEY is not set. Copy .env.example to .env and add your key.")

    lang = language if language in SUPPORTED_LANGS else ""
    transcript = text.strip()

    if audio is not None and not transcript:
        data = await audio.read()
        if len(data) < 800:
            raise HTTPException(400, "Recording is too short. Please hold the mic and speak.")
        transcript, detected = await speech_to_text(
            data, audio.filename or "voice.webm", audio.content_type or "audio/webm"
        )
        lang = lang or detected
    lang = lang or DEFAULT_LANG

    if not transcript:
        raise HTTPException(400, "Could not understand the audio. Please try again.")

    cands = index.search(transcript, k=6)
    if not cands:
        cands = index.search("farmer health education pension", k=4)

    result = await ask_llm(transcript, lang, cands) or fallback_answer(cands, lang)

    ids = [i for i in result.get("scheme_ids", []) if i in index.by_id]
    if not ids:
        ids = [s["id"] for s in cands[:3]]
    matched = [index.by_id[i] for i in ids]

    answer = (result.get("answer") or "").strip()
    followup = (result.get("followup") or "").strip()
    spoken = f"{answer} {followup}".strip()

    # De-duplicated document checklist across all matched schemes.
    docs: list[str] = []
    for s in matched:
        for d in s["documents"]:
            if d not in docs:
                docs.append(d)

    audios = await text_to_speech(spoken, lang)

    return JSONResponse(
        {
            "transcript": transcript,
            "language": lang,
            "answer": answer,
            "followup": followup,
            "schemes": [
                {
                    "id": s["id"], "name": s["name"], "name_hi": s["name_hi"],
                    "summary": s["summary"], "benefit": s["benefit"],
                    "eligibility": s["eligibility"], "documents": s["documents"],
                    "how_to_apply": s["how_to_apply"],
                }
                for s in matched
            ],
            "documents": docs,
            "audios": audios,
        }
    )


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
async def home():
    return FileResponse(BASE / "static" / "index.html")
