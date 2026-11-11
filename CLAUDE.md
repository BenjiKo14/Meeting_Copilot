# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meeting Copilot is a Windows-only, local, real-time interview assistant. It captures two audio streams (browser microphone + system loopback via WASAPI), transcribes them live with Deepgram, and generates answer suggestions via OpenAI. Backend is FastAPI; frontend is a single self-contained HTML file. User-facing text, README and LLM prompts are primarily in French (with an English variant).

## Commands

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt

# Run (port 8001 is hardcoded in hotkeys/meeting-copilot.ahk — keep it)
.\.venv\Scripts\uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```

Then open `http://127.0.0.1:8001/`.

`python -m backend.audio_devices` lists devices, but it relies on `sounddevice`, which is **not** in `requirements.txt` (install it manually). The app itself never imports this module.

There is no test suite, linter or build step.

### Environment (`.env`, see `.env.example`)
- `DEEPGRAM_API_KEY` (required at session start), `OPENAI_API_KEY`
- `OPENAI_MODEL` — defaults to `gpt-4o`; keep it over `gpt-4o-mini`, which hallucinates profile facts
- `DEFAULT_LANGUAGE` — `fr` (default) or `en`
- `LOOPBACK_DEVICE` — documented in README/.env.example but **currently ignored**: `start_session()` auto-selects the WASAPI loopback matching the Windows default output device (`get_default_wasapi_loopback_device`). `POST /api/devices` is a no-op kept for compatibility.

## Architecture

### Audio → transcript pipeline (`backend/app.py`)
- **Mic (ME)**: browser `MediaRecorder` (WebM/Opus chunks) → base64 `audio_blob` message over `/ws`, each chunk tagged with the speaker selected at capture time → `mic_queue` → Deepgram with `is_containerized=True` (no encoding/sample_rate params).
- **Loopback (OTHER)**: `PyAudioWPatch` stream read in a daemon thread → downmixed to mono, resampled to 16 kHz with `scipy.signal.resample_poly` → pushed into `loop_queue` via `call_soon_threadsafe` → Deepgram as raw `linear16`.
- Each stream has its own bounded queue (maxsize 80, `_enqueue_dropping_oldest` drops old audio rather than blocking) and its own Deepgram WebSocket (`run_deepgram` in `deepgram_stream.py`, model `nova-2`).
- `start_session()` builds everything into a `RunningSession`; a `supervise` task stops the session if either Deepgram task dies. `_stop_locked()` must be called under `STATE.lock`.
- `handle_transcript()` broadcasts interim and final results to all WS clients; **only final results** are appended to `STATE.history` (capped at 500) and the meeting JSON is rewritten on every final.

### Speaker attribution
- `auto`: mic → ME, loopback → OTHER.
- `manual`: mic transcripts are attributed via `RunningSession.get_speaker_at(start)`, which looks up the timestamped speaker history recorded from `audio_blob` chunk tags (last 60 s kept). Loopback transcripts use `STATE.current_mic_speaker`.

### LLM (`backend/openai_suggest.py`)
- `POST /api/ask` → `generate_suggestions()` (JSON mode, temp 0.3) → broadcast as a `suggestions` WS message with `best`, `alternative`, `clarifying_question`, `technical_terms`. Requires a running session.
- `POST /api/chat` → `generate_chat_response()` — free-form question about the conversation, returned in the HTTP response (works without a running session).
- Transcript context is built by `_build_transcript()`: last 200 lines as "history" + last 80 as "current" (overlapping windows).
- The selected profile (`profiles/*.txt|*.md`) is injected into the system prompt, which explicitly forbids inventing experience absent from the profile — preserve this when editing prompts.
- OpenAI calls are synchronous and run via `asyncio.to_thread`; keep any new blocking work off the event loop the same way.

### Global state & frontend
- A single module-level `STATE = AppState()` holds language, current meeting/profile, speaker mode, history and the active session; WS clients receive `state`, `transcript`, `suggestions`, `reset`, `error`, `meeting_changed`, `profile_changed`, `speaker_changed`, `speaker_mode_changed` messages.
- `web/index.html` (~1000 lines, no build, no external JS) talks to `/ws` for live data and `/api/*` REST endpoints for control. The AutoHotkey script calls `/api/ask`, `/api/language`, `/api/start`, `/api/stop` directly.

### Storage
- `meetings/<id>.json` — transcript array `[{speaker, text, ts}, ...]`
- `profiles/*.txt|*.md` — first file found is the default profile
- `keywords.txt` — Deepgram keyword boosting, `#` lines ignored. Loaded **once per process** (`_keywords_cache`), so restart the server after editing. Keep it under ~10 rare terms: more causes false positives.
