# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meeting Copilot is a Windows-only, local, real-time interview assistant. It captures two audio streams (browser microphone + system loopback via WASAPI), transcribes them live with Deepgram, and generates answer suggestions with Google Gemini. Backend is FastAPI; frontend is a single self-contained HTML file. User-facing text, README and LLM prompts are primarily in French (with an English variant).

## Commands

```powershell
python -m venv .venv          # Python 3.11+ (google-genai 3.x requires it)
.\.venv\Scripts\activate
pip install -r backend\requirements.txt

# Run: double-click start.bat, or
.\.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

Port 8001 is hardcoded in `hotkeys/meeting-copilot.ahk` — keep it. Avoid `--reload` when testing audio: the frontend has no WS auto-reconnect, so every reload kills the page's session (it shows "Connexion au serveur perdue").

`python -m backend.audio_devices` is a standalone diagnostic (uses `sounddevice`); the app never imports it.

There is no test suite, linter or build step.

### Environment (`.env`, see `.env.example`)
- `DEEPGRAM_API_KEY` (required at session start), `GEMINI_API_KEY` (required for `/api/ask` and `/api/chat`)
- `GEMINI_MODEL` — default `gemini-3.6-flash` (latest *Stable* Flash as of 2026-11; `gemini-2.5-*` is deprecated)
- `GEMINI_CLASSIFIER_MODEL` — default `gemini-3.5-flash-lite`, used only by the auto-answer classifier
- `GEMINI_THINKING_LEVEL` — `minimal|low|medium|high`, default `low` (suggestions + chat; the model default `medium` is ~2× slower; 3.x can't disable thinking). Invalid → ignored (model default) and logged once. The classifier always uses `minimal`.
- `AUTO_ANSWER` — initial auto-answer state (`true`/`1`/`yes` → on, default off); toggled at runtime with `POST /api/auto-answer`
- `DEFAULT_LANGUAGE` — `fr` (default) or `en`
- There is no device setting: `get_default_wasapi_loopback_device()` picks the WASAPI loopback matching the Windows default output device. `POST /api/devices` is a no-op kept for compatibility.

## Architecture

### Audio → transcript pipeline (`backend/app.py`, `backend/deepgram_stream.py`)
- **Mic (ME)**: browser `MediaRecorder` (WebM/Opus, 250 ms chunks) → `audio_blob` WS message `{data: base64, speaker, stream_id}` → unbounded `mic_queue` → Deepgram with `is_containerized=True`. Mic chunks must **never** be dropped: the first chunk carries the WebM header. Each browser capture has a fresh `stream_id`; a never-seen id triggers a takeover in `RunningSession.start_mic_stream` (new queue + new Deepgram mic task), retired ids are ignored (`AppState.retired_stream_ids`), so reloads/second tabs can't inject a second WebM header into one connection. The mic Deepgram connection opens lazily on the first chunk.
- **Loopback (OTHER)**: `PyAudioWPatch` stream read in a daemon thread → downmixed to mono (any channel count), resampled to 16 kHz with `scipy.signal.resample_poly` → `call_soon_threadsafe` into a bounded drop-oldest `loop_queue` → Deepgram as raw `linear16`.
- `run_deepgram` sends `KeepAlive` after ~3 s without audio (WASAPI loopback delivers nothing during silence; Deepgram closes after ~10 s idle). On `stop_event` it sends `CloseStream`, drains final results for ≤1.8 s and returns; any connection close while not stopping raises `RuntimeError`.
- Deepgram URL has `endpointing=500` (→ `speech_final` on finals) and `utterance_end_ms=1000` (needs `interim_results=true`); `run_deepgram` forwards `UtteranceEnd` messages as events with empty text and `utterance_end: True`, which `handle_transcript` ignores.
- `_supervise(sess)` watches the session's *current* tasks: a task that raises stops the session and broadcasts `Erreur Deepgram`; tasks replaced by a mic takeover are ignored. It only stops if `STATE.session is sess` (prevents killing the new session after a language switch). `_stop_locked()` (call under `STATE.lock`) sets `stop_event`, waits `STOP_GRACE_SECONDS` (2.5 s) for the drain, then cancels.
- `handle_transcript()` broadcasts interim and final results; **only finals** are appended to `STATE.history` (unbounded) and the meeting JSON is atomically rewritten (temp file + `os.replace`) on every final.

### Speaker attribution
- `auto`: mic → ME, loopback → OTHER.
- `manual`: mic transcripts are attributed via `RunningSession.get_speaker_at(start)` over the speaker history recorded from `audio_blob` tags, timed from the arrival of the current mic stream's first chunk (Deepgram's `start` is relative to the stream). Loopback transcripts use `STATE.current_mic_speaker`.

### LLM (`backend/gemini_suggest.py`, SDK `google-genai`)
- `POST /api/ask` → `_suggest_and_broadcast()` (shared with the auto path) runs **two parallel Gemini calls** (`asyncio.to_thread` + `gather`), same system instruction and transcript, Pydantic `response_schema`, JSON mime type, temp 0.3, fallback parsing of `response.text`:
  - `generate_quick_suggestions()` → `{best, key_points}` (3–5 short ideas, max 5 kept), broadcast as `suggestions_partial` (`best`, `key_points`, `auto`) as soon as it answers;
  - `generate_detailed_suggestions()` → `{alternative, clarifying_question, technical_terms}`;
  - when both are done: the full `suggestions` message (all fields + `language`, `auto`), built by `combine_suggestions()`; a failed call leaves its fields empty. Only if both fail: `error` "Erreur Gemini: …" (and HTTP 400). Requires a running session.
  - Measured with `low`: partial ≈ 2–2.6 s, full ≈ 2.3–3.3 s.
  - `_get_client()` is lock-protected: parallel first calls would otherwise create two clients and the discarded one closes its HTTP connection mid-request.
- **Auto-answer** (`STATE.auto_answer`): `on_mic`/`on_loop` feed `_auto_observe(sess, …)` after `handle_transcript`. Final texts attributed to OTHER are buffered per session; the buffer is evaluated on a final with `raw.speech_final` or an `UtteranceEnd` from a stream that contributed OTHER text. Candidates pass the pure prefilter `looks_like_question()` (≥3 words + `?` or FR/EN cues), then `classify_question()` (classifier model, temp 0, `{is_question}`, last 15 lines; returns False on error). If confirmed: `suggestions_pending` (`auto`, `question`) → `suggestions_partial` → `suggestions` with `auto: true`. A `suggestions_pending` is **always** followed by `suggestions` or `error` (on cancellation, e.g. stop: `error` "Réponse automatique annulée"), since the client keeps placeholders until then. One worker task per session (`sess.auto_task`) with a latest-only slot (`sess.auto_pending`); cancelled in `_stop_locked`. Nothing is buffered while the mode is off.
- `POST /api/chat` → `generate_chat_response()` — free-form question, answered in the HTTP response (works without a running session).
- `_build_transcript()` sends lines `[-200:-80]` as "history" and the last 80 as "current" (non-overlapping).
- The selected profile (`profiles/*.txt|*.md`) is injected into the system instruction, which explicitly forbids inventing experience absent from the profile and technical terms absent from the transcript — preserve this when editing prompts.
- Gemini calls are synchronous and run via `asyncio.to_thread`; keep blocking work off the event loop the same way.

### Security model
Local-only: no CORS. The `local_only` HTTP middleware and the `/ws` handshake reject requests whose `Host` isn't `127.0.0.1`/`localhost`/`[::1]` (DNS rebinding) or whose `Origin`, when present, isn't `http://<Host>`. Requests without `Origin` (AutoHotkey, curl) are allowed. Keep this when adding endpoints.

### Global state & frontend
- A single module-level `STATE = AppState()` holds language, current meeting/profile, speaker mode, history and the active session; WS clients receive `state` (`running`, `language`, `auto_answer`), `transcript`, `suggestions`, `suggestions_pending`, `suggestions_partial`, `auto_answer_changed`, `reset`, `error`, `meeting_changed`, `profile_changed`, `speaker_changed`, `speaker_mode_changed` messages.
- `web/index.html` (no build, no external JS) uses `/ws` for live data and `/api/*` for control. LLM output is rendered with `textContent` only (it is influenced by what the interviewer says). The AutoHotkey script calls `/api/ask`, `/api/toggle-running`, `/api/toggle-language` (state lives on the server).

### Storage
- `meetings/<id>.json` — transcript array `[{speaker, text, ts}, ...]`; ids are sanitized by `_sanitize_meeting_name`; creating a meeting with an existing name returns 409.
- `profiles/*.txt|*.md` — first file found is the default profile
- `keywords.txt` — Deepgram keyword boosting, `#` lines ignored. Loaded **once per process** (`_keywords_cache`), so restart the server after editing. Keep it under ~10 rare terms: more causes false positives.
