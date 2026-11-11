from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyaudiowpatch as pyaudio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.deepgram_stream import DeepgramConfig, run_deepgram
from backend.openai_suggest import generate_suggestions, generate_chat_response


load_dotenv()


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 800  # ~50ms at 16kHz
MEETINGS_DIR = Path(__file__).resolve().parents[1] / "meetings"
MEETINGS_DIR.mkdir(exist_ok=True)
PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)
KEYWORDS_FILE = Path(__file__).resolve().parents[1] / "keywords.txt"

# Transcript window sizes for LLM context
TRANSCRIPT_HISTORY_WINDOW = 200
TRANSCRIPT_CURRENT_WINDOW = 80

# Module-level keywords cache — loaded once at first session start
_keywords_cache: Optional[tuple[str, ...]] = None


def _load_keywords() -> tuple[str, ...]:
    """Load keywords from keywords.txt, cached for the process lifetime."""
    global _keywords_cache
    if _keywords_cache is not None:
        return _keywords_cache
    if not KEYWORDS_FILE.exists():
        _keywords_cache = ()
        return _keywords_cache
    try:
        lines = KEYWORDS_FILE.read_text(encoding="utf-8").splitlines()
        _keywords_cache = tuple(
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        )
    except Exception:
        _keywords_cache = ()
    return _keywords_cache


def _sanitize_meeting_name(name: str) -> str:
    """Sanitize a meeting name to safe filesystem characters."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    return safe or "default"


def _get_meeting_file(meeting_id: str) -> Path:
    return MEETINGS_DIR / f"{_sanitize_meeting_name(meeting_id)}.json"


def _read_json_file(path: Path) -> Any:
    """Read and parse a JSON file, returning None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_meetings() -> list[dict[str, Any]]:
    meetings = []
    for f in MEETINGS_DIR.glob("*.json"):
        data = _read_json_file(f)
        entry_count = len(data) if isinstance(data, list) else 0
        meetings.append({
            "id": f.stem,
            "name": f.stem,
            "entries": entry_count,
            "modified": f.stat().st_mtime if f.exists() else 0,
        })
    meetings.sort(key=lambda x: x["modified"], reverse=True)
    return meetings


def _load_meeting(meeting_id: str) -> list[dict[str, Any]]:
    data = _read_json_file(_get_meeting_file(meeting_id))
    return data if isinstance(data, list) else []


def _save_meeting(meeting_id: str, history: list[dict[str, Any]]) -> None:
    try:
        _get_meeting_file(meeting_id).write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _delete_meeting(meeting_id: str) -> bool:
    meeting_file = _get_meeting_file(meeting_id)
    try:
        if meeting_file.exists():
            meeting_file.unlink()
            return True
    except Exception:
        pass
    return False


def _generate_meeting_id() -> str:
    return datetime.now().strftime("Meeting_%Y-%m-%d_%H-%M")


def _list_profiles() -> list[dict[str, Any]]:
    profiles = []
    for f in itertools.chain(PROFILES_DIR.glob("*.txt"), PROFILES_DIR.glob("*.md")):
        try:
            content = f.read_text(encoding="utf-8")
            preview = content[:100].replace("\n", " ").strip()
            if len(content) > 100:
                preview += "..."
            profiles.append({
                "id": f.name,
                "name": f.stem,
                "preview": preview,
                "size": len(content),
            })
        except Exception:
            profiles.append({
                "id": f.name,
                "name": f.stem,
                "preview": "(erreur lecture)",
                "size": 0,
            })
    profiles.sort(key=lambda x: x["name"].lower())
    return profiles


def _load_profile(profile_id: str) -> str:
    if not profile_id:
        return ""
    profile_file = PROFILES_DIR / Path(profile_id).name
    try:
        return profile_file.read_text(encoding="utf-8")
    except Exception:
        return ""


def _get_default_profile() -> str:
    for f in itertools.chain(PROFILES_DIR.glob("*.txt"), PROFILES_DIR.glob("*.md")):
        return f.name
    return ""


def _enqueue_dropping_oldest(q: "asyncio.Queue[bytes]", item: bytes) -> None:
    """Put item in queue, dropping the oldest entry if full (non-blocking, safe from threads)."""
    if q.full():
        try:
            q.get_nowait()
        except Exception:
            pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


def _build_transcript(
    history: list[dict[str, Any]],
    history_label: str,
    current_label: str,
    empty_fallback: str = "",
) -> str:
    """Build a two-section transcript string from history, or return empty_fallback."""
    history_items = history[-TRANSCRIPT_HISTORY_WINDOW:]
    current_items = history[-TRANSCRIPT_CURRENT_WINDOW:]

    def _fmt(items: list[dict[str, Any]]) -> str:
        return "\n".join(f'{x["speaker"]}: {x["text"]}' for x in items).strip()

    history_text = _fmt(history_items)
    current_text = _fmt(current_items)

    if history_text and current_text:
        return (
            f"=== {history_label} ===\n{history_text}\n\n"
            f"=== {current_label} ===\n{current_text}"
        )
    return history_text or current_text or empty_fallback


@dataclass
class RunningSession:
    stop_event: asyncio.Event
    tasks: list[asyncio.Task[Any]]
    loop_stream: pyaudio.Stream
    loop_thread: threading.Thread
    mic_queue: "asyncio.Queue[bytes]"
    loop_queue: "asyncio.Queue[bytes]"
    pa_instance: pyaudio.PyAudio
    speaker_history: list = field(default_factory=list)
    session_start_time: float = field(default_factory=time.time)

    def record_speaker(self, speaker: str) -> None:
        """Record a speaker change with current timestamp."""
        elapsed = time.time() - self.session_start_time
        self.speaker_history.append((elapsed, speaker))
        # Keep only last 60 seconds of history
        cutoff = elapsed - 60
        self.speaker_history = [(t, s) for t, s in self.speaker_history if t > cutoff]

    def get_speaker_at(self, audio_start_time: float) -> str:
        """Get the speaker that was active at a given audio timestamp."""
        speaker = "ME"
        for ts, sp in self.speaker_history:
            if ts <= audio_start_time:
                speaker = sp
            else:
                break
        return speaker


class AppState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()
        self.running: bool = False
        self.language: str = (os.getenv("DEFAULT_LANGUAGE") or "fr").strip().lower()
        if self.language not in ("fr", "en"):
            self.language = "fr"
        self.loopback_selector: Optional[object] = None
        self.current_meeting: str = _generate_meeting_id()
        self.current_profile: str = _get_default_profile()
        self.speaker_mode: str = "auto"  # 'auto' or 'manual'
        self.current_mic_speaker: str = "ME"
        self.history: list[dict[str, Any]] = []
        self.session: Optional[RunningSession] = None

    def load_meeting(self, meeting_id: str) -> None:
        self.current_meeting = meeting_id
        self.history = _load_meeting(meeting_id)

    def _buffer_append(self, speaker: str, text: str, ts: float) -> None:
        self.history.append({"speaker": speaker, "text": text, "ts": ts})
        if len(self.history) > 500:
            self.history = self.history[-500:]
        _save_meeting(self.current_meeting, self.history)

    def reset_history(self) -> None:
        self.history = []
        _save_meeting(self.current_meeting, self.history)


STATE = AppState()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _web_root() -> Path:
    return Path(__file__).resolve().parents[1] / "web"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_web_root() / "index.html"))


async def broadcast(msg: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    payload = json.dumps(msg, ensure_ascii=False)
    for ws in list(STATE.clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.clients.discard(ws)


async def push_state() -> None:
    await broadcast({"type": "state", "running": STATE.running, "language": STATE.language})


def get_default_wasapi_loopback_device(p: pyaudio.PyAudio) -> dict[str, Any]:
    """Find the default WASAPI loopback device."""
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output_idx = wasapi_info["defaultOutputDevice"]
        default_output = p.get_device_info_by_index(default_output_idx)

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and dev.get("name", "").startswith(default_output.get("name", "")[:20]):
                return dev

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice"):
                return dev

    except Exception:
        pass

    raise RuntimeError("Aucun périphérique loopback WASAPI trouvé")


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


async def start_session() -> None:
    deepgram_key = _require_env("DEEPGRAM_API_KEY")

    stop_event = asyncio.Event()
    event_loop = asyncio.get_running_loop()

    mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=80)
    loop_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=80)

    p = pyaudio.PyAudio()
    loopback_device = get_default_wasapi_loopback_device(p)

    device_sample_rate = int(loopback_device["defaultSampleRate"])
    device_channels = loopback_device["maxInputChannels"]

    meta = {
        "loopback_device": {
            "index": loopback_device["index"],
            "name": loopback_device["name"],
        }
    }

    loop_stream = p.open(
        format=pyaudio.paInt16,
        channels=device_channels,
        rate=device_sample_rate,
        input=True,
        input_device_index=loopback_device["index"],
        frames_per_buffer=CHUNK_SIZE,
    )

    def loopback_reader():
        import scipy.signal as sig

        # Precompute resampling parameters once
        needs_resample = device_sample_rate != SAMPLE_RATE
        if needs_resample:
            from math import gcd
            g = gcd(SAMPLE_RATE, device_sample_rate)
            up, down = SAMPLE_RATE // g, device_sample_rate // g

        while not stop_event.is_set():
            try:
                data = loop_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)

                if device_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

                if needs_resample:
                    audio = sig.resample_poly(audio, up, down).astype(np.int16)

                chunk = audio.tobytes()
                event_loop.call_soon_threadsafe(_enqueue_dropping_oldest, loop_q, chunk)
            except Exception:
                if stop_event.is_set():
                    break
                time.sleep(0.01)

    loop_thread = threading.Thread(target=loopback_reader, daemon=True)
    loop_thread.start()

    async def on_mic(ev: dict[str, Any]) -> None:
        if STATE.session and STATE.speaker_mode == "manual":
            raw = ev.get("raw", {})
            audio_start = raw.get("start", 0.0)
            speaker = STATE.session.get_speaker_at(audio_start)
        elif STATE.speaker_mode == "auto":
            speaker = "ME"
        else:
            speaker = STATE.current_mic_speaker
        await handle_transcript(speaker=speaker, ev=ev)

    async def on_loop(ev: dict[str, Any]) -> None:
        if STATE.speaker_mode == "auto":
            speaker = "OTHER"
        else:
            speaker = STATE.current_mic_speaker
        await handle_transcript(speaker=speaker, ev=ev)

    keywords = _load_keywords()

    mic_cfg = DeepgramConfig(
        api_key=deepgram_key,
        language=STATE.language,
        interim_results=True,
        punctuate=True,
        is_containerized=True,
        keywords=keywords,
    )

    loop_cfg = DeepgramConfig(
        api_key=deepgram_key,
        language=STATE.language,
        sample_rate=SAMPLE_RATE,
        channels=1,
        interim_results=True,
        punctuate=True,
        is_containerized=False,
        keywords=keywords,
    )

    t1 = asyncio.create_task(
        run_deepgram(cfg=mic_cfg, audio_queue=mic_q, on_event=on_mic, stop_event=stop_event, tag="mic"),
        name="deepgram-mic",
    )
    t2 = asyncio.create_task(
        run_deepgram(cfg=loop_cfg, audio_queue=loop_q, on_event=on_loop, stop_event=stop_event, tag="loopback"),
        name="deepgram-loopback",
    )

    STATE.session = RunningSession(
        stop_event=stop_event,
        tasks=[t1, t2],
        loop_stream=loop_stream,
        loop_thread=loop_thread,
        mic_queue=mic_q,
        loop_queue=loop_q,
        pa_instance=p,
    )
    await broadcast({"type": "state", "running": True, "language": STATE.language, **meta})

    async def supervise() -> None:
        try:
            await asyncio.gather(t1, t2)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await broadcast({"type": "error", "message": f"Erreur Deepgram: {e}"})
        finally:
            try:
                async with asyncio.timeout(0.5):
                    async with STATE.lock:
                        if STATE.running:
                            await _stop_locked()
            except asyncio.TimeoutError:
                pass

    asyncio.create_task(supervise(), name="supervise")


async def _stop_locked() -> None:
    sess = STATE.session
    STATE.running = False
    STATE.session = None

    if sess is None:
        await push_state()
        return

    sess.stop_event.set()

    for t in sess.tasks:
        try:
            t.cancel()
        except Exception:
            pass

    try:
        await asyncio.wait_for(
            asyncio.gather(*sess.tasks, return_exceptions=True),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        pass

    def cleanup_audio():
        try:
            sess.loop_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            sess.loop_stream.stop_stream()
            sess.loop_stream.close()
        except Exception:
            pass
        try:
            sess.pa_instance.terminate()
        except Exception:
            pass

    await asyncio.to_thread(cleanup_audio)
    await push_state()


async def handle_transcript(*, speaker: str, ev: dict[str, Any]) -> None:
    text = str(ev.get("text") or "").strip()
    if not text:
        return
    is_final = bool(ev.get("is_final"))
    ts = float(ev.get("ts") or time.time())

    await broadcast({
        "type": "transcript",
        "speaker": speaker,
        "is_final": is_final,
        "text": text,
        "ts": ts,
    })
    if is_final:
        STATE._buffer_append(speaker, text, ts)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    STATE.clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "state", "running": STATE.running, "language": STATE.language}))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "audio_blob" and STATE.session is not None:
                    audio_b64 = msg.get("data", "")
                    chunk_speaker = msg.get("speaker", "").upper()
                    if chunk_speaker in ("ME", "OTHER"):
                        STATE.session.record_speaker(chunk_speaker)
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        _enqueue_dropping_oldest(STATE.session.mic_queue, audio_bytes)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        STATE.clients.discard(ws)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    meta: dict[str, Any] = {}
    try:
        p = pyaudio.PyAudio()
        loopback_device = get_default_wasapi_loopback_device(p)
        meta["loopback_device"] = {
            "index": loopback_device["index"],
            "name": loopback_device["name"],
        }
        p.terminate()
    except Exception as e:
        meta["devices_error"] = str(e)

    return JSONResponse({
        "ok": True,
        "running": STATE.running,
        "language": STATE.language,
        **meta,
    })


@app.get("/api/devices")
async def api_devices() -> JSONResponse:
    out: list[dict[str, Any]] = []
    default_device_name: Optional[str] = None
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("isLoopbackDevice"):
                out.append({
                    "index": d["index"],
                    "name": d.get("name"),
                    "max_input_channels": d.get("maxInputChannels"),
                    "default_samplerate": d.get("defaultSampleRate"),
                    "is_loopback": True,
                })
        # Also grab the default device name in the same PyAudio session
        try:
            loopback_device = get_default_wasapi_loopback_device(p)
            default_device_name = loopback_device.get("name")
        except Exception:
            pass
        p.terminate()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    state_obj: dict[str, Any] = {"ok": True, "running": STATE.running, "language": STATE.language}
    if default_device_name:
        state_obj["loopback_device"] = {"name": default_device_name}

    return JSONResponse({"ok": True, "devices": out, "state": state_obj})


@app.post("/api/devices")
async def api_set_devices(payload: dict[str, Any]) -> JSONResponse:
    """Kept for compatibility. Loopback is auto-detected; microphone is captured from browser."""
    await push_state()
    return JSONResponse({"ok": True})


@app.post("/api/start")
async def api_start() -> JSONResponse:
    async with STATE.lock:
        if STATE.running:
            return JSONResponse({"ok": True})
        try:
            STATE.running = True
            await start_session()
            return JSONResponse({"ok": True})
        except Exception as e:
            STATE.running = False
            await broadcast({"type": "error", "message": str(e)})
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/stop")
async def api_stop() -> JSONResponse:
    async with STATE.lock:
        await _stop_locked()
        return JSONResponse({"ok": True})


@app.post("/api/reset")
async def api_reset() -> JSONResponse:
    async with STATE.lock:
        STATE.reset_history()
        await broadcast({"type": "reset"})
        return JSONResponse({"ok": True})


@app.post("/api/language")
async def api_language(payload: dict[str, Any]) -> JSONResponse:
    lang = str(payload.get("language") or "").strip().lower()
    if lang not in ("fr", "en"):
        return JSONResponse({"ok": False, "error": "language must be fr|en"}, status_code=400)

    async with STATE.lock:
        STATE.language = lang
        was_running = STATE.running
        if was_running:
            await _stop_locked()
            await asyncio.sleep(0.2)
            try:
                STATE.running = True
                await start_session()
            except Exception as e:
                STATE.running = False
                await broadcast({"type": "error", "message": str(e)})
                return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        await push_state()
        return JSONResponse({"ok": True})


@app.post("/api/speaker")
async def api_speaker(payload: dict[str, Any]) -> JSONResponse:
    """Set who is speaking into the microphone (ME or OTHER) — manual mode only."""
    speaker = str(payload.get("speaker", "")).strip().upper()
    if speaker not in ("ME", "OTHER"):
        return JSONResponse({"ok": False, "error": "speaker must be ME or OTHER"}, status_code=400)

    async with STATE.lock:
        STATE.current_mic_speaker = speaker
        await broadcast({"type": "speaker_changed", "speaker": speaker})
        return JSONResponse({"ok": True, "speaker": speaker})


@app.post("/api/speaker-mode")
async def api_speaker_mode(payload: dict[str, Any]) -> JSONResponse:
    """Set speaker detection mode: 'auto' (mic=ME, loopback=OTHER) or 'manual' (user toggles)."""
    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in ("auto", "manual"):
        return JSONResponse({"ok": False, "error": "mode must be 'auto' or 'manual'"}, status_code=400)

    async with STATE.lock:
        STATE.speaker_mode = mode
        speaker = str(payload.get("speaker", "")).strip().upper()
        if speaker in ("ME", "OTHER"):
            STATE.current_mic_speaker = speaker
        await broadcast({"type": "speaker_mode_changed", "mode": mode, "speaker": STATE.current_mic_speaker})
        return JSONResponse({"ok": True, "mode": mode, "speaker": STATE.current_mic_speaker})


@app.post("/api/ask")
async def api_ask() -> JSONResponse:
    async with STATE.lock:
        if not STATE.running:
            return JSONResponse({"ok": False, "error": "not running"}, status_code=400)
        history = list(STATE.history)
        profile_id = STATE.current_profile

    profile_content = _load_profile(profile_id)
    transcript = _build_transcript(
        history,
        history_label="HISTORIQUE DE LA CONVERSATION",
        current_label="CONVERSATION ACTUELLE",
    )

    if not transcript:
        return JSONResponse({"ok": False, "error": "no transcript yet"}, status_code=400)

    try:
        sugg = await asyncio.to_thread(
            generate_suggestions,
            lang=STATE.language,
            transcript=transcript,
            profile=profile_content
        )
        msg = {
            "type": "suggestions",
            "language": STATE.language,
            "best": sugg.best,
            "alternative": sugg.alternative,
            "clarifying_question": sugg.clarifying_question,
            "technical_terms": sugg.technical_terms,
        }
        await broadcast(msg)
        return JSONResponse({"ok": True})
    except Exception as e:
        await broadcast({"type": "error", "message": f"Erreur OpenAI: {e}"})
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/chat")
async def api_chat(payload: dict[str, Any]) -> JSONResponse:
    """Answer a free-form question about the conversation."""
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"ok": False, "error": "no question provided"}, status_code=400)

    async with STATE.lock:
        history = list(STATE.history)
        profile_id = STATE.current_profile

    profile_content = _load_profile(profile_id)
    transcript = _build_transcript(
        history,
        history_label="HISTORIQUE",
        current_label="ACTUEL",
        empty_fallback="(pas encore de conversation)",
    )

    try:
        response = await asyncio.to_thread(
            generate_chat_response,
            lang=STATE.language,
            transcript=transcript,
            profile=profile_content,
            question=question
        )
        return JSONResponse({"ok": True, "response": response})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


# ============ MEETINGS API ============

@app.get("/api/meetings")
async def api_list_meetings() -> JSONResponse:
    meetings = _list_meetings()
    return JSONResponse({
        "ok": True,
        "meetings": meetings,
        "current": STATE.current_meeting,
    })


@app.post("/api/meetings/new")
async def api_new_meeting(payload: dict[str, Any] = None) -> JSONResponse:
    async with STATE.lock:
        name = (payload or {}).get("name", "").strip() if payload else ""
        meeting_id = _sanitize_meeting_name(name) if name else _generate_meeting_id()

        STATE.current_meeting = meeting_id
        STATE.history = []
        _save_meeting(meeting_id, [])

        await broadcast({"type": "meeting_changed", "meeting": meeting_id})
        return JSONResponse({"ok": True, "meeting": meeting_id})


@app.post("/api/meetings/load")
async def api_load_meeting(payload: dict[str, Any]) -> JSONResponse:
    meeting_id = str(payload.get("id", "")).strip()
    if not meeting_id:
        return JSONResponse({"ok": False, "error": "meeting id required"}, status_code=400)

    async with STATE.lock:
        STATE.load_meeting(meeting_id)
        history = list(STATE.history)
        await broadcast({"type": "meeting_changed", "meeting": meeting_id})
        return JSONResponse({
            "ok": True,
            "meeting": meeting_id,
            "entries": len(history),
            "history": history
        })


@app.delete("/api/meetings/{meeting_id}")
async def api_delete_meeting(meeting_id: str) -> JSONResponse:
    if meeting_id == STATE.current_meeting:
        return JSONResponse({"ok": False, "error": "cannot delete current meeting"}, status_code=400)

    if _delete_meeting(meeting_id):
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": False, "error": "meeting not found"}, status_code=404)


# ============ PROFILES API ============

@app.get("/api/profiles")
async def api_list_profiles() -> JSONResponse:
    profiles = _list_profiles()
    return JSONResponse({
        "ok": True,
        "profiles": profiles,
        "current": STATE.current_profile,
    })


@app.post("/api/profiles/select")
async def api_select_profile(payload: dict[str, Any]) -> JSONResponse:
    profile_id = str(payload.get("id", "")).strip()

    async with STATE.lock:
        if profile_id and not _load_profile(profile_id):
            return JSONResponse({"ok": False, "error": "profile not found"}, status_code=404)

        STATE.current_profile = profile_id
        await broadcast({"type": "profile_changed", "profile": profile_id})
        return JSONResponse({"ok": True, "profile": profile_id})
