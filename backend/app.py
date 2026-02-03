from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from dataclasses import dataclass
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


def _load_keywords() -> tuple[str, ...]:
    """Load keywords from keywords.txt file."""
    if not KEYWORDS_FILE.exists():
        return ()
    try:
        content = KEYWORDS_FILE.read_text(encoding="utf-8")
        keywords = []
        for line in content.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            keywords.append(line)
        return tuple(keywords)
    except Exception:
        return ()


@dataclass
class RunningSession:
    stop_event: asyncio.Event
    tasks: list[asyncio.Task[Any]]
    loop_stream: pyaudio.Stream
    loop_thread: threading.Thread
    mic_queue: "asyncio.Queue[bytes]"
    loop_queue: "asyncio.Queue[bytes]"
    pa_instance: pyaudio.PyAudio
    # Track speaker changes with timestamps for accurate attribution
    speaker_history: list = None  # List of (timestamp, speaker)
    session_start_time: float = 0.0
    
    def __post_init__(self):
        if self.speaker_history is None:
            self.speaker_history = []
        if self.session_start_time == 0.0:
            self.session_start_time = time.time()
    
    def record_speaker(self, speaker: str):
        """Record a speaker change with current timestamp."""
        elapsed = time.time() - self.session_start_time
        self.speaker_history.append((elapsed, speaker))
        # Keep only last 60 seconds of history
        cutoff = elapsed - 60
        self.speaker_history = [(t, s) for t, s in self.speaker_history if t > cutoff]
    
    def get_speaker_at(self, audio_start_time: float) -> str:
        """Get the speaker that was active at a given audio timestamp."""
        if not self.speaker_history:
            return "ME"
        # Find the most recent speaker change before audio_start_time
        speaker = "ME"
        for ts, sp in self.speaker_history:
            if ts <= audio_start_time:
                speaker = sp
            else:
                break
        return speaker


def _get_meeting_file(meeting_id: str) -> Path:
    """Get the file path for a meeting."""
    # Sanitize meeting_id to prevent path traversal
    safe_id = "".join(c for c in meeting_id if c.isalnum() or c in "-_ ").strip()
    if not safe_id:
        safe_id = "default"
    return MEETINGS_DIR / f"{safe_id}.json"


def _list_meetings() -> list[dict[str, Any]]:
    """List all available meetings."""
    meetings = []
    for f in MEETINGS_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                entry_count = len(data) if isinstance(data, list) else 0
                meetings.append({
                    "id": f.stem,
                    "name": f.stem,
                    "entries": entry_count,
                    "modified": f.stat().st_mtime,
                })
        except Exception:
            meetings.append({
                "id": f.stem,
                "name": f.stem,
                "entries": 0,
                "modified": 0,
            })
    # Sort by modification time, most recent first
    meetings.sort(key=lambda x: x["modified"], reverse=True)
    return meetings


def _load_meeting(meeting_id: str) -> list[dict[str, Any]]:
    """Load transcript history from a meeting file."""
    meeting_file = _get_meeting_file(meeting_id)
    if meeting_file.exists():
        try:
            with open(meeting_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_meeting(meeting_id: str, history: list[dict[str, Any]]) -> None:
    """Save transcript history to a meeting file."""
    meeting_file = _get_meeting_file(meeting_id)
    try:
        with open(meeting_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _delete_meeting(meeting_id: str) -> bool:
    """Delete a meeting file."""
    meeting_file = _get_meeting_file(meeting_id)
    try:
        if meeting_file.exists():
            meeting_file.unlink()
            return True
    except Exception:
        pass
    return False


def _generate_meeting_id() -> str:
    """Generate a new meeting ID based on current date/time."""
    from datetime import datetime
    return datetime.now().strftime("Meeting_%Y-%m-%d_%H-%M")


def _list_profiles() -> list[dict[str, Any]]:
    """List all available profile files (.txt or .md)."""
    profiles = []
    for ext in ["*.txt", "*.md"]:
        for f in PROFILES_DIR.glob(ext):
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
    """Load a profile's content."""
    if not profile_id:
        return ""
    # Sanitize to prevent path traversal
    safe_id = Path(profile_id).name
    profile_file = PROFILES_DIR / safe_id
    if profile_file.exists():
        try:
            return profile_file.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def _get_default_profile() -> str:
    """Get the first available profile file."""
    for ext in ["*.txt", "*.md"]:
        for f in PROFILES_DIR.glob(ext):
            return f.name
    return ""


class AppState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()
        self.running: bool = False
        self.language: str = (os.getenv("DEFAULT_LANGUAGE") or "fr").strip().lower()
        if self.language not in ("fr", "en"):
            self.language = "fr"
        # Optional override for loopback device (index or substring). Falls back to .env.
        self.loopback_selector: Optional[object] = None
        self.buffer: list[dict[str, Any]] = []  # {speaker,text,ts} - current session
        self.current_meeting: str = _generate_meeting_id()  # Current meeting ID
        self.current_profile: str = _get_default_profile()  # First available profile
        self.speaker_mode: str = "auto"  # 'auto' or 'manual'
        self.current_mic_speaker: str = "ME"  # Who is speaking into the mic (ME or OTHER) - used in manual mode
        self.history: list[dict[str, Any]] = []  # persistent history for current meeting
        self.session: Optional[RunningSession] = None

    def load_meeting(self, meeting_id: str) -> None:
        """Load a meeting's history."""
        self.current_meeting = meeting_id
        self.history = _load_meeting(meeting_id)
        self.buffer = self.history[-250:] if self.history else []

    def _buffer_append(self, speaker: str, text: str, ts: float) -> None:
        entry = {"speaker": speaker, "text": text, "ts": ts}
        self.buffer.append(entry)
        # buffer glissant pour la session courante
        if len(self.buffer) > 250:
            self.buffer = self.buffer[-250:]
        
        # Ajouter à l'historique persistant
        self.history.append(entry)
        # Garder les 500 dernières entrées dans l'historique
        if len(self.history) > 500:
            self.history = self.history[-500:]
        # Sauvegarder
        _save_meeting(self.current_meeting, self.history)

    def reset_history(self) -> None:
        """Clear current meeting's history."""
        self.buffer = []
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
    # project_root/web/index.html
    here = Path(__file__).resolve()
    return here.parents[1] / "web"


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
        # Get default WASAPI output device info
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output_idx = wasapi_info["defaultOutputDevice"]
        default_output = p.get_device_info_by_index(default_output_idx)
        
        # Find the loopback device for this output
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and dev.get("name", "").startswith(default_output.get("name", "")[:20]):
                return dev
        
        # Fallback: just find any loopback device
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

    # Mic audio comes from browser via WebSocket, so we just create a queue
    mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=80)
    loop_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=80)

    # Setup PyAudioWPatch for WASAPI loopback
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

    # Create loopback stream
    loop_stream = p.open(
        format=pyaudio.paInt16,
        channels=device_channels,
        rate=device_sample_rate,
        input=True,
        input_device_index=loopback_device["index"],
        frames_per_buffer=CHUNK_SIZE,
    )

    # Thread to read loopback audio and resample to 16kHz mono
    def loopback_reader():
        import scipy.signal as sig
        
        while not stop_event.is_set():
            try:
                data = loop_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                # Convert to numpy
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Convert to mono if stereo
                if device_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # Resample if needed
                if device_sample_rate != SAMPLE_RATE:
                    num_samples = int(len(audio) * SAMPLE_RATE / device_sample_rate)
                    audio = sig.resample(audio, num_samples).astype(np.int16)
                
                chunk = audio.tobytes()
                
                def _put():
                    if loop_q.full():
                        try:
                            loop_q.get_nowait()
                        except Exception:
                            pass
                    try:
                        loop_q.put_nowait(chunk)
                    except Exception:
                        pass
                
                event_loop.call_soon_threadsafe(_put)
            except Exception:
                if stop_event.is_set():
                    break
                time.sleep(0.01)

    loop_thread = threading.Thread(target=loopback_reader, daemon=True)
    loop_thread.start()

    async def on_mic(ev: dict[str, Any]) -> None:
        # Get the speaker based on when the audio was recorded (using Deepgram's start timestamp)
        if STATE.session and STATE.speaker_mode == "manual":
            # Extract audio start time from Deepgram response
            raw = ev.get("raw", {})
            audio_start = raw.get("start", 0.0)  # Deepgram provides start time in seconds
            speaker = STATE.session.get_speaker_at(audio_start)
        elif STATE.speaker_mode == "auto":
            speaker = "ME"
        else:
            speaker = STATE.current_mic_speaker
        await handle_transcript(speaker=speaker, ev=ev)

    async def on_loop(ev: dict[str, Any]) -> None:
        # In auto mode: loopback = OTHER, in manual mode: also use toggle (opposite of mic)
        if STATE.speaker_mode == "auto":
            speaker = "OTHER"
        else:
            # In manual mode, loopback is not used (or same as mic toggle)
            speaker = STATE.current_mic_speaker
        await handle_transcript(speaker=speaker, ev=ev)

    # Load keywords for better recognition of technical terms
    keywords = _load_keywords()
    
    # Config for mic: WebM/Opus containerized audio from MediaRecorder (no encoding param)
    mic_cfg = DeepgramConfig(
        api_key=deepgram_key,
        language=STATE.language,
        interim_results=True,
        punctuate=True,
        is_containerized=True,  # MediaRecorder sends WebM/Opus
        keywords=keywords,
    )
    
    # Config for loopback: raw PCM16 audio
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
            # Normal cancellation from stop - don't report as error
            pass
        except Exception as e:
            await broadcast({"type": "error", "message": f"Erreur Deepgram: {e}"})
        finally:
            # Ensure session stops if streams end unexpectedly.
            # Use wait_for to avoid deadlock if lock is held by api_stop
            try:
                async with asyncio.timeout(0.5):
                    async with STATE.lock:
                        if STATE.running:
                            await _stop_locked()
            except asyncio.TimeoutError:
                # Lock is held by api_stop, which will handle cleanup
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

    # Cancel tasks first
    for t in sess.tasks:
        try:
            t.cancel()
        except Exception:
            pass

    # Wait for tasks to be cancelled (with timeout)
    try:
        await asyncio.wait_for(
            asyncio.gather(*sess.tasks, return_exceptions=True),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        pass

    # Stop loopback in a thread to avoid blocking the event loop
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

    await broadcast(
        {
            "type": "transcript",
            "speaker": speaker,
            "is_final": is_final,
            "text": text,
            "ts": ts,
        }
    )
    if is_final:
        STATE._buffer_append(speaker, text, ts)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    STATE.clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "state", "running": STATE.running, "language": STATE.language}))
        while True:
            # Receive messages from client (audio data or keep-alive)
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                # Handle audio_blob from MediaRecorder (WebM/Opus format)
                if msg.get("type") == "audio_blob" and STATE.session is not None:
                    audio_b64 = msg.get("data", "")
                    # Record speaker with timestamp for accurate attribution
                    chunk_speaker = msg.get("speaker", "").upper()
                    if chunk_speaker in ("ME", "OTHER"):
                        STATE.session.record_speaker(chunk_speaker)
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        q = STATE.session.mic_queue
                        if q.full():
                            try:
                                q.get_nowait()
                            except Exception:
                                pass
                        try:
                            q.put_nowait(audio_bytes)
                        except Exception:
                            pass
            except json.JSONDecodeError:
                pass  # ignore invalid JSON
    except Exception:
        pass
    finally:
        STATE.clients.discard(ws)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    # Try to find the default loopback device info
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
    
    return JSONResponse(
        {
            "ok": True,
            "running": STATE.running,
            "language": STATE.language,
            **meta,
        }
    )


@app.get("/api/devices")
async def api_devices() -> JSONResponse:
    # Return list of loopback devices for UI dropdowns.
    out: list[dict[str, Any]] = []
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
        p.terminate()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    
    state = (await api_state()).body
    try:
        state_obj = json.loads(state.decode("utf-8"))
    except Exception:
        state_obj = {"ok": True, "running": STATE.running, "language": STATE.language}
    return JSONResponse({"ok": True, "devices": out, "state": state_obj})


@app.post("/api/devices")
async def api_set_devices(payload: dict[str, Any]) -> JSONResponse:
    """
    Note: Loopback device is now auto-detected (default WASAPI output).
    Microphone is captured from browser.
    This endpoint is kept for compatibility but doesn't do much.
    """
    # Loopback is auto-detected, nothing to configure
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
            # Small delay to let devices/WS close cleanly.
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
    """Set who is speaking into the microphone (ME or OTHER) - manual mode only."""
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
        # Also update speaker if provided
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
        # Use full history for better context
        history_items = STATE.history[-200:]  # Last 200 entries from history
        current_items = STATE.buffer[-80:]  # Current session items
        profile_id = STATE.current_profile

    # Load profile content
    profile_content = _load_profile(profile_id) if profile_id else ""

    # Build transcript with history context
    history_transcript = "\n".join([f'{x["speaker"]}: {x["text"]}' for x in history_items]).strip()
    current_transcript = "\n".join([f'{x["speaker"]}: {x["text"]}' for x in current_items]).strip()
    
    # Combine: history provides context, current is the focus
    if history_transcript and current_transcript:
        transcript = f"=== HISTORIQUE DE LA CONVERSATION ===\n{history_transcript}\n\n=== CONVERSATION ACTUELLE ===\n{current_transcript}"
    else:
        transcript = current_transcript or history_transcript
    
    if not transcript:
        return JSONResponse({"ok": False, "error": "no transcript yet"}, status_code=400)

    try:
        # Avoid blocking event loop with sync OpenAI call
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
        # Use full history for context
        history_items = STATE.history[-200:]
        current_items = STATE.buffer[-80:]
        profile_id = STATE.current_profile

    # Load profile content
    profile_content = _load_profile(profile_id) if profile_id else ""

    # Build transcript
    history_transcript = "\n".join([f'{x["speaker"]}: {x["text"]}' for x in history_items]).strip()
    current_transcript = "\n".join([f'{x["speaker"]}: {x["text"]}' for x in current_items]).strip()
    
    if history_transcript and current_transcript:
        transcript = f"=== HISTORIQUE ===\n{history_transcript}\n\n=== ACTUEL ===\n{current_transcript}"
    else:
        transcript = current_transcript or history_transcript or "(pas encore de conversation)"

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
    """List all available meetings."""
    meetings = _list_meetings()
    return JSONResponse({
        "ok": True,
        "meetings": meetings,
        "current": STATE.current_meeting,
    })


@app.post("/api/meetings/new")
async def api_new_meeting(payload: dict[str, Any] = None) -> JSONResponse:
    """Create a new meeting."""
    async with STATE.lock:
        # Generate new meeting ID or use provided name
        name = (payload or {}).get("name", "").strip() if payload else ""
        if name:
            # Sanitize the name
            meeting_id = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        else:
            meeting_id = _generate_meeting_id()
        
        STATE.current_meeting = meeting_id
        STATE.history = []
        STATE.buffer = []
        _save_meeting(meeting_id, [])
        
        await broadcast({"type": "meeting_changed", "meeting": meeting_id})
        return JSONResponse({"ok": True, "meeting": meeting_id})


@app.post("/api/meetings/load")
async def api_load_meeting(payload: dict[str, Any]) -> JSONResponse:
    """Load an existing meeting."""
    meeting_id = str(payload.get("id", "")).strip()
    if not meeting_id:
        return JSONResponse({"ok": False, "error": "meeting id required"}, status_code=400)
    
    async with STATE.lock:
        STATE.load_meeting(meeting_id)
        history = list(STATE.history)  # Copy for response
        await broadcast({"type": "meeting_changed", "meeting": meeting_id})
        return JSONResponse({
            "ok": True, 
            "meeting": meeting_id, 
            "entries": len(history),
            "history": history  # Return history in response
        })


@app.delete("/api/meetings/{meeting_id}")
async def api_delete_meeting(meeting_id: str) -> JSONResponse:
    """Delete a meeting."""
    if meeting_id == STATE.current_meeting:
        return JSONResponse({"ok": False, "error": "cannot delete current meeting"}, status_code=400)
    
    if _delete_meeting(meeting_id):
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": False, "error": "meeting not found"}, status_code=404)


# ============ PROFILES API ============

@app.get("/api/profiles")
async def api_list_profiles() -> JSONResponse:
    """List all available profile files."""
    profiles = _list_profiles()
    return JSONResponse({
        "ok": True,
        "profiles": profiles,
        "current": STATE.current_profile,
    })


@app.post("/api/profiles/select")
async def api_select_profile(payload: dict[str, Any]) -> JSONResponse:
    """Select a profile to use for context."""
    profile_id = str(payload.get("id", "")).strip()
    
    async with STATE.lock:
        # Validate profile exists (or allow empty to disable)
        if profile_id:
            content = _load_profile(profile_id)
            if not content:
                return JSONResponse({"ok": False, "error": "profile not found"}, status_code=404)
        
        STATE.current_profile = profile_id
        await broadcast({"type": "profile_changed", "profile": profile_id})
        return JSONResponse({"ok": True, "profile": profile_id})

