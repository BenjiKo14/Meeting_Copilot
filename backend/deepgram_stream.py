from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

# Send a KeepAlive when no audio has been sent for this long (Deepgram times out after 10 s).
KEEPALIVE_INTERVAL_S = 3.0
# After stop_event is set, keep forwarding results for at most this long waiting for the server close.
CLOSE_DEADLINE_S = 1.8
# Bound on the websocket closing handshake when leaving the connection context.
CLOSE_HANDSHAKE_TIMEOUT_S = 0.4
# Poll period of sender/receiver loops: bounds how late stop_event is noticed.
_POLL_S = 0.1
# Worst case from stop_event.set() to return: _POLL_S + CLOSE_DEADLINE_S + CLOSE_HANDSHAKE_TIMEOUT_S ~= 2.3 s.

_KEEPALIVE_MSG = json.dumps({"type": "KeepAlive"})  # str => text frame, as required by Deepgram
_CLOSE_STREAM_MSG = json.dumps({"type": "CloseStream"})


def _close_code(exc: ConnectionClosed) -> int:
    frame = exc.rcvd or exc.sent
    return frame.code if frame is not None else 1006  # 1006 = closed without a close frame


@dataclass(frozen=True)
class DeepgramConfig:
    api_key: str
    language: str  # "fr" | "en"
    sample_rate: int = 16000
    channels: int = 1
    interim_results: bool = True
    punctuate: bool = True
    endpoint: str = "wss://api.deepgram.com/v1/listen"
    model: str = "nova-2"  # Best model: nova-2 > nova > enhanced > base
    # If True, audio is containerized (WebM/Opus from MediaRecorder) and encoding/sample_rate should be omitted
    is_containerized: bool = False
    # Keywords to boost recognition of specific terms
    keywords: tuple[str, ...] = ()
    # Gap between words (ms) after which Deepgram sends an UtteranceEnd message (needs interim_results).
    # None disables it. Deepgram recommends >= 1000 (interim results are sent about every second).
    utterance_end_ms: Optional[int] = 1000

    def url(self) -> str:
        params = {
            "model": self.model,
            "language": self.language,
            "interim_results": "true" if self.interim_results else "false",
            "punctuate": "true" if self.punctuate else "false",
            "endpointing": "500",  # Wait 500ms of silence before finalizing
            "vad_events": "true",  # Voice activity detection
            "smart_format": "true",  # Better formatting (numbers, dates, etc.)
        }
        # Only add encoding/sample_rate for raw audio (not containerized)
        if not self.is_containerized:
            params["encoding"] = "linear16"
            params["sample_rate"] = str(self.sample_rate)
            params["channels"] = str(self.channels)
        if self.interim_results and self.utterance_end_ms is not None:
            params["utterance_end_ms"] = str(self.utterance_end_ms)
        qs = "&".join([f"{k}={v}" for k, v in params.items()])
        
        # Add keywords for better recognition of technical terms
        if self.keywords:
            from urllib.parse import quote
            keywords_qs = "&".join([f"keywords={quote(kw)}" for kw in self.keywords])
            qs = f"{qs}&{keywords_qs}"
        
        return f"{self.endpoint}?{qs}"


TranscriptCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _extract_transcript(payload: Any) -> tuple[str, bool]:
    """
    Returns (text, is_final). If text empty, caller may ignore.
    Deepgram realtime payload typically contains:
      - is_final: bool
      - channel.alternatives[0].transcript: str
    """
    # Handle non-dict payloads (metadata, errors, etc.)
    if not isinstance(payload, dict):
        return "", False
    
    is_final = bool(payload.get("is_final"))
    channel = payload.get("channel") or {}
    
    # Handle case where channel is not a dict
    if not isinstance(channel, dict):
        return "", is_final
        
    alts = channel.get("alternatives") or []
    if not alts or not isinstance(alts, list):
        return "", is_final
    
    first_alt = alts[0]
    if not isinstance(first_alt, dict):
        return "", is_final
        
    text = (first_alt.get("transcript") or "").strip()
    return text, is_final


async def run_deepgram(
    *,
    cfg: DeepgramConfig,
    audio_queue: "asyncio.Queue[bytes]",
    on_event: TranscriptCallback,
    stop_event: asyncio.Event,
    tag: str,
) -> None:
    """
    Connect to Deepgram websocket, stream audio from audio_queue, and call on_event for transcripts.

    on_event receives dict containing:
      - text, is_final, ts, raw, tag
    Deepgram `UtteranceEnd` messages are forwarded as an event with empty text and
    `utterance_end: True` (is_final True, raw = the UtteranceEnd payload).

    Sends KeepAlive during audio silence. When stop_event is set, sends CloseStream and keeps
    forwarding results until the server closes or CLOSE_DEADLINE_S elapses, then returns.
    Raises RuntimeError if the connection closes (normally or not) while not stopping.
    """
    headers = {"Authorization": f"Token {cfg.api_key}"}
    last_interim: Optional[str] = None
    loop = asyncio.get_running_loop()

    async with websockets.connect(
        cfg.url(),
        additional_headers=headers,
        ping_interval=10,
        ping_timeout=20,
        max_queue=64,
        close_timeout=CLOSE_HANDSHAKE_TIMEOUT_S,
    ) as ws:
        async def sender() -> None:
            # Deepgram closes the stream (NET-0001) after 10 s without audio or KeepAlive.
            last_sent = loop.time()
            try:
                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=_POLL_S)
                    except asyncio.TimeoutError:
                        if loop.time() - last_sent >= KEEPALIVE_INTERVAL_S:
                            await ws.send(_KEEPALIVE_MSG)
                            last_sent = loop.time()
                        continue
                    await ws.send(chunk)
                    last_sent = loop.time()

                # Graceful stop: flush already-queued audio, then ask Deepgram to finalize and close.
                while not audio_queue.empty():
                    await ws.send(audio_queue.get_nowait())
                await ws.send(_CLOSE_STREAM_MSG)
            except ConnectionClosed:
                # The receiver is the authority on connection closure and reports it.
                return

        async def receiver() -> None:
            nonlocal last_interim
            deadline: Optional[float] = None
            while True:
                timeout = _POLL_S
                if stop_event.is_set():
                    if deadline is None:
                        deadline = loop.time() + CLOSE_DEADLINE_S
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        return
                    timeout = min(timeout, remaining)

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                except ConnectionClosedOK as e:
                    if stop_event.is_set():
                        return
                    raise RuntimeError(
                        f"Deepgram [{tag}] a fermé la connexion (code {_close_code(e)}), transcription interrompue"
                    ) from e
                except ConnectionClosedError as e:
                    if stop_event.is_set():
                        return
                    raise RuntimeError(
                        f"Deepgram [{tag}] connexion perdue (code {_close_code(e)}): {e}"
                    ) from e

                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and payload.get("type") == "UtteranceEnd":
                    await on_event(
                        {
                            "text": "",
                            "is_final": True,
                            "utterance_end": True,
                            "ts": time.time(),
                            "raw": payload,
                            "tag": tag,
                        }
                    )
                    continue
                text, is_final = _extract_transcript(payload)
                if not text:
                    continue

                # Avoid spamming identical interim.
                if not is_final:
                    if text == last_interim:
                        continue
                    last_interim = text
                else:
                    last_interim = None

                await on_event(
                    {
                        "text": text,
                        "is_final": is_final,
                        "ts": time.time(),
                        "raw": payload,
                        "tag": tag,
                    }
                )

        recv_task = asyncio.create_task(receiver(), name=f"deepgram-{tag}-recv")
        send_task = asyncio.create_task(sender(), name=f"deepgram-{tag}-send")
        tasks = [recv_task, send_task]
        try:
            # The receiver decides when the stream is over; the sender may finish first.
            pending: set[asyncio.Task[None]] = set(tasks)
            while recv_task in pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    exc = t.exception()
                    if exc is not None:
                        raise exc
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

