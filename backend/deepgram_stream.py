from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import websockets


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
    """
    headers = {"Authorization": f"Token {cfg.api_key}"}
    last_interim: Optional[str] = None

    async with websockets.connect(
        cfg.url(),
        additional_headers=headers,
        ping_interval=10,
        ping_timeout=20,
        max_queue=64,
    ) as ws:
        async def sender() -> None:
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                await ws.send(chunk)

        async def receiver() -> None:
            nonlocal last_interim
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                payload = json.loads(raw)
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

        await asyncio.gather(sender(), receiver())

