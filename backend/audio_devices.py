from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import sounddevice as sd


DeviceSelector = Union[int, str, None]


@dataclass(frozen=True)
class ResolvedDevice:
    index: int
    name: str
    kind: str  # "input" | "output"


def _as_int(s: str) -> Optional[int]:
    try:
        return int(s.strip())
    except Exception:
        return None


def list_devices() -> list[dict[str, Any]]:
    """Return all devices as plain dicts (JSON friendly)."""
    devices = sd.query_devices()
    out: list[dict[str, Any]] = []
    for idx, d in enumerate(devices):
        out.append(
            {
                "index": idx,
                "name": d.get("name"),
                "max_input_channels": d.get("max_input_channels"),
                "max_output_channels": d.get("max_output_channels"),
                "default_samplerate": d.get("default_samplerate"),
                "hostapi": d.get("hostapi"),
            }
        )
    return out


def resolve_device(selector: DeviceSelector, *, kind: str) -> Optional[ResolvedDevice]:
    """
    Resolve device either by index (int or numeric str) or by substring of the name.

    kind:
      - "input": must have max_input_channels > 0
      - "output": must have max_output_channels > 0
    """
    if selector is None:
        return None

    if isinstance(selector, str):
        n = _as_int(selector)
        if n is not None:
            selector = n

    devices = sd.query_devices()

    def ok(dev: dict[str, Any]) -> bool:
        if kind == "input":
            return int(dev.get("max_input_channels") or 0) > 0
        if kind == "output":
            return int(dev.get("max_output_channels") or 0) > 0
        raise ValueError("kind must be input|output")

    if isinstance(selector, int):
        if selector < 0 or selector >= len(devices):
            return None
        dev = devices[selector]
        if not ok(dev):
            return None
        return ResolvedDevice(index=selector, name=str(dev.get("name")), kind=kind)

    # substring search
    needle = str(selector).lower()
    for idx, dev in enumerate(devices):
        if not ok(dev):
            continue
        name = str(dev.get("name") or "")
        if needle in name.lower():
            return ResolvedDevice(index=idx, name=name, kind=kind)
    return None


def env_device(name: str) -> DeviceSelector:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def format_devices_text() -> str:
    lines: list[str] = []
    lines.append("Audio devices (sounddevice):")
    for d in list_devices():
        lines.append(
            f'[{d["index"]:>2}] in={d["max_input_channels"]} out={d["max_output_channels"]} '
            f'hostapi={d["hostapi"]} rate={d["default_samplerate"]}  name={d["name"]}'
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_devices_text())
