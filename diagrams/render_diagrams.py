"""Render the PlantUML diagrams to PNG using the public PlantUML web service.

Run from the project root::

    python diagrams/render_diagrams.py

PlantUML and Java are not installed locally, so this encodes each ``.puml`` file
with PlantUML's deflate + custom base64 scheme and downloads the server-rendered
image. The ``.puml`` sources remain the source of truth; the PNGs are convenience
copies for the appendix/submission PDF.
"""

from __future__ import annotations

import base64
import urllib.error
import urllib.request
import zlib
from pathlib import Path

DIAGRAM_DIR = Path(__file__).resolve().parent
SERVER = "https://www.plantuml.com/plantuml/png/"

def _plantuml_base64(data: bytes) -> str:
    """PlantUML's variant of base64 (its own alphabet, no padding)."""

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    out = []
    for index in range(0, len(data), 3):
        chunk = data[index : index + 3]
        buffer = 0
        for offset, byte in enumerate(chunk):
            buffer |= byte << (16 - 8 * offset)
        out.append(alphabet[(buffer >> 18) & 0x3F])
        out.append(alphabet[(buffer >> 12) & 0x3F])
        if len(chunk) > 1:
            out.append(alphabet[(buffer >> 6) & 0x3F])
        if len(chunk) > 2:
            out.append(alphabet[buffer & 0x3F])
    return "".join(out)


def encode(source: str) -> str:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    compressed = compressor.compress(source.encode("utf-8")) + compressor.flush()
    return _plantuml_base64(compressed)


def render(name: str) -> Path:
    source = (DIAGRAM_DIR / f"{name}.puml").read_text(encoding="utf-8")
    url = SERVER + encode(source)
    target = DIAGRAM_DIR / f"{name}.png"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed host
        payload = response.read()
    if not payload.startswith(b"\x89PNG"):
        raise RuntimeError(f"PlantUML did not return a PNG for {name}")
    target.write_bytes(payload)
    return target


def main() -> None:
    for name in ("initial_workflow", "final_workflow", "phase1_pipeline"):
        try:
            path = render(name)
        except (urllib.error.URLError, RuntimeError) as exc:  # pragma: no cover - network
            print(f"{name}: FAILED ({exc})")
        else:
            print(f"{name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
