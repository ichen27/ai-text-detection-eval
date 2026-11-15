"""The detector as a small web app and an MCP server, in one process.

    python -m detecteval.serve          # needs the "ml" and "serve" dependency groups

    GET  /            the web page
    GET  /health      200 when the reference data and the trained classifier are in place
    GET  /api/info    what the detector is and its limits
    POST /api/detect  {"text": "...", "binoculars": false}
    /mcp              MCP over streamable HTTP, tool ``detect_ai_text``

Binds HOST and PORT (default 127.0.0.1:8941). The MCP endpoint keeps the SDK's default host check, so it
answers only requests addressed to localhost: a hub on the same machine can use it, a reverse proxy
serving the web page under another hostname cannot.

The models run in a separate worker process that starts on the first request and is stopped after
IDLE_SECONDS without one, so the service costs almost no memory while idle. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server import MCPServer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route

from . import calibration, config, detector
from .train_classifier import model_dir
from .worker import RECIPE

INDEX = Path(__file__).parent / "static" / "index.html"
IDLE_SECONDS = 300
START_TIMEOUT = 180
REQUEST_TIMEOUT = 300
MAX_BODY_BYTES = 400_000
MIN_FREE_MEMORY_PERCENT = 35      # Binoculars loads about 3 GB; skip it rather than strain the machine

ABOUT = {
    "detectors": {
        "classifier-mirrors": "A DistilRoBERTa classifier trained on RAID and updated with texts from four "
                              "2026 models. The main signal. Higher scores are more AI-like.",
        "binoculars": "Binoculars with a 0.5B model pair: how predictable the text is to small language "
                      "models. Optional second opinion; weak on short text from current models.",
    },
    "measured": "At 300 words the classifier caught 91% of AI text with the bar set to flag 5% of public "
                "human texts. On one student's private pre-ChatGPT essays that same bar flagged 8%.",
    "limits": detector.LIMITS,
}


class Worker:
    """Owns the model process: starts it on demand, serialises requests, stops it when idle."""

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.last_used = 0.0

    async def _start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "detecteval.worker", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, limit=8 * 1024 * 1024)
        await self._read(START_TIMEOUT)

    async def _read(self, timeout: float) -> dict:
        assert self.process and self.process.stdout
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not line:
            raise RuntimeError("the model process stopped unexpectedly")
        return json.loads(line)

    async def score(self, texts: list[str], binoculars: bool) -> dict:
        async with self.lock:
            try:
                if self.process is None or self.process.returncode is not None:
                    await self._start()
                assert self.process and self.process.stdin
                self.process.stdin.write((json.dumps({"texts": texts, "binoculars": binoculars}) + "\n").encode())
                await self.process.stdin.drain()
                reply = await self._read(REQUEST_TIMEOUT)
            except (TimeoutError, RuntimeError, OSError, json.JSONDecodeError) as error:
                await self._stop()
                raise RuntimeError(f"the model process failed: {error or type(error).__name__}") from error
            finally:
                self.last_used = time.monotonic()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    async def _stop(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        self.process = None

    async def reap_when_idle(self) -> None:
        while True:
            await asyncio.sleep(15)
            if self.process and not self.lock.locked() and time.monotonic() - self.last_used > IDLE_SECONDS:
                async with self.lock:
                    await self._stop()


worker = Worker()


async def free_memory_percent() -> int | None:
    """Free memory as macOS reports it; None where that is not available."""
    if not shutil.which("memory_pressure"):
        return None
    process = await asyncio.create_subprocess_exec("memory_pressure", stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.DEVNULL)
    output, _ = await process.communicate()
    for line in output.decode().splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return None


async def detect(text: str, binoculars: bool = False) -> dict:
    passages, truncated = detector.split_passages(text)
    if not passages:
        return detector.too_short()
    notes = []
    if binoculars:
        free = await free_memory_percent()
        if free is not None and free < MIN_FREE_MEMORY_PERCENT:
            binoculars = False
            notes.append("Binoculars was skipped because the machine is low on memory.")
    reply = await worker.score([passage["text"] for passage in passages], binoculars)
    scores = {"classifier-mirrors": reply["classifier-mirrors"], "binoculars": reply.get("binoculars")}
    return detector.build_result(passages, truncated, scores, calibration.load(), notes)


def ready() -> tuple[bool, str]:
    if not calibration.public_path().exists():
        return False, "reference data is missing; run python -m detecteval.calibration"
    if not (model_dir(RECIPE) / "config.json").exists():
        return False, "the trained classifier is missing; see the README section on artifacts"
    return True, "ready"


async def index(_: Request) -> FileResponse:
    return FileResponse(INDEX, headers={"Cache-Control": "no-store"})


async def health(_: Request) -> JSONResponse:
    ok, message = ready()
    return JSONResponse({"ok": ok, "message": message}, status_code=200 if ok else 503)


async def info(_: Request) -> JSONResponse:
    return JSONResponse({**ABOUT, "private_reference": calibration.private_path().exists()})


async def api_detect(request: Request) -> JSONResponse:
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"ok": False, "error": "Text is too long; send at most about 60,000 words."}, 413)
    try:
        payload = json.loads(body)
        text = payload["text"]
        if not isinstance(text, str):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError):
        return JSONResponse({"ok": False, "error": 'Send JSON like {"text": "..."}.'}, 400)
    try:
        return JSONResponse(await detect(text, bool(payload.get("binoculars"))))
    except RuntimeError as error:
        return JSONResponse({"ok": False, "error": f"Could not score the text: {error}"}, 503)


mcp = MCPServer(
    "AI text detector",
    instructions="Scores how AI-like a text is, with measured error rates. A result is evidence, not a verdict: "
                 "never present it as proof that someone used AI.")


@mcp.tool()
async def detect_ai_text(text: str, use_binoculars: bool = False) -> dict:
    """Score how AI-like a piece of English prose is. Runs locally; the text is not sent anywhere.

    The text is cut into passages of up to 500 words. For each passage the result gives a band
    (none, weak, moderate, strong) and the share of human reference texts that score at least as high,
    for public human texts and, when available, one student's private pre-ChatGPT essays. It never says
    "AI" or "human". Needs at least 63 words; under 150 words it is unreliable. Text a person wrote and
    an AI then polished usually scores as AI. Set use_binoculars for a slower second opinion.
    """
    try:
        return await detect(text, use_binoculars)
    except RuntimeError as error:
        return {"ok": False, "error": f"Could not score the text: {error}"}


@mcp.tool()
def about_ai_text_detector() -> dict:
    """What the detector is, how well it measured, and its limits."""
    return ABOUT


mcp_app = mcp.streamable_http_app(stateless_http=True, json_response=True)


@asynccontextmanager
async def lifespan(_: Starlette) -> AsyncIterator[None]:
    reaper = asyncio.create_task(worker.reap_when_idle())
    async with mcp.session_manager.run():
        yield
    reaper.cancel()
    await worker._stop()


app = Starlette(
    routes=[Route("/", index), Route("/health", health), Route("/api/info", info),
            Route("/api/detect", api_detect, methods=["POST"]), Mount("/", app=mcp_app)],
    lifespan=lifespan,
)


def main() -> None:
    import uvicorn

    config.load_env()
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8941")),
                log_level="info")


if __name__ == "__main__":
    main()
