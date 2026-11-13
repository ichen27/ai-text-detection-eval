"""Call Workers AI through an MCP endpoint that exposes Cloudflare's API ``execute`` tool.

This works against Cloudflare's own API MCP server or any gateway that fronts it. Configure with:

    MCP_URL          streamable-HTTP MCP endpoint
    MCP_TOKEN_FILE   file holding the bearer token (one line, no "Bearer" prefix)
    MCP_TOOL_PREFIX  optional prefix a gateway adds to tool names, e.g. "cloudflare_ai__"

The token is read from the file on each client construction and is never logged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class McpError(RuntimeError):
    """The MCP endpoint or the tool reported a failure."""


class McpClient:
    def __init__(self, url: str | None = None, token_file: str | None = None, prefix: str | None = None):
        self.url = url or os.environ.get("MCP_URL", "")
        token_path = token_file or os.environ.get("MCP_TOKEN_FILE", "")
        self.prefix = prefix if prefix is not None else os.environ.get("MCP_TOOL_PREFIX", "")
        if not self.url or not token_path:
            raise McpError("Set MCP_URL and MCP_TOKEN_FILE (see README).")
        self._token = Path(token_path).read_text().strip()
        if not self._token:
            raise McpError("The MCP token file is empty.")

    def _rpc(self, method: str, params: dict, timeout: float) -> dict:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        request = urllib.request.Request(self.url, data=body, headers={
            "Authorization": "Bearer " + self._token,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode()
                if "text/event-stream" in response.headers.get("Content-Type", ""):
                    for line in raw.splitlines():
                        if line.startswith("data:"):
                            message = json.loads(line[5:])
                            if "result" in message or "error" in message:
                                return message
                    raise McpError("The MCP stream ended without a result.")
                return json.loads(raw)
        except urllib.error.HTTPError as error:
            raise McpError(f"MCP endpoint returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise McpError(f"MCP endpoint unreachable: {error}") from error

    def call_tool(self, name: str, arguments: dict, timeout: float = 300, unprefixed: bool = False) -> str:
        """Call a tool and return its text content. Raises McpError when the tool reports an error."""
        full_name = name if unprefixed else self.prefix + name
        message = self._rpc("tools/call", {"name": full_name, "arguments": arguments}, timeout)
        if "error" in message:
            raise McpError(f"{full_name}: {message['error'].get('message', 'request failed')}")
        result = message["result"]
        text = "".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
        if result.get("isError"):
            raise McpError(f"{full_name}: {text[:500]}")
        return text


_RUN_BATCH_JS = """async () => {
  const items = %(items)s;
  const out = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      try {
        const r = await cloudflare.request({method: "POST",
          path: `/accounts/${accountId}/ai/run/${items[i].model}`, body: items[i].body});
        out[i] = {ok: true, result: r.result};
      } catch (e) { out[i] = {ok: false, error: String(e).slice(0, 500)}; }
    }
  }
  await Promise.all(Array.from({length: %(concurrency)d}, worker));
  return out;
}"""


class WorkersAI:
    """Run Workers AI models in small batches through the MCP ``execute`` tool."""

    def __init__(self, client: McpClient | None = None):
        self.client = client or McpClient()

    def run_batch(self, items: list[dict], concurrency: int = 4, retries: int = 2) -> list[dict]:
        """``items`` are ``{"model": "@cf/...", "body": {...}}``. Returns one ``{"ok", "result"|"error"}``
        per item, in order. Items that fail are retried ``retries`` times."""
        results: list[dict | None] = [None] * len(items)
        pending = list(range(len(items)))
        for attempt in range(retries + 1):
            if not pending:
                break
            if attempt:
                time.sleep(2 * attempt)
            code = _RUN_BATCH_JS % {"items": json.dumps([items[i] for i in pending]), "concurrency": concurrency}
            try:
                batch = json.loads(self.client.call_tool("execute", {"code": code, "truncate": False}))
                if not isinstance(batch, list) or len(batch) != len(pending):
                    raise McpError("execute returned an unexpected result shape")
            except (McpError, json.JSONDecodeError) as error:
                batch = [{"ok": False, "error": str(error)[:500]}] * len(pending)
            still = []
            for index, outcome in zip(pending, batch, strict=True):
                results[index] = outcome
                if not outcome.get("ok"):
                    still.append(index)
            pending = still
        return [r if r is not None else {"ok": False, "error": "no result"} for r in results]
