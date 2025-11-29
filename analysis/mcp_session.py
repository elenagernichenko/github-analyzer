
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class MCPDockerSession:
    """Minimal MCP client that streams JSON-RPC messages via docker run."""

    def __init__(
        self,
        env_file: Path,
        image: str = "ghcr.io/github/github-mcp-server:latest",
        client_name: str = "mcp-analyzer",
        client_version: str = "0.1.0",
    ) -> None:
        self.env_file = Path(env_file)
        self.image = image
        self.client_name = client_name
        self.client_version = client_version
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_buffer: List[str] = []
        self._next_id = 1

    # ------------------------------------------------------------------ context
    def __enter__(self) -> "MCPDockerSession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._proc is not None:
            return

        cmd = [
            "docker",
            "run",
            "-i",
            "--rm",
            "--env-file",
            str(self.env_file),
            self.image,
        ]
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        if self._proc.stderr:
            threading.Thread(
                target=self._drain_stderr,
                args=(self._proc.stderr,),
                daemon=True,
            ).start()

        # Standard MCP handshake: initialize + initialized notification
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
                "capabilities": {},
            },
        }
        self._send(init_request)
        self._read_until_id(0)
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def close(self) -> None:
        if self._proc is None:
            return

        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass

        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        self._proc = None

    # ---------------------------------------------------------------- requests
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an MCP tool and return the raw JSON-RPC response."""
        req_id = self._next_id
        self._next_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        self._send(request)
        response = self._read_until_id(req_id)
        if "error" in response:
            raise RuntimeError(f"MCP tool error: {response['error']}")
        return response

    def list_tools(self) -> Dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": req_id, "method": "tools/list"}
        self._send(request)
        return self._read_until_id(req_id)

    # ----------------------------------------------------------------- helpers
    def _send(self, message: Dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("MCP session is not started")
        payload = json.dumps(message, separators=(",", ":"))
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()

    def _read_until_id(self, target_id: int) -> Dict[str, Any]:
        while True:
            message = self._read_message()
            if message.get("id") == target_id:
                return message

    def _read_message(self) -> Dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("MCP session is not started")
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the stream unexpectedly")
        return json.loads(line)

    def _drain_stderr(self, stream: Any) -> None:
        for line in stream:
            self._stderr_buffer.append(line.rstrip())


def extract_text_content(result: Dict[str, Any]) -> str:
    """Extract the first text payload from the MCP result object."""
    content = result.get("result", {}).get("content", [])
    for item in content:
        if item.get("type") == "text":
            return item.get("text", "")
    return ""


def batch_call(
    session: MCPDockerSession,
    name: str,
    argument_list: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Call the same tool multiple times and return the parsed responses."""
    responses: List[Dict[str, Any]] = []
    for arguments in argument_list:
        responses.append(session.call_tool(name, arguments))
    return responses


