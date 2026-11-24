"""HTTP server helpers for the local Tic Tac Toe UI."""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from xrl.ui.session import TicTacToeSession


def make_handler(session: TicTacToeSession, *, html: str):
    """Build a request handler bound to one in-memory session."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send_html(html)
            elif self.path == "/state":
                session.refresh_runs()
                self._send_json(session.payload())
            elif self.path == "/runs":
                session.refresh_runs()
                self._send_json({"runs": [run.to_json_dict() for run in session.runs]})
            elif self.path == "/replay":
                self._send_json(session.replay_payload())
            elif self.path == "/sverl":
                self._send_json(session.sverl_payload())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/move":
                    self._send_json(session.human_move(int(body["action"])))
                elif self.path == "/step":
                    self._send_json(session.step_agent())
                elif self.path == "/reset":
                    self._send_json(session.reset())
                elif self.path == "/refresh-runs":
                    self._send_json(session.refresh_runs())
                elif self.path in {"/run", "/agent"}:
                    self._send_json(session.set_run(str(body.get("run_id", body.get("agent", "")))))
                else:
                    self.send_error(404)
            except Exception as exc:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, content: str) -> None:
            data = content.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def resolve_runs_dir(path: str | Path) -> Path:
    """Resolve a relative runs directory from the current directory or its parents."""

    runs_dir = Path(path)
    if runs_dir.is_absolute():
        return runs_dir
    if runs_dir.exists():
        return runs_dir

    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / runs_dir
        if candidate.exists():
            return candidate
    return runs_dir


def serve_ui(
    *,
    html: str,
    session_factory: Callable[..., TicTacToeSession],
    host: str,
    port: int,
    runs_dir: str | Path,
    run_id: str | None,
    seed: int,
) -> None:
    """Create a session and serve the local browser UI forever."""

    resolved_runs_dir = resolve_runs_dir(runs_dir)
    session = session_factory(
        runs_dir=resolved_runs_dir,
        run_id=run_id,
        seed=seed,
    )
    server = ThreadingHTTPServer((host, port), make_handler(session, html=html))
    print(f"Serving xrl Tic Tac Toe at http://{host}:{port}")
    print(f"Using runs dir: {resolved_runs_dir} ({len(session.runs)} runs)")
    server.serve_forever()
