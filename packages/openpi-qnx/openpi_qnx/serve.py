"""openpi-compatible websocket policy server for QNX.

Wire protocol identical to openpi.serving.websocket_policy_server (215abfb), so the stock
openpi_client.websocket_client_policy.WebsocketClientPolicy works unchanged:
  - on connect the server sends msgpack(metadata)
  - each request is msgpack(obs); each reply is msgpack(action dict) with "server_timing"
  - on error the server sends the traceback as a text frame and closes with INTERNAL_ERROR
  - GET /healthz returns 200 "OK"

  python3 -m openpi_qnx.serve --model-dir /usr/share/openpi-qnx/models/<name> [--port 8000] [--profile pi0_libero]
"""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
import socket
import time
import traceback

from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from .policy import QnxPolicy

logger = logging.getLogger("openpi_qnx")


class WebsocketPolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata: dict | None = None):
        self._policy, self._host, self._port = policy, host, port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                 process_request=_health_check) as server:
            logger.info("serving on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                t = time.monotonic()
                action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": (time.monotonic() - t) * 1000}
                if prev_total is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total * 1000
                await websocket.send(packer.pack(action))
                prev_total = time.monotonic() - start
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="Internal server error. Traceback included in previous frame.")
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--profile", help="openpi config to reproduce (default: from meta.json)")
    ap.add_argument("--default-prompt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = ORT default)")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    policy = QnxPolicy(a.model_dir, profile=a.profile, default_prompt=a.default_prompt, threads=a.threads,
                       seed=a.seed)
    logger.info("policy %s on %s (%s)", policy.metadata["openpi_config"], socket.gethostname(), a.model_dir)
    WebsocketPolicyServer(policy, a.host, a.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
