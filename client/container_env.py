import json
import websockets
from typing import Any, Dict, Tuple

class ContainerEnvClient:
    """Async client for Container Port OpenEnv."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = ws_url.rstrip("/") + "/ws"
        self._ws = None

    async def __aenter__(self):
        self._ws = await websockets.connect(self.ws_url)
        return self

    async def __aexit__(self, *args):
        if self._ws:
            await self._ws.close()

    async def reset(self, difficulty: str = "medium") -> Dict[str, Any]:
        await self._ws.send(json.dumps({"type": "reset", "difficulty": difficulty}))
        resp = json.loads(await self._ws.recv())
        return resp["observation"]

    async def step(self, stack_index: int) -> Tuple[Dict, float, bool, Dict]:
        await self._ws.send(json.dumps({
            "type": "step",
            "action": {"stack_index": stack_index}
        }))
        resp = json.loads(await self._ws.recv())
        return resp["observation"], resp["reward"], resp["done"], resp.get("info", {})

    async def state(self) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"type": "state"}))
        resp = json.loads(await self._ws.recv())
        return resp["state"]
