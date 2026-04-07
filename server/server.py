import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from server.environment import ContainerYardEnv
from server.models import ContainerAction

app = FastAPI(title="Container Port OpenEnv", version="0.1.0")

sessions: dict = {}

@app.get("/ping")
def ping():
    return {"status": "ok", "env": "container-port-env"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "difficulties": ["easy", "medium", "hard"],
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = ContainerYardEnv(difficulty="medium")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            env = sessions[session_id]

            if msg_type == "reset":
                difficulty = msg.get("difficulty", "medium")
                if difficulty not in ["easy", "medium", "hard"]:
                    difficulty = "medium"
                sessions[session_id] = ContainerYardEnv(difficulty=difficulty)
                env = sessions[session_id]
                obs = env.reset()
                await websocket.send_text(json.dumps({
                    "type": "reset",
                    "observation": obs,
                    "reward": 0.0,
                    "done": False,
                    "session_id": session_id,
                }))

            elif msg_type == "step":
                try:
                    action = ContainerAction(**msg["action"])
                    obs, reward, done, info = env.step(action.stack_index)
                    await websocket.send_text(json.dumps({
                        "type": "step",
                        "observation": obs,
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))

            elif msg_type == "state":
                state = env.get_state()
                await websocket.send_text(json.dumps({
                    "type": "state",
                    "state": state,
                }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))

    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(session_id, None)
