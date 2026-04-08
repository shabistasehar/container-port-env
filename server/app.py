"""
FastAPI app for Container Port Environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

from openenv.core.env_server import create_web_interface_app
from fastapi.responses import HTMLResponse
import uvicorn

from models import ContainerAction, ContainerObservation
from server.environment import ContainerYardEnvironment

app = create_web_interface_app(
    ContainerYardEnvironment,
    ContainerAction,
    ContainerObservation,
    env_name="container-port-env",
)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
        return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Container Port Dashboard</title>
    <style>
        :root {
            --bg: #f4f5ef;
            --card: #ffffff;
            --ink: #18211f;
            --accent: #0b6e4f;
            --muted: #5f6a66;
            --line: #d7ddd7;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 24px;
            background: radial-gradient(circle at 80% 20%, #dbeee5 0, var(--bg) 45%);
            color: var(--ink);
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }
        .wrap { max-width: 980px; margin: 0 auto; }
        h1 { margin: 0 0 8px; }
        p { margin: 0 0 16px; color: var(--muted); }
        .panel {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        select, input, button {
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 14px;
            background: #fff;
            color: var(--ink);
        }
        button {
            cursor: pointer;
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
            font-weight: 600;
        }
        button.secondary {
            background: #fff;
            color: var(--ink);
            border-color: var(--line);
            font-weight: 500;
        }
        pre {
            margin: 0;
            background: #0e1a17;
            color: #eaf8f1;
            border-radius: 12px;
            padding: 14px;
            overflow: auto;
            min-height: 220px;
            font-size: 12px;
            line-height: 1.35;
        }
        .hint { font-size: 12px; color: var(--muted); margin-top: 8px; }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Container Port Dashboard</h1>
        <p>Pick a difficulty and step the environment manually.</p>

        <div class="panel">
            <div class="row">
                <label for="difficulty">Difficulty</label>
                <select id="difficulty">
                    <option value="easy">Easy</option>
                    <option value="medium" selected>Medium</option>
                    <option value="hard">Hard</option>
                </select>
                <button id="resetBtn">Reset</button>
                <button id="stateBtn" class="secondary">State</button>
            </div>
            <div class="hint">Reset calls <code>/web/reset</code> with the selected mode.</div>
        </div>

        <div class="panel">
            <div class="row">
                <label for="stack">stack_index</label>
                <input id="stack" type="number" min="0" step="1" value="0" />
                <button id="stepBtn">Step</button>
            </div>
            <div class="hint">Step calls <code>/web/step</code> with action <code>{"stack_index": n}</code>.</div>
        </div>

        <div class="panel">
            <pre id="out">Click Reset to start an episode.</pre>
        </div>
    </div>

    <script>
        const out = document.getElementById('out');
        const difficulty = document.getElementById('difficulty');
        const stack = document.getElementById('stack');

        function show(data) {
            out.textContent = JSON.stringify(data, null, 2);
        }

        async function postJson(url, payload) {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            show(data);
        }

        async function getJson(url) {
            const res = await fetch(url);
            const data = await res.json();
            show(data);
        }

        document.getElementById('resetBtn').addEventListener('click', async () => {
            try {
                await postJson('/web/reset', { difficulty: difficulty.value });
            } catch (err) {
                show({ error: String(err) });
            }
        });

        document.getElementById('stepBtn').addEventListener('click', async () => {
            const idx = Number(stack.value);
            try {
                await postJson('/web/step', { action: { stack_index: idx } });
            } catch (err) {
                show({ error: String(err) });
            }
        });

        document.getElementById('stateBtn').addEventListener('click', async () => {
            try {
                await getJson('/web/state');
            } catch (err) {
                show({ error: String(err) });
            }
        });
    </script>
</body>
</html>
"""


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
