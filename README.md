---
title: Container Port
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: A simulation-based RL environment for container management
---

# Container Port Environment

An OpenEnv environment for container-yard stack planning at a shipping terminal.

## Task

Incoming containers have priority `1`, `2`, or `3`. The agent places each one into a bounded stack. During retrieval, every container sitting above the target counts as a rehandle and adds cost.

Goal: minimize total rehandles across the episode.

## Difficulty Levels

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| Stacks | 6 | 8 | 10 |
| Max height | 4 | 5 | 6 |
| Containers | 20 | 35 | 50 |
| Retrieval interval | 5 | 5 | 4 |
| Lookahead | 5 | 3 | 0 |

## Run Locally

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Web UI: `http://127.0.0.1:7860/web`

Interactive dashboard with difficulty dropdown: `http://127.0.0.1:7860/dashboard`

For manual stateful checks, use the web endpoints:

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/web/reset -H "Content-Type: application/json" -d "{\"difficulty\":\"easy\"}"
curl -X POST http://127.0.0.1:7860/web/step -H "Content-Type: application/json" -d "{\"action\":{\"stack_index\":0}}"
```

`/reset` and `/step` are stateless simulation endpoints in `openenv-core 0.2.3`. For browser-style interactive testing, use `/web`, `/web/reset`, `/web/step`, or the WebSocket flow used by `inference.py`.

## Run Inference

```bash
python inference.py --difficulty all
python inference.py --difficulty easy
python inference.py --url http://127.0.0.1:7860 --difficulty all
```

LLM mode is enabled by default in `inference.py` and requires:

```bash
export API_BASE_URL="https://api.openai.com/v1"  # or validator-provided proxy URL
export OPENAI_API_KEY="your-validator-provided-token"
```

`MODEL_NAME` is optional and defaults to `meta-llama/Llama-3.1-8B-Instruct`.
For compatibility with different validator versions, `API_KEY` and `HF_TOKEN` are also accepted.

To run greedy mode locally without LLM calls:

```bash
python inference.py --no-llm
```

## Docker

```bash
docker build -t container-port-env .
docker run -p 7860:7860 container-port-env
```

## Tests

Run the full test suite:

```bash
pytest tests/test_openenv_env.py -v
```

| Test | What it covers |
|---|---|
| test_reset_returns_valid_obs | Reset returns correct stack count, step=0, no rehandles |
| test_step_valid_action | Valid placement increments step and fills stack |
| test_step_invalid_action_penalized | Out-of-range stack index returns -2.0 reward |
| test_score_in_range | Full episode score stays in [0.0, 1.0] |
| test_full_episode_completes | All 3 difficulties reach done=True within 500 steps |
| test_lookahead_visibility | Easy shows more upcoming retrievals than hard (hard=0) |
| test_reward_is_dense | At least 50% of steps have non-zero reward |
| test_no_double_retrieval | retrieval_pointer never exceeds queue length |
| test_health_route | GET /health returns 200 |
| test_web_ui_route | GET /web returns 200 (Gradio UI) |
| test_http_reset_returns_observation | POST /reset returns valid easy-mode observation |
| test_http_reset_then_step_preserves_state | Sequential reset+step operates on same episode |
