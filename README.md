# Container Port Environment

An OpenEnv-compatible RL environment for container yard management at a shipping terminal.

## Task

A ship arrives with N containers (priority 1=urgent, 2=normal, 3=low). The agent places each into
stacks. At regular intervals, specific containers are retrieved. If a target is buried under others,
each container above it is a **rehandle** — expensive in real port operations.

**Goal: minimize total rehandle operations across the episode.**

## Difficulty Levels

| Parameter          | Easy     | Medium   | Hard     |
|--------------------|----------|----------|----------|
| Stacks             | 6        | 8        | 10       |
| Max stack height   | 4        | 5        | 6        |
| Containers         | 20       | 35       | 50       |
| Retrieval interval | every 5  | every 5  | every 4  |
| Lookahead shown    | 5        | 3        | 0        |

## Reward

| Event | Reward |
|---|---|
| Accessible placement of priority-1 (near top) | up to +0.45 |
| General placement | +0.03 to +0.30 |
| Burying high-priority under low-priority | -0.10 to -0.20 |
| Invalid action (full stack / bad index) | -2.0 |
| Each rehandle at retrieval time | -0.40 |

## Score

`score = 1.0 - (actual_rehandles / worst_case_rehandles)`, in [0.0, 1.0].

## Setup
```bash
pip install -r requirements.txt
uvicorn server.server:app --host 0.0.0.0 --port 7860
```

## Run inference
```bash
# Greedy agent, all difficulties
python inference.py --difficulty all

# LLM agent (requires HF token in env)
export HF_TOKEN=hf_your_token_here
python inference.py --use-llm --difficulty all

# Against deployed HF Space
python inference.py --url https://YOUR_USERNAME-container-port-env.hf.space --difficulty all
```

## Docker
```bash
docker build -t container-port-env .
docker run -p 7860:7860 container-port-env
```

## API

- `GET /ping` — health check
- `GET /health` — server stats
- `WS /ws` — WebSocket interface

WebSocket messages:
- `{"type": "reset", "difficulty": "easy"}` — start episode
- `{"type": "step", "action": {"stack_index": 2}}` — place container
- `{"type": "state"}` — get full state with score
