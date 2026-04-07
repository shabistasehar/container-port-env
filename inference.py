#!/usr/bin/env python3
"""
Container Port OpenEnv — Baseline Inference Script
SST x Meta PyTorch OpenEnv Hackathon 2026

Stdout format (grader parses these exactly):
  [START] task=<task> env=container-port-env model=<model>
  [STEP]  step=<n> action=<stack_idx> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
  python inference.py
  python inference.py --difficulty easy
  python inference.py --difficulty all
  python inference.py --use-llm
  python inference.py --url https://YOUR_USERNAME-container-port-env.hf.space
"""

import asyncio
import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

# Load .env file if present (before os.getenv calls)
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

_load_dotenv()

#  Required configuration variables 
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN          = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME  = os.getenv("LOCAL_IMAGE_NAME")
API_KEY           = HF_TOKEN or os.getenv("API_KEY")
# 

ENV_URL       = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME     = "container-stacking"
BENCHMARK     = "container-port-env"
MAX_STEPS     = 200  # hard mode has 50 containers, safety ceiling
SUCCESS_SCORE_THRESHOLD = 0.5


#  Logging helpers (exact SST format) 

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


#  Agents 

def greedy_decide(obs: dict) -> int:
    stacks     = obs["stack_states"]
    current    = obs.get("current_container")
    max_height = obs["max_height"]
    upcoming   = set(obs.get("upcoming_retrievals", []))

    if current is None:
        return 0

    cur_priority = current["priority"]
    best_stack, best_score = -1, float("-inf")

    for i, stack in enumerate(stacks):
        depth = len(stack)
        if depth >= max_height:
            continue
        score = 0.0
        accessibility = (max_height - depth) / max_height
        score += accessibility * (4 - cur_priority)

        if depth > 0:
            top_p = stack[-1]["priority"]
            if cur_priority > top_p:
                score -= 10.0 * (cur_priority - top_p)
            elif cur_priority < top_p:
                score += 3.0

        if current["id"] in upcoming:
            score += 5.0 * accessibility

        if depth > 0:
            score += 0.5

        if score > best_score:
            best_score = score
            best_stack = i

    if best_stack == -1:
        for i, stack in enumerate(stacks):
            if len(stack) < max_height:
                return i
    return max(best_stack, 0)


def llm_decide(obs: dict, client: OpenAI) -> int:
    stacks     = obs["stack_states"]
    current    = obs.get("current_container")
    n_stacks   = obs["n_stacks"]
    max_height = obs["max_height"]
    upcoming   = obs.get("upcoming_retrievals", [])
    difficulty = obs.get("difficulty", "medium")

    lines = []
    for i, stack in enumerate(stacks):
        if not stack:
            lines.append(f"  Stack {i}: EMPTY (0/{max_height})")
        else:
            contents = ", ".join(f"{c['id']}(p{c['priority']})" for c in stack)
            lines.append(
                f"  Stack {i}: [{contents}] depth={len(stack)}/{max_height},"
                f" top=priority-{stack[-1]['priority']}"
            )

    prompt = (
        f"You are a container yard planner. Minimize rehandle operations.\n"
        f"Priority 1=URGENT (retrieved first), 2=Normal, 3=Low.\n"
        f"RULE: containers above the target at retrieval = rehandles (costly).\n\n"
        f"DIFFICULTY: {difficulty}\n"
        f"UPCOMING RETRIEVALS: {upcoming or 'Unknown (hard mode)'}\n\n"
        f"CONTAINER TO PLACE: id={current['id']}, priority={current['priority']}, "
        f"weight={current['weight']}kg\n\n"
        f"STACKS (bottom->top):\n" + "\n".join(lines) + "\n\n"
        f"Reply ONLY with valid JSON: {{\"stack_index\": <int 0-{n_stacks-1}>}}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=64,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        decision = json.loads(text.strip())
        idx = int(decision["stack_index"])
        if 0 <= idx < n_stacks and len(obs["stack_states"][idx]) < max_height:
            return idx
    except Exception as e:
        print(f"[DEBUG] LLM fallback: {e}", file=sys.stderr, flush=True)

    return greedy_decide(obs)


#  Episode runner 

async def run_episode(
    url: str,
    difficulty: str = "medium",
    use_llm: bool = False,
) -> float:
    import websockets

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
    if not ws_url.endswith("/ws"):
        ws_url = ws_url.rstrip("/") + "/ws"

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if use_llm else None
    model_label = MODEL_NAME if use_llm else "greedy"

    log_start(task=f"{TASK_NAME}-{difficulty}", env=BENCHMARK, model=model_label)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "reset", "data": {"difficulty": difficulty}}))
            resp = json.loads(await ws.recv())
            payload = resp.get("data", {})
            obs = payload.get("observation", payload)

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                action_idx = llm_decide(obs, client) if use_llm else greedy_decide(obs)

                await ws.send(json.dumps({
                    "type": "step",
                    "data": {"stack_index": action_idx},
                }))
                resp = json.loads(await ws.recv())
                payload = resp.get("data", {})
                obs    = payload.get("observation", payload)
                reward = float(payload.get("reward", obs.get("last_reward", 0.0)) or obs.get("last_reward", 0.0))
                done   = payload.get("done", obs.get("done", False))
                error  = payload.get("error", None)

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=str(action_idx), reward=reward, done=done, error=error)

                if done:
                    break

            # Fetch final score
            await ws.send(json.dumps({"type": "state"}))
            state_resp = json.loads(await ws.recv())
            state = state_resp.get("data", {})
            score = float(state.get("score", obs.get("score", 0.0)))
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def run_all(url: str, use_llm: bool = False) -> None:
    for diff in ["easy", "medium", "hard"]:
        await run_episode(url, difficulty=diff, use_llm=use_llm)


#  Entry point 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Container Port Baseline Agent")
    parser.add_argument("--url",        default=ENV_URL)
    parser.add_argument("--difficulty", default="all",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--use-llm",    action="store_true",
                        help="Use LLM agent via HF router (requires HF_TOKEN)")
    args = parser.parse_args()

    if args.difficulty == "all":
        asyncio.run(run_all(args.url, use_llm=args.use_llm))
    else:
        asyncio.run(run_episode(args.url, difficulty=args.difficulty, use_llm=args.use_llm))
