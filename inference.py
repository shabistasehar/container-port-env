#!/usr/bin/env python3
"""
Container Port OpenEnv — Baseline Inference Script
SST x Meta PyTorch OpenEnv Hackathon

Required environment variables (or set below):
  HF_TOKEN      - Your Hugging Face token
  API_BASE_URL  - LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    - Model identifier (default: meta-llama/Llama-3.1-8B-Instruct)

Usage:
  python inference.py
  python inference.py --url https://YOUR_USERNAME-container-port-env.hf.space --difficulty all
  python inference.py --difficulty easy
"""

import os
import sys
import json
import asyncio
import argparse
import websockets
from openai import OpenAI

#  Required configuration variables 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")   # set your HF token here or via env var
# 

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


def _llm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at HF Inference Router."""
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def greedy_decide(obs: dict) -> int:
    """
    Greedy heuristic agent — no LLM call.
    Scores each valid stack by accessibility and priority compatibility.
    """
    stacks    = obs["stack_states"]
    current   = obs.get("current_container")
    max_height = obs["max_height"]
    upcoming  = set(obs.get("upcoming_retrievals", []))

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
            top_priority = stack[-1]["priority"]
            if cur_priority > top_priority:
                score -= 10.0 * (cur_priority - top_priority)
            elif cur_priority < top_priority:
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
    return best_stack


def llm_decide(obs: dict) -> int:
    """Use HF-hosted LLM via OpenAI-compatible client to choose a stack."""
    stacks    = obs["stack_states"]
    current   = obs.get("current_container")
    n_stacks  = obs["n_stacks"]
    max_height = obs["max_height"]
    upcoming  = obs.get("upcoming_retrievals", [])
    difficulty = obs.get("difficulty", "medium")

    stack_lines = []
    for i, stack in enumerate(stacks):
        if not stack:
            stack_lines.append(f"  Stack {i}: EMPTY (0/{max_height})")
        else:
            contents = ", ".join(f"{c['id']}(p{c['priority']})" for c in stack)
            stack_lines.append(
                f"  Stack {i}: [{contents}] depth={len(stack)}/{max_height},"
                f" top=priority-{stack[-1]['priority']}"
            )

    prompt = (
        f"You are an expert container yard planner.\n"
        f"TASK: Place the incoming container into a stack to MINIMIZE future rehandle operations.\n"
        f"RULE: When a container is retrieved, every container ON TOP of it must be moved (rehandle).\n"
        f"Priority 1=URGENT (retrieved first), 2=Normal, 3=Low (retrieved last).\n\n"
        f"DIFFICULTY: {difficulty}\n"
        f"UPCOMING RETRIEVALS (next to be retrieved, in order): "
        f"{upcoming if upcoming else 'Unknown (hard mode)'}\n\n"
        f"CONTAINER TO PLACE: id={current['id']}, priority={current['priority']}, "
        f"weight={current['weight']}kg\n\n"
        f"STACK STATES (bottomtop):\n" + "\n".join(stack_lines) + "\n\n"
        f"Respond with ONLY valid JSON: {{\"stack_index\": <integer 0-{n_stacks-1}>}}"
    )

    try:
        client = _llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=64,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        # strip markdown fences if model wraps in ```json ... ```
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        decision = json.loads(text.strip())
        idx = int(decision["stack_index"])
        if 0 <= idx < n_stacks and len(obs["stack_states"][idx]) < max_height:
            return idx
    except Exception as e:
        print(f"  [LLM fallback: {e}]", file=sys.stderr)

    return greedy_decide(obs)


async def run_episode(url: str, difficulty: str = "medium", use_llm: bool = False) -> float:
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
    if not ws_url.endswith("/ws"):
        ws_url = ws_url.rstrip("/") + "/ws"

    #  [START] log 
    print(json.dumps({"type": "[START]", "task": difficulty, "difficulty": difficulty,
                      "env_url": url, "model": MODEL_NAME if use_llm else "greedy"}))
    sys.stdout.flush()

    total_reward = 0.0
    step = 0

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "reset", "difficulty": difficulty}))
        resp = json.loads(await ws.recv())
        obs  = resp["observation"]

        while not obs.get("done", False):
            action_idx = llm_decide(obs) if use_llm else greedy_decide(obs)

            await ws.send(json.dumps({"type": "step", "action": {"stack_index": action_idx}}))
            resp  = json.loads(await ws.recv())
            obs   = resp["observation"]
            reward = resp["reward"]
            done  = resp["done"]
            total_reward += reward
            step += 1

            #  [STEP] log 
            print(json.dumps({
                "type": "[STEP]",
                "step": step,
                "action": action_idx,
                "reward": round(reward, 4),
                "total_reward": round(total_reward, 4),
                "done": done,
                "rehandle_count": obs["rehandle_count"],
            }))
            sys.stdout.flush()

        # fetch final state for score
        await ws.send(json.dumps({"type": "state"}))
        state_resp = json.loads(await ws.recv())
        state = state_resp["state"]

    final_score = state.get("score", 0.0)

    #  [END] log 
    print(json.dumps({
        "type": "[END]",
        "task": difficulty,
        "difficulty": difficulty,
        "total_reward": round(total_reward, 4),
        "final_score": final_score,
        "total_steps": step,
        "rehandle_count": state.get("rehandle_count", 0),
    }))
    sys.stdout.flush()

    return final_score


async def run_all(url: str, use_llm: bool = False):
    for diff in ["easy", "medium", "hard"]:
        await run_episode(url, difficulty=diff, use_llm=use_llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Container Port Baseline Agent")
    parser.add_argument("--url",        default=ENV_URL)
    parser.add_argument("--difficulty", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--use-llm",    action="store_true")
    args = parser.parse_args()

    if args.difficulty == "all":
        asyncio.run(run_all(args.url, use_llm=args.use_llm))
    else:
        asyncio.run(run_episode(args.url, difficulty=args.difficulty, use_llm=args.use_llm))
