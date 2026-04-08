import pytest
from fastapi.testclient import TestClient

from models import ContainerAction
from server.app import app
from server.environment import ContainerYardEnvironment, DIFFICULTY_CONFIG


def as_dict(observation):
    return observation.model_dump() if hasattr(observation, "model_dump") else observation


#  Unit tests: pure environment logic (no HTTP) 

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_reset_returns_valid_obs(difficulty):
    env = ContainerYardEnvironment()
    obs = as_dict(env.reset(difficulty=difficulty, seed=42))
    cfg = DIFFICULTY_CONFIG[difficulty]
    assert len(obs["stack_states"]) == cfg["n_stacks"]
    assert obs["current_container"] is not None
    assert obs["step"] == 0
    assert obs["rehandle_count"] == 0
    assert obs["difficulty"] == difficulty
    assert obs["done"] is False


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_step_valid_action(difficulty):
    env = ContainerYardEnvironment()
    env.reset(difficulty=difficulty, seed=42)
    obs = as_dict(env.step(ContainerAction(stack_index=0)))
    assert obs["step"] == 1
    assert len(obs["stack_states"][0]) == 1
    assert isinstance(obs["last_reward"], float)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_step_invalid_action_penalized(difficulty):
    env = ContainerYardEnvironment()
    env.reset(difficulty=difficulty, seed=42)
    obs = as_dict(env.step(ContainerAction(stack_index=999)))
    assert obs["last_reward"] == -2.0


def test_score_in_range():
    env = ContainerYardEnvironment()
    env.reset(difficulty="medium", seed=42)
    done = False
    while not done:
        stacks = as_dict(env._observe())["stack_states"]
        chosen = next(
            (i for i, stack in enumerate(stacks) if len(stack) < env.max_height), 0
        )
        obs = as_dict(env.step(ContainerAction(stack_index=chosen)))
        done = obs["done"]
    # Score must be strictly between 0 and 1 (grader requirement)
    assert 0.0 < env.score() < 1.0


def test_score_varies_across_seeds():
    scores = []
    for seed in [1, 7, 13, 21, 42]:
        env = ContainerYardEnvironment()
        env.reset(difficulty="medium", seed=seed)
        done = False
        while not done:
            stacks = as_dict(env._observe())["stack_states"]
            chosen = next(
                (i for i, stack in enumerate(stacks) if len(stack) < env.max_height), 0
            )
            obs = as_dict(env.step(ContainerAction(stack_index=chosen)))
            done = obs["done"]
        scores.append(env.score())

    # Avoid disqualification: grader must not return a constant score.
    assert len(set(scores)) > 1, f"Scores are constant across seeds: {scores}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_full_episode_completes(difficulty):
    env = ContainerYardEnvironment()
    env.reset(difficulty=difficulty, seed=42)
    cfg = DIFFICULTY_CONFIG[difficulty]
    done = False
    steps = 0
    while not done:
        stacks = as_dict(env._observe())["stack_states"]
        chosen = next(
            (i for i, s in enumerate(stacks) if len(s) < cfg["max_height"]), 0
        )
        obs = as_dict(env.step(ContainerAction(stack_index=chosen)))
        done = obs["done"]
        steps += 1
        assert steps < 500, "Episode did not complete"
    assert done is True


def test_lookahead_visibility():
    easy_env = ContainerYardEnvironment()
    hard_env = ContainerYardEnvironment()
    easy_obs = as_dict(easy_env.reset(difficulty="easy", seed=42))
    hard_obs = as_dict(hard_env.reset(difficulty="hard", seed=42))
    assert len(easy_obs["upcoming_retrievals"]) > len(hard_obs["upcoming_retrievals"])
    assert len(hard_obs["upcoming_retrievals"]) == 0


def test_reward_is_dense():
    env = ContainerYardEnvironment()
    env.reset(difficulty="medium", seed=42)
    rewards = []
    done = False
    step = 0
    while not done and step < 20:
        stacks = as_dict(env._observe())["stack_states"]
        chosen = step % env.n_stacks
        if len(stacks[chosen]) >= env.max_height:
            chosen = 0
        obs = as_dict(env.step(ContainerAction(stack_index=chosen)))
        rewards.append(obs["last_reward"])
        done = obs["done"]
        step += 1
    nonzero = sum(1 for r in rewards if abs(r) > 1e-6)
    assert nonzero >= len(rewards) * 0.5, f"Too many zero rewards: {rewards}"


def test_no_double_retrieval():
    env = ContainerYardEnvironment()
    env.reset(difficulty="easy", seed=42)
    for _ in range(env.n_containers):
        if env.done:
            break
        stacks = env.stacks
        chosen = next(
            (i for i, s in enumerate(stacks) if len(s) < env.max_height), 0
        )
        env.step(ContainerAction(stack_index=chosen))
    assert env.retrieval_pointer <= len(env.retrieval_queue)


#  HTTP integration tests 

def test_health_route():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_web_ui_route():
    client = TestClient(app, follow_redirects=True)
    resp = client.get("/web")
    assert resp.status_code == 200


def test_http_reset_returns_observation():
    client = TestClient(app)
    resp = client.post("/reset", json={"difficulty": "easy"})
    assert resp.status_code == 200
    body = resp.json()
    obs = body.get("observation", body)
    assert obs.get("difficulty") == "easy"
    assert obs.get("step") == 0
    assert obs.get("containers_remaining") == DIFFICULTY_CONFIG["easy"]["n_containers"]


def test_http_reset_then_step_preserves_state():
    client = TestClient(app)

    reset_resp = client.post("/web/reset", json={"difficulty": "easy"})
    assert reset_resp.status_code == 200
    reset_body = reset_resp.json()

    session_id = reset_body.get("session_id") or reset_body.get("id")
    obs_after_reset = reset_body.get("observation", reset_body)
    assert obs_after_reset.get("step") == 0
    n_containers = DIFFICULTY_CONFIG["easy"]["n_containers"]
    assert obs_after_reset.get("containers_remaining") == n_containers

    step_payload = {"action": {"stack_index": 0}}
    if session_id:
        step_payload["session_id"] = session_id

    step_resp = client.post("/web/step", json=step_payload)
    assert step_resp.status_code == 200
    step_body = step_resp.json()
    obs_after_step = step_body.get("observation", step_body)

    assert obs_after_step.get("step") == 1
    assert obs_after_step.get("containers_remaining") == n_containers - 1
    assert len(obs_after_step["stack_states"][0]) == 1
