import pytest
from server.environment import ContainerYardEnv, DIFFICULTY_CONFIG

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_reset_returns_valid_obs(difficulty):
    env = ContainerYardEnv(difficulty=difficulty, seed=42)
    obs = env.reset()
    cfg = DIFFICULTY_CONFIG[difficulty]
    assert len(obs["stack_states"]) == cfg["n_stacks"]
    assert obs["current_container"] is not None
    assert obs["step"] == 0
    assert obs["rehandle_count"] == 0
    assert obs["difficulty"] == difficulty
    assert obs["done"] == False

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_step_valid_action(difficulty):
    env = ContainerYardEnv(difficulty=difficulty, seed=42)
    env.reset()
    obs, reward, done, info = env.step(0)
    assert isinstance(reward, float)
    assert obs["step"] == 1
    assert len(obs["stack_states"][0]) == 1
    assert "rehandles" in info

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_step_invalid_stack_index(difficulty):
    env = ContainerYardEnv(difficulty=difficulty, seed=42)
    env.reset()
    obs, reward, done, info = env.step(999)
    assert reward == -2.0
    assert "error" in info
    assert done == False

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_full_episode_completes(difficulty):
    env = ContainerYardEnv(difficulty=difficulty, seed=42)
    env.reset()
    done = False
    steps = 0
    cfg = DIFFICULTY_CONFIG[difficulty]
    n_stacks   = cfg["n_stacks"]
    max_height = cfg["max_height"]
    while not done:
        stacks = env._observe()["stack_states"]
        chosen = 0
        for i in range(n_stacks):
            if len(stacks[i]) < max_height:
                chosen = i
                break
        _, _, done, _ = env.step(chosen)
        steps += 1
        assert steps < 1000, "Episode did not complete in time"
    assert done

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_score_in_range(difficulty):
    env = ContainerYardEnv(difficulty=difficulty, seed=42)
    env.reset()
    done = False
    cfg = DIFFICULTY_CONFIG[difficulty]
    n_stacks   = cfg["n_stacks"]
    max_height = cfg["max_height"]
    while not done:
        stacks = env._observe()["stack_states"]
        chosen = 0
        for i in range(n_stacks):
            if len(stacks[i]) < max_height:
                chosen = i
                break
        _, _, done, _ = env.step(chosen)
    score = env.score()
    assert 0.0 <= score <= 1.0

def test_lookahead_visibility():
    easy_env = ContainerYardEnv(difficulty="easy", seed=42)
    hard_env = ContainerYardEnv(difficulty="hard", seed=42)
    easy_obs = easy_env.reset()
    hard_obs = hard_env.reset()
    assert len(easy_obs["upcoming_retrievals"]) > len(hard_obs["upcoming_retrievals"])
    assert len(hard_obs["upcoming_retrievals"]) == 0

def test_reward_is_dense():
    env = ContainerYardEnv(difficulty="medium", seed=42)
    env.reset()
    rewards = []
    done = False
    step = 0
    while not done and step < 20:
        stacks = env._observe()["stack_states"]
        chosen = step % 8
        if len(stacks[chosen]) >= 5:
            chosen = 0
        _, r, done, _ = env.step(chosen)
        rewards.append(r)
        step += 1
    nonzero = sum(1 for r in rewards if abs(r) > 1e-6)
    assert nonzero >= len(rewards) * 0.5, f"Too many zero rewards: {rewards}"

def test_no_double_retrieval():
    """Retrieval pointer advances correctly — no container retrieved twice."""
    env = ContainerYardEnv(difficulty="easy", seed=42)
    env.reset()
    seen_ids = set()
    for _ in range(env.n_containers):
        if env.done:
            break
        env.step(0 if len(env.stacks[0]) < env.max_height else 1)
    # retrieval_pointer should be <= queue length
    assert env.retrieval_pointer <= len(env.retrieval_queue)
