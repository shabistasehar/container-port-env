import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

@dataclass
class Container:
    id: str
    priority: int   # 1=urgent, 2=normal, 3=low
    weight: float

DIFFICULTY_CONFIG = {
    "easy": {
        "n_stacks": 6,
        "max_height": 4,
        "n_containers": 20,
        "retrieval_interval": 5,
        "lookahead": 5,
        "priority_weights": [0.4, 0.4, 0.2],
    },
    "medium": {
        "n_stacks": 8,
        "max_height": 5,
        "n_containers": 35,
        "retrieval_interval": 5,
        "lookahead": 3,
        "priority_weights": [0.33, 0.34, 0.33],
    },
    "hard": {
        "n_stacks": 10,
        "max_height": 6,
        "n_containers": 50,
        "retrieval_interval": 4,
        "lookahead": 0,
        "priority_weights": [0.25, 0.35, 0.40],
    },
}

class ContainerYardEnv:
    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        assert difficulty in DIFFICULTY_CONFIG, f"difficulty must be one of {list(DIFFICULTY_CONFIG.keys())}"
        self.difficulty = difficulty
        self.seed = seed
        cfg = DIFFICULTY_CONFIG[difficulty]
        self.n_stacks = cfg["n_stacks"]
        self.max_height = cfg["max_height"]
        self.n_containers = cfg["n_containers"]
        self.retrieval_interval = cfg["retrieval_interval"]
        self.lookahead = cfg["lookahead"]
        self.priority_weights = cfg["priority_weights"]
        self.reset()

    def reset(self) -> Dict[str, Any]:
        if self.seed is not None:
            random.seed(self.seed)
        self.stacks: List[List[Container]] = [[] for _ in range(self.n_stacks)]
        self.rehandle_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.manifest: List[Container] = self._generate_manifest()
        self.retrieval_queue: List[str] = self._generate_retrieval_queue()
        self.retrieval_pointer = 0
        self.current_idx = 0
        return self._observe(last_reward=0.0)

    def _generate_manifest(self) -> List[Container]:
        containers = []
        for i in range(self.n_containers):
            priority = random.choices([1, 2, 3], weights=self.priority_weights)[0]
            containers.append(Container(
                id=f"C{i:03d}",
                priority=priority,
                weight=round(random.uniform(5.0, 30.0), 1)
            ))
        return containers

    def _generate_retrieval_queue(self) -> List[str]:
        ids_by_priority = {1: [], 2: [], 3: []}
        for c in self.manifest:
            ids_by_priority[c.priority].append(c.id)
        for p in ids_by_priority:
            random.shuffle(ids_by_priority[p])
        queue = ids_by_priority[1] + ids_by_priority[2] + ids_by_priority[3]
        return queue

    def step(self, stack_index: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._observe(0.0), 0.0, True, {"error": "episode already done"}

        if stack_index < 0 or stack_index >= self.n_stacks:
            reward = -2.0
            self.total_reward += reward
            return self._observe(reward), reward, False, {"error": f"invalid stack_index {stack_index}, must be 0-{self.n_stacks-1}"}

        if len(self.stacks[stack_index]) >= self.max_height:
            reward = -2.0
            self.total_reward += reward
            return self._observe(reward), reward, False, {"error": f"stack {stack_index} is full (height {self.max_height})"}

        current = self.manifest[self.current_idx]
        self.stacks[stack_index].append(current)
        placement_reward = self._placement_reward(stack_index, current)

        self.current_idx += 1
        self.step_count += 1

        retrieval_cost = 0.0
        retrievals_done = []
        if self.step_count % self.retrieval_interval == 0:
            cost, done_ids = self._trigger_retrieval()
            retrieval_cost = cost
            retrievals_done = done_ids

        reward = placement_reward - retrieval_cost
        self.total_reward += reward
        self.done = (self.current_idx >= len(self.manifest))

        return self._observe(reward), reward, self.done, {
            "rehandles": self.rehandle_count,
            "step": self.step_count,
            "placement_reward": round(placement_reward, 4),
            "retrieval_cost": round(retrieval_cost, 4),
            "retrievals_done": retrievals_done,
        }

    def _placement_reward(self, stack_index: int, container: Container) -> float:
        # stack_depth = zero-based index of the just-placed container
        stack_depth = len(self.stacks[stack_index]) - 1
        accessibility = (self.max_height - stack_depth) / self.max_height
        priority_weight = (4 - container.priority) / 3.0  # priority 11.0, 20.67, 30.33

        base = 0.3 * accessibility * priority_weight

        # Bonus: high-priority container placed near top (accessible for fast retrieval)
        if container.priority == 1 and stack_depth <= 1:
            base += 0.15

        # Penalty: placing lower-priority on top of higher-priority container (causes future rehandles)
        if stack_depth > 0:
            top_container = self.stacks[stack_index][-2]  # container directly below
            if container.priority > top_container.priority:
                base -= 0.2 * (container.priority - top_container.priority) / 2.0

        return round(base, 4)

    def _trigger_retrieval(self) -> Tuple[float, List[str]]:
        total_cost = 0.0
        done_ids = []
        for _ in range(2):
            if self.retrieval_pointer >= len(self.retrieval_queue):
                break
            target_id = self.retrieval_queue[self.retrieval_pointer]
            self.retrieval_pointer += 1
            cost = self._retrieve(target_id)
            total_cost += cost
            done_ids.append(target_id)
        return total_cost, done_ids

    def _retrieve(self, target_id: str) -> float:
        for stack in self.stacks:
            for i, c in enumerate(stack):
                if c.id == target_id:
                    rehandles = len(stack) - 1 - i  # containers above target
                    self.rehandle_count += rehandles
                    stack.pop(i)
                    return round(rehandles * 0.4, 4)
        return 0.0  # container not yet in yard — no penalty

    def _get_upcoming_retrievals(self) -> List[str]:
        start = self.retrieval_pointer
        end = min(start + self.lookahead, len(self.retrieval_queue))
        return self.retrieval_queue[start:end]

    def _observe(self, last_reward: float = 0.0) -> Dict[str, Any]:
        stack_states = []
        for s in self.stacks:
            stack_states.append([{"id": c.id, "priority": c.priority} for c in s])

        current = None
        if self.current_idx < len(self.manifest):
            c = self.manifest[self.current_idx]
            current = {"id": c.id, "priority": c.priority, "weight": c.weight}

        return {
            "stack_states": stack_states,
            "current_container": current,
            "upcoming_retrievals": self._get_upcoming_retrievals(),
            "rehandle_count": self.rehandle_count,
            "step": self.step_count,
            "containers_remaining": len(self.manifest) - self.current_idx,
            "n_stacks": self.n_stacks,
            "max_height": self.max_height,
            "difficulty": self.difficulty,
            "last_reward": last_reward,
            "done": self.done,
        }

    def get_state(self) -> Dict[str, Any]:
        obs = self._observe()
        obs["score"] = self.score()
        obs["total_reward"] = round(self.total_reward, 4)
        return obs

    def score(self) -> float:
        """Normalized score in [0.0, 1.0]. Based on actual retrievals attempted."""
        n_retrieved = self.retrieval_pointer  # only count retrievals that actually happened
        worst_case = n_retrieved * (self.max_height - 1)
        if worst_case == 0:
            return 1.0
        score = max(0.0, 1.0 - self.rehandle_count / worst_case)
        return round(min(score, 1.0), 4)
