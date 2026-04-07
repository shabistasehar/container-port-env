from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ContainerInfo(BaseModel):
    id: str
    priority: int = Field(..., ge=1, le=3)
    weight: float

class StackEntry(BaseModel):
    id: str
    priority: int

class ContainerAction(BaseModel):
    stack_index: int = Field(..., description="Which stack (0-indexed) to place the current container into")

class ContainerObservation(BaseModel):
    stack_states: List[List[Dict[str, Any]]]
    current_container: Optional[Dict[str, Any]]
    upcoming_retrievals: List[str]
    rehandle_count: int
    step: int
    containers_remaining: int
    n_stacks: int
    max_height: int
    difficulty: str
    last_reward: float
    done: bool

class ContainerState(BaseModel):
    stack_states: List[List[Dict[str, Any]]]
    rehandle_count: int
    step: int
    score: float
    difficulty: str
    done: bool
    total_reward: float
