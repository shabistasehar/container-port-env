from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ContainerAction(Action):
    """Place the current container into a stack."""

    stack_index: int = Field(
        ...,
        description="Zero-indexed stack to place the incoming container into",
        ge=0,
    )


class ContainerObservation(Observation):
    """Observation returned after each step."""

    stack_states: list[list[dict[str, Any]]] = Field(
        ..., description="Each stack is a list of {id, priority} dicts (bottom->top)"
    )
    current_container: dict[str, Any] | None = Field(
        None, description="Container to place now: {id, priority, weight}"
    )
    upcoming_retrievals: list[str] = Field(
        default_factory=list,
        description="IDs of next containers to be retrieved (lookahead)",
    )
    rehandle_count: int = Field(0, description="Cumulative rehandles so far")
    step: int = Field(0, description="Steps completed")
    containers_remaining: int = Field(0)
    n_stacks: int = Field(0)
    max_height: int = Field(0)
    difficulty: str = Field("medium")
    last_reward: float = Field(0.0)
    score: float = Field(
        0.5,
        description="Normalized score strictly in (0.0, 1.0)",
        gt=0.0,
        lt=1.0,
    )
    done: bool = Field(False)
