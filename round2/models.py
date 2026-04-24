"""Template-compatible OpenEnv models for CLI packaging checks."""

from typing import Any, Dict

from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field


class ConflictAction(Action):
    intent: str = Field(..., description="Conflict-resolution intent")
    owner: str = Field(..., description="Owner/team assignment")
    priority: str = Field(..., description="Priority level")
    proposed_slot: str = Field(default="", description="Optional time slot proposal")
    needs_clarification: bool = Field(default=False, description="Whether missing context blocks execution")
    message_template: str = Field(default="", description="Action explanation or message template")


class ConflictObservation(Observation):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default="")
    task_difficulty: str = Field(default="")
    step_index: int = Field(default=0)
    max_steps: int = Field(default=0)
    processed_count: int = Field(default=0)
    remaining_count: int = Field(default=0)
    last_feedback: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
