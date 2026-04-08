"""Template-compatible OpenEnv models for CLI packaging checks."""

from typing import Any, Dict

from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field


class EmailTriageAction(Action):
    category: str = Field(..., description="Email category routing decision")
    priority: str = Field(..., description="Priority decision")
    team: str = Field(..., description="Owning team")
    mark_spam: bool = Field(default=False, description="Whether email is spam/abuse")
    response_template: str = Field(default="", description="Response summary")


class EmailTriageObservation(Observation):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default="")
    task_difficulty: str = Field(default="")
    step_index: int = Field(default=0)
    max_steps: int = Field(default=0)
    processed_count: int = Field(default=0)
    remaining_count: int = Field(default=0)
    last_feedback: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
