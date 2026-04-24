from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ActionIntent(str, Enum):
    ROUTE_MESSAGE = "route_message"
    PROPOSE_PLAN = "propose_plan"
    RESCHEDULE_EVENT = "reschedule_event"
    DELEGATE_TASK = "delegate_task"
    ASK_CLARIFICATION = "ask_clarification"
    FINALIZE_ITINERARY = "finalize_itinerary"


class Owner(str, Enum):
    SELF = "self"
    WORK = "work"
    FAMILY = "family"
    TRAVEL = "travel"
    FINANCE = "finance"
    LEGAL = "legal"


class ConflictExpectation(BaseModel):
    intent: ActionIntent
    owner: Owner
    priority: Priority
    require_slot: bool = False
    expected_slot_hint: str = ""
    block_if_missing_context: bool = False
    required_keywords: List[str] = Field(default_factory=list)


class ConflictCase(BaseModel):
    conflict_id: str
    source: str
    summary: str
    constraints: List[str] = Field(default_factory=list)
    expected: ConflictExpectation


class TaskDefinition(BaseModel):
    id: str
    title: str
    difficulty: str
    description: str
    max_steps: int
    conflicts: List[ConflictCase]


class ConflictDecision(BaseModel):
    conflict_id: str
    intent: ActionIntent
    owner: Owner
    priority: Priority
    proposed_slot: str = ""
    needs_clarification: bool = False
    message_template: str = ""


class ConflictAction(BaseModel):
    intent: ActionIntent
    owner: Owner
    priority: Priority
    proposed_slot: str = Field(default="", max_length=80)
    needs_clarification: bool = False
    message_template: str = Field(default="", max_length=500)

    @field_validator("proposed_slot", mode="before")
    @classmethod
    def normalize_slot(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("message_template", mode="before")
    @classmethod
    def normalize_template(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class ConflictObservation(BaseModel):
    task_id: str
    task_difficulty: str
    step_index: int
    max_steps: int
    processed_count: int
    remaining_count: int
    current_conflict: Optional[ConflictCase] = None
    history: List[ConflictDecision] = Field(default_factory=list)
    open_risks: List[str] = Field(default_factory=list)
    last_feedback: str = ""


class ConflictReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)


class ConflictState(BaseModel):
    task_id: str
    task_difficulty: str
    step_index: int
    max_steps: int
    done: bool
    cumulative_reward: float = 0.0
    average_step_score: float = 0.0
    final_score: float = 0.0
    decisions: List[ConflictDecision] = Field(default_factory=list)
    score_history: List[float] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    clarification_requests: int = 0
    last_action_error: Optional[str] = None


class ConflictStepResult(BaseModel):
    observation: ConflictObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
