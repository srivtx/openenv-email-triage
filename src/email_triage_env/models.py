from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    BILLING = "billing"
    BUG = "bug"
    ACCOUNT = "account"
    SALES = "sales"
    ABUSE = "abuse"
    LEGAL = "legal"
    OTHER = "other"


class Team(str, Enum):
    SUPPORT = "support"
    BILLING = "billing"
    ENGINEERING = "engineering"
    SALES = "sales"
    TRUST_SAFETY = "trust_safety"
    LEGAL = "legal"


class EmailExpectation(BaseModel):
    category: Category
    priority: Priority
    team: Team
    mark_spam: bool = False
    required_keywords: List[str] = Field(default_factory=list)


class EmailCase(BaseModel):
    email_id: str
    sender: str
    subject: str
    body: str
    expected: EmailExpectation


class TaskDefinition(BaseModel):
    id: str
    title: str
    difficulty: str
    description: str
    max_steps: int
    emails: List[EmailCase]


class EmailDecision(BaseModel):
    email_id: str
    category: Category
    priority: Priority
    team: Team
    mark_spam: bool = False
    response_template: str = ""


class EmailTriageAction(BaseModel):
    category: Category
    priority: Priority
    team: Team
    mark_spam: bool = False
    response_template: str = Field(default="", max_length=400)

    @field_validator("response_template", mode="before")
    @classmethod
    def normalize_template(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class EmailTriageObservation(BaseModel):
    task_id: str
    task_difficulty: str
    step_index: int
    max_steps: int
    processed_count: int
    remaining_count: int
    current_email: Optional[EmailCase] = None
    history: List[EmailDecision] = Field(default_factory=list)
    last_feedback: str = ""


class EmailTriageReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)


class EmailTriageState(BaseModel):
    task_id: str
    task_difficulty: str
    step_index: int
    max_steps: int
    done: bool
    cumulative_reward: float = 0.0
    average_step_score: float = 0.0
    final_score: float = 0.0
    decisions: List[EmailDecision] = Field(default_factory=list)
    score_history: List[float] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    last_action_error: Optional[str] = None


class EmailTriageStepResult(BaseModel):
    observation: EmailTriageObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
