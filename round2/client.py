"""Template-compatible OpenEnv client for CLI packaging checks."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ConflictAction, ConflictObservation
except ImportError:
    from round2.models import ConflictAction, ConflictObservation


class PersonalAssistantConflictEnv(EnvClient[ConflictAction, ConflictObservation, State]):
    def _step_payload(self, action: ConflictAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[ConflictObservation]:
        raw_observation = payload.get("observation", {})
        if "metadata" not in raw_observation:
            raw_observation["metadata"] = {}

        try:
            observation = ConflictObservation.model_validate(raw_observation)
        except Exception:
            observation = ConflictObservation(metadata={"raw_observation": raw_observation})

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
