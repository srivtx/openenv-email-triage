"""Top-level package exports for OpenEnv CLI packaging checks."""

try:
    # Preferred path when OpenEnv dependencies are installed.
    from .client import PersonalAssistantConflictEnv
    from .models import ConflictAction, ConflictObservation
except Exception:
    # Local testing fallback that avoids hard dependency on openenv-core imports.
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parent
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from assistant_conflict_env.environment import PersonalAssistantConflictEnv
    from assistant_conflict_env.models import ConflictAction, ConflictObservation

__all__ = [
    "ConflictAction",
    "ConflictObservation",
    "PersonalAssistantConflictEnv",
]
