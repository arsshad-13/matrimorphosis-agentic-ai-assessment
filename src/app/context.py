
from dataclasses import dataclass


@dataclass
class Ctx:
    """Per-request runtime context for the agent and its tools."""

    session_id: str
    user_id: str
    token: str = ""
