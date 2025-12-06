from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ContextState:
    member_type: Optional[str] = None
    grade_of_concrete: Optional[str] = None
    exposure_condition: Optional[str] = None
    cement_type: Optional[str] = None
    test_results: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextState':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class ChatResponse:
    reply: str
    follow_up_needed: bool = False
    missing_fields: List[str] = field(default_factory=list)
    context_state: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, str]] = field(default_factory=list)
