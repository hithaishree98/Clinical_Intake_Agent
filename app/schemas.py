from pydantic import BaseModel, Field
from typing import Dict, List

class SubjectiveOut(BaseModel):
    chief_complaint: str = ""
    opqrst: Dict[str, str] = Field(default_factory=lambda: {
        "onset": "",
        "provocation": "",
        "quality": "",
        "radiation": "",
        "severity": "",
        "timing": "",
    })
    is_complete: bool = False
    reply: str = ""

class MedsOut(BaseModel):
    medications: List[Dict[str, str]] = Field(default_factory=list)
    reply: str = ""
