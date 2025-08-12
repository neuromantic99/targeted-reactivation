from dataclasses import dataclass
from typing import List

from pydantic import BaseModel
from ripples.models import CandidateEvent


@dataclass
class Sound:
    time: float
    frequency: int


@dataclass
class LED:
    time: float
    color: str


class RipplesCache(BaseModel):
    candidate_events: List[CandidateEvent]
    common_average_reference_check: List[bool]
    frequency_check: List[bool]
    super_ripple_check: List[bool]
    length_recording: float
    state: List[str]
    state_lengths: dict[str, float]
