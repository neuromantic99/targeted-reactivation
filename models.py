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
    common_average_reference_check_less_restrictive: List[bool]
    frequency_check: List[bool]
    super_ripple_check: List[bool]
    super_ripple_check_less_restrictive: List[bool]
    length_recording: float
    state: List[str]
    state_lengths: dict[str, float]


class CandidateSpindle(BaseModel):
    onset: int
    offset: int
    peak_amplitude: float
    peak_idx: int
    frequency: float


class SpindleCache(BaseModel):
    spindles: List[CandidateSpindle]
    length_recording: float
    state: List[str]
    state_lengths: dict[str, float]


class SlowOscillationCache(BaseModel):
    starts: List[int]
    ends: List[int]
    state: List[str]
    state_lengths: dict[str, float]
    downsampled_lfp: List[float]
    downsample_factor: int
