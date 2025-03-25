from dataclasses import dataclass


@dataclass
class Sound:
    time: float
    frequency: int


@dataclass
class LED:
    time: float
    color: str
