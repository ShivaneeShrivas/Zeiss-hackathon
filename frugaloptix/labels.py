"""Every reported number carries exactly one provenance label (Round 1 promise)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Label(str, Enum):
    MEASURED = "Measured"    # real code / device on real inputs
    SIMULATED = "Simulated"  # depends on the optical simulation
    ESTIMATED = "Estimated"  # assumption-based model (cost, CO2)
    TARGET = "Target"        # goal, not yet verified


@dataclass
class Metric:
    name: str
    value: object
    unit: str
    label: Label
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label.value
        return d
