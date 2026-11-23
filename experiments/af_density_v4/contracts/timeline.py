"""Canonical Event Timeline Contracts and Schemas (V4).

Defines the universal event-timeline structure mandated in T-CAIREM Research Directive V4:
- Fields: subject_id, record_id, dataset_id, start_s, end_s, state, state_confidence,
          quality_class, observation_status, annotation_source.
- Strict half-open intervals [start_s, end_s) with end_s > start_s.
- Permitted rhythm states: AF, NON_AF, OTHER_ARRHYTHMIA, UNKNOWN.
- Separate storage for expert reference truth vs detector outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Sequence
import numpy as np


class RhythmState(str, Enum):
    AF = "AF"
    NON_AF = "NON_AF"
    OTHER_ARRHYTHMIA = "OTHER_ARRHYTHMIA"
    UNKNOWN = "UNKNOWN"


class QualityClass(str, Enum):
    CLEAN = "CLEAN"
    N1_QRS_OK = "N1_QRS_OK"
    N2_QRS_LOST = "N2_QRS_LOST"
    UNASSESSED = "UNASSESSED"


class ObservationStatus(str, Enum):
    OBSERVED = "OBSERVED"
    MISSING_DROPOUT = "MISSING_DROPOUT"
    CORRUPTED = "CORRUPTED"
    UNOBSERVABLE = "UNOBSERVABLE"


@dataclass(frozen=True)
class CanonicalInterval:
    """Represents a single half-open temporal interval [start_s, end_s)."""
    subject_id: str
    record_id: str
    dataset_id: str
    start_s: float
    end_s: float
    state: RhythmState
    state_confidence: float = 1.0
    quality_class: QualityClass = QualityClass.CLEAN
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    annotation_source: str = "reference_expert"

    def __post_init__(self):
        if self.end_s <= self.start_s:
            raise ValueError(f"Invalid interval: end_s ({self.end_s}) must be strictly greater than start_s ({self.start_s})")
        if not (0.0 <= self.state_confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.state_confidence}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["quality_class"] = self.quality_class.value
        d["observation_status"] = self.observation_status.value
        return d


@dataclass
class CanonicalTimeline:
    """Ordered sequence of canonical intervals for an individual subject/recording."""
    subject_id: str
    record_id: str
    dataset_id: str
    total_duration_s: float
    intervals: List[CanonicalInterval] = field(default_factory=list)

    def __post_init__(self):
        self._validate_and_sort()

    def _validate_and_sort(self):
        if not self.intervals:
            return
        # Sort by start time
        self.intervals.sort(key=lambda x: (x.start_s, x.end_s))
        # Validate ordering and non-overlapping within same stream
        for i in range(len(self.intervals) - 1):
            cur = self.intervals[i]
            nxt = self.intervals[i + 1]
            if cur.end_s > nxt.start_s + 1e-6:
                raise ValueError(
                    f"Overlapping intervals detected: [{cur.start_s}, {cur.end_s}) overlaps [{nxt.start_s}, {nxt.end_s})"
                )

    def add_interval(self, interval: CanonicalInterval):
        if interval.end_s > self.total_duration_s + 1e-3:
            self.total_duration_s = max(self.total_duration_s, interval.end_s)
        self.intervals.append(interval)
        self._validate_and_sort()

    def get_af_intervals(self) -> List[CanonicalInterval]:
        return [iv for iv in self.intervals if iv.state == RhythmState.AF]

    def get_unknown_intervals(self) -> List[CanonicalInterval]:
        return [iv for iv in self.intervals if iv.state == RhythmState.UNKNOWN or iv.observation_status != ObservationStatus.OBSERVED]

    def compute_af_burden(self, effective_horizon_s: Optional[float] = None) -> float:
        """Computes true AF burden = (Total AF seconds) / (Horizon seconds)."""
        T = effective_horizon_s or self.total_duration_s
        if T <= 0:
            return 0.0
        af_dur = sum(iv.duration_s for iv in self.get_af_intervals() if iv.start_s < T)
        return min(1.0, af_dur / T)

    def rasterize(self, step_s: float = 1.0, horizon_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Rasterizes the continuous timeline into uniform discrete time steps.
        
        Returns:
            rhythm_grid: int8 array where 1=AF, 0=Non-AF, 2=Other, -1=Unknown
            quality_grid: int8 array where 0=Clean, 1=N1, 2=N2
        """
        T = horizon_s or self.total_duration_s
        n_steps = int(np.ceil(T / step_s))
        rhythm_grid = np.zeros(n_steps, dtype=np.int8)  # Default Non-AF
        quality_grid = np.zeros(n_steps, dtype=np.int8)

        for iv in self.intervals:
            if iv.start_s >= T:
                continue
            idx_start = max(0, int(np.floor(iv.start_s / step_s)))
            idx_end = min(n_steps, int(np.ceil(iv.end_s / step_s)))

            val = 0
            if iv.state == RhythmState.AF:
                val = 1
            elif iv.state == RhythmState.OTHER_ARRHYTHMIA:
                val = 2
            elif iv.state == RhythmState.UNKNOWN or iv.observation_status != ObservationStatus.OBSERVED:
                val = -1

            q_val = 0
            if iv.quality_class == QualityClass.N1_QRS_OK:
                q_val = 1
            elif iv.quality_class == QualityClass.N2_QRS_LOST:
                q_val = 2

            rhythm_grid[idx_start:idx_end] = val
            quality_grid[idx_start:idx_end] = q_val

        return rhythm_grid, quality_grid

    def slice_window(self, start_s: float, window_duration_s: float) -> CanonicalTimeline:
        """Extracts a continuous sub-window [start_s, start_s + window_duration_s)."""
        end_s = start_s + window_duration_s
        sliced = CanonicalTimeline(
            subject_id=self.subject_id,
            record_id=self.record_id,
            dataset_id=self.dataset_id,
            total_duration_s=window_duration_s,
            intervals=[]
        )
        for iv in self.intervals:
            if iv.end_s <= start_s or iv.start_s >= end_s:
                continue
            # Clip interval to window boundaries
            cl_start = max(start_s, iv.start_s) - start_s
            cl_end = min(end_s, iv.end_s) - start_s
            if cl_end > cl_start:
                sliced.intervals.append(
                    CanonicalInterval(
                        subject_id=iv.subject_id,
                        record_id=iv.record_id,
                        dataset_id=iv.dataset_id,
                        start_s=cl_start,
                        end_s=cl_end,
                        state=iv.state,
                        state_confidence=iv.state_confidence,
                        quality_class=iv.quality_class,
                        observation_status=iv.observation_status,
                        annotation_source=iv.annotation_source
                    )
                )
        sliced._validate_and_sort()
        return sliced
