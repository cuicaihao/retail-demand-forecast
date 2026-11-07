"""Versioned benchmark specifications and week-aligned fold generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .common import ROOT

PROTOCOLS_DIR = ROOT / "data" / "benchmarks" / "protocols"


@dataclass(frozen=True)
class RollingFold:
    """One immutable forecast-origin evaluation fold."""

    fold_id: int
    period_name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_weeks: int | None
    eval_weeks: int
    n_test_days: int

    def manifest_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("train_start", "train_end", "test_start", "test_end"):
            record[key] = str(record[key].date())
        return record


@dataclass(frozen=True)
class BenchmarkSpec:
    """The complete, hashable contract for one comparable benchmark."""

    benchmark_id: str
    protocol_version: str
    label: str
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    horizon_days: int
    step_days: int
    training_policy: str
    train_weeks: int | None
    selection_end: pd.Timestamp
    selection_policy: str
    scenarios: tuple[str, ...]
    valid_label_rule: str
    metrics_version: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchmarkSpec:
        values = dict(raw)
        for key in ("evaluation_start", "evaluation_end", "selection_end"):
            values[key] = pd.Timestamp(values[key])
        values["scenarios"] = tuple(values["scenarios"])
        spec = cls(**values)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.protocol_version != "retail-benchmark-v4":
            raise ValueError(f"Unsupported protocol: {self.protocol_version}")
        if self.training_policy not in {"expanding", "sliding"}:
            raise ValueError(f"Unsupported training policy: {self.training_policy}")
        if self.training_policy == "sliding" and not self.train_weeks:
            raise ValueError("Sliding protocols require train_weeks")
        if self.training_policy == "expanding" and self.train_weeks is not None:
            raise ValueError("Expanding protocols must not set train_weeks")
        if self.horizon_days % 7 or self.step_days % 7:
            raise ValueError("Retail-week horizons and steps must use complete weeks")
        if self.evaluation_start.day_name() != "Monday":
            raise ValueError("evaluation_start must be a Monday")
        if self.evaluation_end.day_name() != "Sunday":
            raise ValueError("evaluation_end must be a Sunday")
        if self.selection_end >= self.evaluation_start:
            raise ValueError("Parameter selection must end before formal evaluation")
        days = (self.evaluation_end - self.evaluation_start).days + 1
        if days < self.horizon_days or (days - self.horizon_days) % self.step_days:
            raise ValueError("Evaluation range does not contain complete folds")

    def canonical_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("evaluation_start", "evaluation_end", "selection_end"):
            record[key] = str(record[key].date())
        record["scenarios"] = list(record["scenarios"])
        return record

    @property
    def protocol_hash(self) -> str:
        payload = json.dumps(
            self.canonical_record(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def folds(self, data_start: pd.Timestamp) -> list[RollingFold]:
        folds: list[RollingFold] = []
        test_start = self.evaluation_start
        fold_id = 1
        while (
            test_start + pd.Timedelta(days=self.horizon_days - 1) <= self.evaluation_end
        ):
            test_end = test_start + pd.Timedelta(days=self.horizon_days - 1)
            train_end = test_start - pd.Timedelta(days=1)
            if self.training_policy == "expanding":
                train_start = pd.Timestamp(data_start)
            else:
                train_start = (
                    train_end
                    - pd.Timedelta(weeks=self.train_weeks)
                    + pd.Timedelta(days=1)
                )
            folds.append(
                RollingFold(
                    fold_id=fold_id,
                    period_name=f"P{fold_id:02d}",
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_weeks=self.train_weeks,
                    eval_weeks=self.horizon_days // 7,
                    n_test_days=self.horizon_days,
                )
            )
            test_start += pd.Timedelta(days=self.step_days)
            fold_id += 1
        return folds


def load_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    path = PROTOCOLS_DIR / f"{benchmark_id}.json"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in PROTOCOLS_DIR.glob("*.json")))
        raise FileNotFoundError(
            f"Unknown benchmark ID {benchmark_id!r}. Available: {available}"
        )
    return BenchmarkSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_benchmark_specs() -> list[BenchmarkSpec]:
    return [
        load_benchmark_spec(path.stem) for path in sorted(PROTOCOLS_DIR.glob("*.json"))
    ]
