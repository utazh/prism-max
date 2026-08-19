"""Dependency-free layer retention budget profiles.

Profile means are checked with an absolute tolerance of 1e-9.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_MEAN_TOLERANCE = 1e-9

_JSON_FIELDS = frozenset(
    {
        "schema_version",
        "model",
        "target_mean_ratio",
        "layer_ratios",
        "calibration",
    }
)


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _ratio(value: object, field_name: str) -> float:
    normalized = _finite_float(value, field_name)
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"{field_name} must be in (0, 1]")
    return normalized


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class LayerBudgetProfile:
    """Validated per-layer retention ratios and their calibration metadata."""

    schema_version: int
    model: str
    target_mean_ratio: float
    layer_ratios: tuple[float, ...]
    calibration: Mapping[str, Any]
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.layer_ratios, tuple) or not self.layer_ratios:
            raise ValueError("layer_ratios must be a non-empty tuple")
        if not isinstance(self.calibration, Mapping):
            raise ValueError("calibration must be a mapping")
        if self.source_path is not None and not isinstance(self.source_path, str):
            raise ValueError("source_path must be a string or None")
        if self.source_sha256 is not None and not isinstance(self.source_sha256, str):
            raise ValueError("source_sha256 must be a string or None")

        target = _ratio(self.target_mean_ratio, "target_mean_ratio")
        ratios = tuple(
            _ratio(value, f"layer_ratios[{index}]")
            for index, value in enumerate(self.layer_ratios)
        )
        actual_mean = math.fsum(ratios) / len(ratios)
        if not math.isclose(
            actual_mean,
            target,
            rel_tol=0.0,
            abs_tol=DEFAULT_MEAN_TOLERANCE,
        ):
            raise ValueError(
                "layer_ratios mean "
                f"{actual_mean!r} does not match target_mean_ratio {target!r} "
                f"within {DEFAULT_MEAN_TOLERANCE}"
            )

        object.__setattr__(self, "target_mean_ratio", target)
        object.__setattr__(self, "layer_ratios", ratios)
        object.__setattr__(
            self,
            "calibration",
            _freeze_json(copy.deepcopy(dict(self.calibration))),
        )


def load_layer_budget_profile(
    path: str | Path,
    expected_layers: int,
    expected_model: str | None = None,
) -> LayerBudgetProfile:
    """Load and validate a profile while recording its exact byte hash."""

    if (
        isinstance(expected_layers, bool)
        or not isinstance(expected_layers, int)
        or expected_layers <= 0
    ):
        raise ValueError("expected_layers must be a positive integer")
    if expected_model is not None and (
        not isinstance(expected_model, str) or not expected_model.strip()
    ):
        raise ValueError("expected_model must be a non-empty string or None")

    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid layer budget JSON in {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("layer budget profile must be a JSON object")

    fields = set(payload)
    missing = sorted(_JSON_FIELDS - fields)
    extra = sorted(fields - _JSON_FIELDS)
    if missing:
        raise ValueError(f"layer budget profile missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"layer budget profile has extra fields: {', '.join(extra)}")

    raw_ratios = payload["layer_ratios"]
    if not isinstance(raw_ratios, list):
        raise ValueError("layer_ratios must be a JSON array")
    if len(raw_ratios) != expected_layers:
        raise ValueError(
            f"layer_ratios has {len(raw_ratios)} layers; expected {expected_layers}"
        )

    profile = LayerBudgetProfile(
        schema_version=payload["schema_version"],
        model=payload["model"],
        target_mean_ratio=payload["target_mean_ratio"],
        layer_ratios=tuple(raw_ratios),
        calibration=payload["calibration"],
        source_path=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if expected_model is not None and profile.model != expected_model:
        raise ValueError(
            f"profile model {profile.model!r} does not match expected_model "
            f"{expected_model!r}"
        )
    return profile


def build_ranked_three_level_profile(
    sensitivity_scores: Sequence[float],
    target_mean_ratio: float,
    delta: float,
    model: str,
    calibration: dict[str, Any],
    extreme_fraction: float = 0.25,
) -> LayerBudgetProfile:
    """Build a balanced high/target/low profile from ranked sensitivities."""

    if isinstance(sensitivity_scores, (str, bytes)) or not isinstance(
        sensitivity_scores, Sequence
    ):
        raise ValueError("sensitivity_scores must be a non-empty sequence")
    scores = tuple(
        _finite_float(score, f"sensitivity_scores[{index}]")
        for index, score in enumerate(sensitivity_scores)
    )
    if not scores:
        raise ValueError("sensitivity_scores must be a non-empty sequence")

    target = _ratio(target_mean_ratio, "target_mean_ratio")
    offset = _finite_float(delta, "delta")
    if offset <= 0.0:
        raise ValueError("delta must be positive")
    low = target - offset
    high = target + offset
    if low <= 0.0 or high > 1.0:
        raise ValueError("target_mean_ratio +/- delta must remain in (0, 1]")
    fraction = _finite_float(extreme_fraction, "extreme_fraction")
    if not 0.0 < fraction <= 0.5:
        raise ValueError("extreme_fraction must be in (0, 0.5]")

    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    group_size = min(
        len(scores) // 2,
        max(1, math.floor(len(scores) * fraction)),
    )
    high_layers = ranked[:group_size]
    low_layers = ranked[-group_size:] if group_size else ()

    ratios = [target] * len(scores)
    for index in high_layers:
        ratios[index] = high
    for index in low_layers:
        ratios[index] = low

    return LayerBudgetProfile(
        schema_version=SCHEMA_VERSION,
        model=model,
        target_mean_ratio=target,
        layer_ratios=tuple(ratios),
        calibration=calibration,
    )


def profile_to_json_dict(profile: LayerBudgetProfile) -> dict[str, Any]:
    """Return the stable on-disk schema without source provenance fields."""

    if not isinstance(profile, LayerBudgetProfile):
        raise TypeError("profile must be a LayerBudgetProfile")
    return {
        "schema_version": profile.schema_version,
        "model": profile.model,
        "target_mean_ratio": profile.target_mean_ratio,
        "layer_ratios": list(profile.layer_ratios),
        "calibration": _thaw_json(profile.calibration),
    }
