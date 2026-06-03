"""Runtime configuration objects for the unified STEP-ROM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import pipeline_defaults


def _merge_runtime_overrides(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Recursively merge user runtime overrides into stage defaults."""

    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            _merge_runtime_overrides(target[key], value)
        else:
            target[key] = value


@dataclass(slots=True)
class UserParameter:
    """Workbench parameter value supplied by the user."""

    name: str
    value: float
    unit: str = "mm"

    def as_stage_spec(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(slots=True)
class PipelineInput:
    """Minimal user input needed to run the four-stage STEP-ROM pipeline."""

    project_archive: Path
    is_parameterized: bool
    parameters: list[UserParameter] = field(default_factory=list)
    runtime_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.project_archive = Path(self.project_archive).expanduser().resolve()
        if self.runtime_config is None:
            self.runtime_config = {}

    def build_stage_configs(self) -> dict[str, dict[str, Any]]:
        configs = pipeline_defaults()
        configs["export"]["input_archive_name"] = str(self.project_archive)
        configs["export"]["parameters"] = (
            [parameter.as_stage_spec() for parameter in self.parameters]
            if self.is_parameterized
            else []
        )
        for stage_name, overrides in self.runtime_config.items():
            if stage_name not in configs:
                raise KeyError(f"Unknown STEP-ROM stage config: {stage_name}")
            if not isinstance(overrides, dict):
                raise TypeError(f"Runtime overrides for {stage_name} must be a dict")
            _merge_runtime_overrides(configs[stage_name], overrides)
        return configs
