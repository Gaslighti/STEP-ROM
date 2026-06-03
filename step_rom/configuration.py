"""Runtime configuration objects for the unified STEP-ROM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import pipeline_defaults


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

    def __post_init__(self) -> None:
        self.project_archive = Path(self.project_archive).expanduser().resolve()

    def build_stage_configs(self) -> dict[str, dict[str, Any]]:
        configs = pipeline_defaults()
        configs["export"]["input_archive_name"] = str(self.project_archive)
        configs["export"]["parameters"] = (
            [parameter.as_stage_spec() for parameter in self.parameters]
            if self.is_parameterized
            else []
        )
        return configs
