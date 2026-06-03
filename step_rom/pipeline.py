"""Unified four-stage STEP-ROM orchestrator."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .configuration import PipelineInput
from .legacy_loader import StageModuleLoader
from .logging_utils import LoggerWriter, log_console


@dataclass(frozen=True, slots=True)
class StageSpec:
    title: str
    module_name: str
    module_path: str
    function_name: str
    config_key: str


class StepRomPipeline:
    """Run all STEP-ROM stages sequentially with a single runtime config."""

    STAGES = (
        StageSpec(
            title="1/4 Export files from Ansys Workbench",
            module_name="step_rom_stage_export",
            module_path="1_Exports_files_LS_DYNA.py",
            function_name="run_export_stage",
            config_key="export",
        ),
        StageSpec(
            title="2/4 Generate LS-DYNA STEP k-files",
            module_name="step_rom_stage_generate_k",
            module_path="2_Generate_k_files.py",
            function_name="run_generate_k_files_stage",
            config_key="generate_k",
        ),
        StageSpec(
            title="3/4 Run LS-DYNA jobs and export results",
            module_name="step_rom_stage_lsdyna",
            module_path="3_Ls_dyna_run_configs.py",
            function_name="run_lsdyna_stage",
            config_key="lsdyna",
        ),
        StageSpec(
            title="4/4 Generate ROM and FE comparison",
            module_name="step_rom_stage_rom",
            module_path="4_Generate_ROM.py",
            function_name="run_rom_stage",
            config_key="rom",
        ),
    )

    def __init__(self, code_root: Path, work_dir: Path, logger: logging.Logger) -> None:
        self.code_root = Path(code_root).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.logger = logger
        self.loader = StageModuleLoader(self.code_root)

    def run(self, pipeline_input: PipelineInput) -> None:
        log_console(self.logger, logging.INFO, "STEP-ROM pipeline started")
        self.logger.info("Project archive: %s", pipeline_input.project_archive)
        self.logger.info("Parameterized model: %s", pipeline_input.is_parameterized)
        if pipeline_input.parameters:
            for parameter in pipeline_input.parameters:
                self.logger.info(
                    "Parameter %s = %s [%s]",
                    parameter.name,
                    parameter.value,
                    parameter.unit,
                )

        configs = pipeline_input.build_stage_configs()
        for stage in self.STAGES:
            self._run_stage(stage, configs[stage.config_key])

        log_console(
            self.logger, logging.INFO, "STEP-ROM pipeline finished successfully"
        )

    def _run_stage(self, stage: StageSpec, config: dict) -> None:
        log_console(self.logger, logging.INFO, "START %s", stage.title)
        module = self.loader.load(stage.module_name, stage.module_path)
        stage_func: Callable[..., object] = getattr(module, stage.function_name)

        stdout_writer = LoggerWriter(self.logger, logging.INFO)
        stderr_writer = LoggerWriter(self.logger, logging.ERROR)
        try:
            with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(
                stderr_writer
            ):
                result = stage_func(config, base_dir=self.work_dir)
            stdout_writer.flush()
            stderr_writer.flush()
        except Exception:
            stdout_writer.flush()
            stderr_writer.flush()
            self.logger.exception("Stage failed: %s", stage.title)
            raise

        if isinstance(result, int) and result != 0:
            log_console(
                self.logger, logging.ERROR, "FAILED %s (code %s)", stage.title, result
            )
            raise RuntimeError(f"Этап завершился с кодом {result}: {stage.title}")
        log_console(self.logger, logging.INFO, "DONE %s", stage.title)
