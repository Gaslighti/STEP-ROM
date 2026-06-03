from __future__ import annotations

import traceback
from pathlib import Path

from step_rom.logging_utils import configure_pipeline_logger
from step_rom.pipeline import StepRomPipeline
from step_rom.ui import collect_pipeline_input


def main() -> int:
    code_root = Path(__file__).resolve().parent
    logger = None

    try:
        pipeline_input = collect_pipeline_input()
        work_dir = pipeline_input.project_archive.parent.resolve()
        log_path = work_dir / "step_rom_pipeline.log"
        logger = configure_pipeline_logger(log_path)
        logger.info("Detailed log file: %s", log_path)
        StepRomPipeline(code_root=code_root, work_dir=work_dir, logger=logger).run(pipeline_input)
    except Exception:
        if logger is not None:
            logger.exception("STEP-ROM pipeline stopped with an error")
        else:
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
