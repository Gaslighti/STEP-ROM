from __future__ import annotations

from pathlib import Path

from step_rom.logging_utils import configure_pipeline_logger
from step_rom.pipeline import StepRomPipeline
from step_rom.ui import collect_pipeline_input


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    log_path = repo_root / "step_rom_pipeline.log"
    logger = configure_pipeline_logger(log_path)
    logger.info("Detailed log file: %s", log_path)

    try:
        pipeline_input = collect_pipeline_input()
        StepRomPipeline(repo_root=repo_root, logger=logger).run(pipeline_input)
    except Exception:
        logger.exception("STEP-ROM pipeline stopped with an error")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
