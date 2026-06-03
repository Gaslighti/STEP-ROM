"""Load legacy stage modules whose filenames are not valid Python identifiers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class StageModuleLoader:
    def __init__(self, code_root: Path) -> None:
        self.code_root = Path(code_root).resolve()

    def load(self, module_name: str, relative_path: str) -> ModuleType:
        module_path = self.code_root / relative_path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Не удалось загрузить модуль этапа: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
