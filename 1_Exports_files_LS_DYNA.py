from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_NAME = "1_Exports_files_LS_DYNA.json"
GENERATED_JOURNAL_NAME = "generated_step_pipeline.wbjn"
LOG_FILE_NAME = "runwb2_python.log"
SUCCESS_MARKER_NAME = "workbench_success.txt"
ERROR_MARKER_NAME = "workbench_error.txt"
MANIFEST_NAME = "export_manifest.json"
MODAL_COLLECT_LOG_NAME = "modal_collect_log.txt"


def find_runwb2_2024r2() -> Path:
    candidates = []

    env_root = os.environ.get("AWP_ROOT242")
    if env_root:
        candidates.append(Path(env_root) / "Framework" / "bin" / "Win64" / "RunWB2.exe")

    candidates.extend([
        Path(r"F:\Program Files\ANSYS Inc\v242\Framework\bin\Win64\RunWB2.exe"),
        Path(r"C:\Program Files\ANSYS Inc\v242\Framework\bin\Win64\RunWB2.exe"),
        Path(r"D:\Program Files\ANSYS Inc\v242\Framework\bin\Win64\RunWB2.exe"),
        Path(r"C:\Program Files\ANSYS Inc\ANSYS Student\v242\Framework\bin\Win64\RunWB2.exe"),
        Path(r"D:\Program Files\ANSYS Inc\ANSYS Student\v242\Framework\bin\Win64\RunWB2.exe"),
    ])

    for exe in candidates:
        if exe.is_file():
            return exe

    raise FileNotFoundError(
        "Не найден RunWB2.exe для Ansys 2024 R2. "
        "Проверьте переменную AWP_ROOT242 или путь установки."
    )


def wb_path(path_obj: Path) -> str:
    return str(path_obj.resolve()).replace("\\", "/")


def format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_expression(spec: dict[str, Any], default_unit: str) -> str:
    if "expression" in spec:
        expression = str(spec["expression"]).strip()
        if not expression:
            raise ValueError(f"Параметр {spec!r}: пустое поле 'expression'.")
        return expression

    if "value" not in spec:
        raise ValueError(f"Параметр {spec!r}: должно быть указано 'value' или 'expression'.")

    value_text = format_scalar(spec["value"])
    unit = str(spec.get("unit", default_unit)).strip()
    return f"{value_text} [{unit}]" if unit else value_text


def remove_path_if_exists(path_obj: Path) -> None:
    if not path_obj.exists():
        return
    if path_obj.is_file():
        path_obj.unlink()
    elif path_obj.is_dir():
        shutil.rmtree(path_obj)
    else:
        raise RuntimeError(f"Не удалось удалить путь: {path_obj}")


def load_json_config(path_obj: Path) -> dict[str, Any]:
    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON config должен содержать объект: {path_obj}")
    return data


def resolve_archive_path(base_dir: Path, requested: object) -> Path:
    """
    requested может быть:
      - имя файла .wbpz;
      - "auto": взять единственный .wbpz рядом со скриптом;
      - отсутствовать/null: то же что "auto".
    """
    if requested is None or str(requested).strip().lower() == "auto":
        candidates = sorted(base_dir.glob("*.wbpz"), key=lambda p: p.name.lower())
        ignored_suffixes = ("_updated.wbpz",)
        candidates = [p for p in candidates if not p.name.lower().endswith(ignored_suffixes)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError("В папке скрипта не найден ни один архив .wbpz.")
        raise RuntimeError(
            "Найдено несколько .wbpz архивов. Укажите input_archive_name явно: "
            + ", ".join(p.name for p in candidates)
        )

    path = Path(str(requested))
    if not path.is_absolute():
        path = base_dir / path
    return path


def normalize_parameters(parameters: list[dict[str, Any]], default_unit: str) -> list[dict[str, Any]]:
    if parameters is None:
        return []
    if not isinstance(parameters, list):
        raise ValueError("parameters должен быть списком объектов или пустым списком.")
    if not parameters:
        return []

    seen = set()
    result: list[dict[str, Any]] = []

    for idx, spec in enumerate(parameters, start=1):
        if "name" not in spec:
            raise ValueError(f"Параметр №{idx}: отсутствует поле 'name'.")

        name = str(spec["name"]).strip()
        if not name:
            raise ValueError(f"Параметр №{idx}: поле 'name' пустое.")
        if name in seen:
            raise ValueError(f"Параметр '{name}' указан более одного раза.")
        seen.add(name)

        expression = build_expression(spec, default_unit)
        result.append({"name": name, "expression": expression})

    return result


def build_cases_from_compact_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    bc_count = int(cfg["bc_count"])
    if bc_count < 1:
        raise ValueError("bc_count должен быть >= 1.")

    modal_start_sys = int(cfg.get("modal_start_sys", 1))
    step_start_sys = int(cfg.get("step_lsdyna_start_sys", cfg.get("lsdyna_start_sys", 2)))
    real_start_sys = int(cfg.get("real_lsdyna_start_sys", 3))
    sys_step = int(cfg.get("sys_step", 3))
    case_prefix = str(cfg.get("case_prefix", "bc")).strip() or "bc"

    modal_file_patterns = cfg.get("modal_file_patterns", ["mode*_step.txt", "modal_reference_info.txt"])
    if not isinstance(modal_file_patterns, list) or not modal_file_patterns:
        raise ValueError("modal_file_patterns должен быть непустым списком.")

    cases: list[dict[str, Any]] = []
    for i in range(bc_count):
        case_index = i + 1
        label = f"{case_prefix}_{case_index:02d}"
        modal_sys_num = modal_start_sys + i * sys_step
        step_sys_num = step_start_sys + i * sys_step
        real_sys_num = real_start_sys + i * sys_step

        export_root = Path("exports") / label

        cases.append({
            "label": label,

            "modal_system_name_wb": f"SYS {modal_sys_num}",
            "modal_sys_folder": f"SYS-{modal_sys_num}",
            "modal_output_dir": export_root / "modal",
            "modal_file_patterns": modal_file_patterns,

            "step_system_name_wb": f"SYS {step_sys_num}",
            "step_sys_folder": f"SYS-{step_sys_num}",
            "step_key_output_path": export_root / "model.key",

            "real_system_name_wb": f"SYS {real_sys_num}",
            "real_sys_folder": f"SYS-{real_sys_num}",
            "real_key_output_path": export_root / "real_load.key",
        })

    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    labels = set()
    outputs = set()

    for case in cases:
        label = case["label"]
        if label in labels:
            raise ValueError(f"Дублирующийся label case: {label}")
        labels.add(label)

        for key_name in ("step_key_output_path", "real_key_output_path"):
            out_path = str(case[key_name])
            if out_path in outputs:
                raise ValueError(f"Дублирующийся путь экспорта: {out_path}")
            outputs.add(out_path)


def build_parameter_block(parameters: list[dict[str, Any]], design_point_name: str) -> str:
    if not parameters:
        return ""
    lines = [f'designPoint1 = Parameters.GetDesignPoint(Name="{design_point_name}")']
    for index, spec in enumerate(parameters, start=1):
        param_var = f"parameter{index}"
        lines.append(f'{param_var} = Parameters.GetParameter(Name="{spec["name"]}")')
        lines.append("designPoint1.SetParameterExpression(")
        lines.append(f"    Parameter={param_var},")
        lines.append(f'    Expression="{spec["expression"]}")')
    return "\n".join(lines)


def build_base_system_update_block(base_system_name: str) -> str:
    lines = [
        f'base_system = GetSystem(Name="{base_system_name}")',
        'base_geometry = base_system.GetComponent(Name="Geometry")',
        'base_geometry.Update(AllDependencies=True)',
        'base_model = base_system.GetComponent(Name="Model")',
        'base_model.Update(AllDependencies=True)',
        '',
    ]
    return "\n".join(lines)


def build_modal_update_block(case_idx: int, modal_system_name: str) -> str:
    sys_var = f"modal_system_{case_idx}"
    model_var = f"modal_model_{case_idx}"
    setup_var = f"modal_setup_{case_idx}"
    solution_var = f"modal_solution_{case_idx}"
    results_var = f"modal_results_{case_idx}"

    lines = [
        f'{sys_var} = GetSystem(Name="{modal_system_name}")',
        f'{model_var} = {sys_var}.GetComponent(Name="Model")',
        f'{model_var}.Update(AllDependencies=True)',
        f'{setup_var} = {sys_var}.GetComponent(Name="Setup")',
        f'{setup_var}.Update(AllDependencies=True)',
        f'{solution_var} = {sys_var}.GetComponent(Name="Solution")',
        f'{solution_var}.Update(AllDependencies=True)',
        f'{results_var} = {sys_var}.GetComponent(Name="Results")',
        f'{results_var}.Update(AllDependencies=True)',
        '',
    ]
    return "\n".join(lines)


def build_lsdyna_update_and_export_block(case_idx: int, system_name: str, key_path: Path, prefix: str) -> str:
    sys_var = f"{prefix}_system_{case_idx}"
    model_var = f"{prefix}_model_{case_idx}"
    setup_comp_var = f"{prefix}_setup_component_{case_idx}"
    setup_container_var = f"{prefix}_setup_container_{case_idx}"

    lines = [
        f'{sys_var} = GetSystem(Name="{system_name}")',
        f'{model_var} = {sys_var}.GetComponent(Name="Model")',
        f'{model_var}.Update(AllDependencies=True)',
        f'{setup_comp_var} = {sys_var}.GetComponent(Name="Setup")',
        f'{setup_comp_var}.Update(AllDependencies=True)',
        f'{setup_container_var} = {sys_var}.GetContainer(ComponentName="Setup")',
        f'{setup_container_var}.Export(',
        f'    Path=r"{wb_path(key_path)}",',
        '    SetupDataType="InputFile")',
        '',
    ]
    return "\n".join(lines)


def build_generated_journal(
    archive_input_file: Path,
    project_file: Path,
    updated_archive_file: Path,
    success_marker_file: Path,
    parameters: list[dict[str, Any]],
    design_point_name: str,
    base_system_name: str,
    cases: list[dict[str, Any]],
    write_updated_archive: bool,
) -> str:
    lines: list[str] = []

    lines.append(f'Unarchive(ArchivePath=r"{wb_path(archive_input_file)}", ProjectPath=r"{wb_path(project_file)}", Overwrite=True)')
    lines.append("")
    lines.append(build_parameter_block(parameters, design_point_name))
    lines.append("")
    lines.append(build_base_system_update_block(base_system_name))

    for idx, case in enumerate(cases, start=1):
        lines.append(build_modal_update_block(idx, case["modal_system_name_wb"]))
        lines.append(
            build_lsdyna_update_and_export_block(
                idx,
                case["step_system_name_wb"],
                project_file.parent / case["step_key_output_path"],
                prefix="step",
            )
        )
        lines.append(
            build_lsdyna_update_and_export_block(
                idx,
                case["real_system_name_wb"],
                project_file.parent / case["real_key_output_path"],
                prefix="real",
            )
        )

    lines.append(f'Save(FilePath=r"{wb_path(project_file)}", Overwrite=True)')
    if write_updated_archive:
        lines.append(f'Archive(FilePath=r"{wb_path(updated_archive_file)}")')

    lines.append("import os")
    lines.append(f'with open(r"{wb_path(success_marker_file)}", "w") as _f:')
    lines.append('    _f.write("OK")')

    return "\n".join(lines) + "\n"


def launch_workbench_with_journal_batch(runwb2: Path, journal: Path, workdir: Path, log_file: Path) -> int:
    cmd = [str(runwb2), "-B", "-R", str(journal)]

    with log_file.open("w", encoding="utf-8", buffering=1) as log:
        log.write("[INFO] START WORKBENCH BATCH\n")
        log.write(f"[INFO] CMD: {' '.join(cmd)}\n")
        log.flush()

        result = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        log.write(f"[INFO] RETURN CODE: {result.returncode}\n")
        log.flush()

    return result.returncode


def copy_modal_outputs(
    base_dir: Path,
    project_file: Path,
    cases: list[dict[str, Any]],
    modal_collect_log_file: Path,
) -> dict[str, Any]:
    project_files_dir = project_file.with_suffix("").parent / f"{project_file.stem}_files"
    dp0_dir = project_files_dir / "dp0"

    manifest_modal: dict[str, Any] = {}
    log_lines: list[str] = []

    if not dp0_dir.exists():
        log_lines.append(f"[WARNING] Не найдена папка dp0: {dp0_dir}")
        modal_collect_log_file.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return manifest_modal

    for case in cases:
        label = case["label"]
        modal_sys_folder = case["modal_sys_folder"]
        src_dir = dp0_dir / modal_sys_folder
        dst_dir = base_dir / case["modal_output_dir"]
        dst_dir.mkdir(parents=True, exist_ok=True)

        case_entries: list[dict[str, str]] = []
        log_lines.append(f"[CASE] {label}")
        log_lines.append(f"       source={src_dir}")
        log_lines.append(f"       target={dst_dir}")

        if not src_dir.exists():
            log_lines.append("       [WARNING] source folder not found")
            manifest_modal[label] = case_entries
            continue

        matched_files: list[Path] = []
        for pattern in case["modal_file_patterns"]:
            matched_files.extend(src_dir.rglob(pattern))

        unique_files = sorted({p.resolve() for p in matched_files if p.is_file()})

        for src_file in unique_files:
            new_name = f"{label}__{modal_sys_folder}__{src_file.name}"
            dst_file = dst_dir / new_name
            shutil.copy2(src_file, dst_file)
            case_entries.append({
                "source": str(src_file),
                "target": str(dst_file),
            })
            log_lines.append(f"       copied: {src_file.name} -> {dst_file.name}")

        if not case_entries:
            log_lines.append("       [WARNING] no modal files matched")

        manifest_modal[label] = case_entries
        log_lines.append("")

    modal_collect_log_file.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return manifest_modal


def print_summary(parameters: list[dict[str, Any]], base_system_name: str, cases: list[dict[str, Any]]) -> None:
    print("[INFO] Параметры:")
    for spec in parameters:
        print(f'       {spec["name"]} = {spec["expression"]}')

    print(f"[INFO] Базовая система модели: {base_system_name}")
    print("[INFO] Cases:")
    for case in cases:
        print(
            f'       {case["label"]}: '
            f'modal={case["modal_system_name_wb"]} ({case["modal_sys_folder"]}), '
            f'step={case["step_system_name_wb"]} -> {case["step_key_output_path"]}, '
            f'real={case["real_system_name_wb"]} -> {case["real_key_output_path"]}'
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Workbench: Модель + модальный анализ + статика LS-DYNA."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_file = base_dir / args.config
    if not config_file.is_file():
        print(f"[ERROR] Не найден config: {config_file}")
        return 1

    try:
        cfg = load_json_config(config_file)

        design_point_name = str(cfg.get("design_point_name", "0"))
        default_unit = str(cfg.get("default_unit", "mm"))
        base_system_name = str(cfg.get("base_system_name", "SYS"))

        input_archive_file = resolve_archive_path(base_dir, cfg.get("input_archive_name", "auto"))
        if not input_archive_file.is_file():
            raise FileNotFoundError(f"Не найден входной архив: {input_archive_file}")

        project_requested = str(cfg.get("project_file_name", "auto")).strip()
        if project_requested.lower() == "auto":
            project_file = base_dir / f"{input_archive_file.stem}_mod.wbpj"
        else:
            project_file = base_dir / project_requested

        write_updated_archive = bool(cfg.get("write_updated_archive", True))
        updated_requested = str(cfg.get("updated_archive_name", "auto")).strip()
        if updated_requested.lower() == "auto":
            updated_archive_file = base_dir / f"{input_archive_file.stem}_updated.wbpz"
        else:
            updated_archive_file = base_dir / updated_requested

        parameters = normalize_parameters(cfg.get("parameters", []), default_unit)
        cases = build_cases_from_compact_config(cfg)
        validate_cases(cases)

        generated_journal = base_dir / GENERATED_JOURNAL_NAME
        log_file = base_dir / LOG_FILE_NAME
        success_marker_file = base_dir / SUCCESS_MARKER_NAME
        error_marker_file = base_dir / ERROR_MARKER_NAME
        manifest_file = base_dir / MANIFEST_NAME
        modal_collect_log_file = base_dir / MODAL_COLLECT_LOG_NAME

        cleanup_paths = [
            generated_journal,
            log_file,
            success_marker_file,
            error_marker_file,
            manifest_file,
            modal_collect_log_file,
            project_file,
        ]
        if write_updated_archive:
            cleanup_paths.append(updated_archive_file)

        for case in cases:
            cleanup_paths.append(base_dir / case["step_key_output_path"])
            cleanup_paths.append(base_dir / case["real_key_output_path"])
            cleanup_paths.append(base_dir / case["modal_output_dir"])

        for p in cleanup_paths:
            remove_path_if_exists(p)

        project_file.parent.mkdir(parents=True, exist_ok=True)
        if write_updated_archive:
            updated_archive_file.parent.mkdir(parents=True, exist_ok=True)

        for case in cases:
            (base_dir / case["step_key_output_path"]).parent.mkdir(parents=True, exist_ok=True)
            (base_dir / case["real_key_output_path"]).parent.mkdir(parents=True, exist_ok=True)
            (base_dir / case["modal_output_dir"]).mkdir(parents=True, exist_ok=True)

        journal_text = build_generated_journal(
            archive_input_file=input_archive_file,
            project_file=project_file,
            updated_archive_file=updated_archive_file,
            success_marker_file=success_marker_file,
            parameters=parameters,
            design_point_name=design_point_name,
            base_system_name=base_system_name,
            cases=cases,
            write_updated_archive=write_updated_archive,
        )
        generated_journal.write_text(journal_text, encoding="utf-8-sig")

        runwb2 = find_runwb2_2024r2()

        print(f"[INFO] Config: {config_file}")
        print(f"[INFO] Generated journal: {generated_journal}")
        print(f"[INFO] Input archive: {input_archive_file}")
        print(f"[INFO] Project file: {project_file}")
        if write_updated_archive:
            print(f"[INFO] Updated archive: {updated_archive_file}")
        print_summary(parameters, base_system_name, cases)

        return_code = launch_workbench_with_journal_batch(runwb2, generated_journal, base_dir, log_file)
        print(f"[INFO] Код возврата Workbench: {return_code}")

        if not success_marker_file.exists():
            error_marker_file.write_text(
                "Workbench завершился без success marker. Проверьте generated_step_pipeline.wbjn и runwb2_python.log\n",
                encoding="utf-8",
            )

        modal_manifest = copy_modal_outputs(
            base_dir=base_dir,
            project_file=project_file,
            cases=cases,
            modal_collect_log_file=modal_collect_log_file,
        )

        manifest = {
            "config_file": str(config_file),
            "input_archive_file": str(input_archive_file),
            "project_file": str(project_file),
            "updated_archive_file": str(updated_archive_file) if write_updated_archive else None,
            "success_marker_exists": success_marker_file.exists(),
            "workbench_return_code": return_code,
            "cases": {},
            "modal_outputs": modal_manifest,
        }

        for case in cases:
            label = case["label"]
            step_key_file = base_dir / case["step_key_output_path"]
            real_key_file = base_dir / case["real_key_output_path"]
            manifest["cases"][label] = {
                "modal_system_name_wb": case["modal_system_name_wb"],
                "modal_sys_folder": case["modal_sys_folder"],
                "step_system_name_wb": case["step_system_name_wb"],
                "step_sys_folder": case["step_sys_folder"],
                "step_key_file": str(step_key_file),
                "step_key_exists": step_key_file.exists(),
                "real_system_name_wb": case["real_system_name_wb"],
                "real_sys_folder": case["real_sys_folder"],
                "real_key_file": str(real_key_file),
                "real_key_exists": real_key_file.exists(),
                "modal_output_dir": str(base_dir / case["modal_output_dir"]),
            }

        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[INFO] Manifest: {manifest_file}")
        print(f"[INFO] Modal collect log: {modal_collect_log_file}")

        for case in cases:
            step_key_file = base_dir / case["step_key_output_path"]
            real_key_file = base_dir / case["real_key_output_path"]

            if step_key_file.exists():
                print(f"[INFO] STEP KEY : {step_key_file}")
            else:
                print(f"[WARNING] STEP KEY не найден: {step_key_file}")

            if real_key_file.exists():
                print(f"[INFO] REAL KEY : {real_key_file}")
            else:
                print(f"[WARNING] REAL KEY не найден: {real_key_file}")

        return return_code

    except Exception as exc:
        msg = f"[ERROR] {exc}"
        print(msg)
        try:
            (base_dir / ERROR_MARKER_NAME).write_text(msg + "\n", encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
