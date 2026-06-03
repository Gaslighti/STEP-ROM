from __future__ import annotations

import json
import math
import re
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np


FIELD_WIDTH = 10
VALID_DOFS = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")
FILENAME_MODE_RE = re.compile(r"mode[^0-9]*([0-9]+)", re.IGNORECASE)


@dataclass
class ModeShape:
    mode_number: int
    source_file: Path
    frequency_hz: Optional[float]
    dominant_dof: str
    normalization_value: float
    node_values: Dict[int, Dict[str, float]]


def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def ensure_list_of_positive_numbers(values: List[float], name: str) -> List[float]:
    clean: List[float] = []
    seen = set()
    for i, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{i}] должно быть числом.")
        v = float(value)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name}[{i}] должно быть конечным числом > 0.")
        key = round(v, 12)
        if key not in seen:
            seen.add(key)
            clean.append(v)
    return clean


def normalize_positive_number_or_auto(value, name: str):
    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return ensure_list_of_positive_numbers(value, name)


def normalize_dofs_or_auto(value, name: str):
    if value is None:
        return "auto"
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return "auto"
        raise ValueError(f"{name} должен быть списком DOF или строкой 'auto'.")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} должен быть непустым списком DOF или строкой 'auto'.")
    out = [str(x).upper() for x in value]
    if any(d not in VALID_DOFS for d in out):
        raise ValueError(f"{name} может содержать только UX, UY, UZ, ROTX, ROTY, ROTZ.")
    return out


def translational_dofs() -> List[str]:
    return ["UX", "UY", "UZ"]


def rotational_dofs() -> List[str]:
    return ["ROTX", "ROTY", "ROTZ"]


def normalize_config(config: dict) -> dict:
    cfg = dict(config)
    cfg["exports_root"] = str(cfg["exports_root"])
    cfg["case_glob"] = str(cfg["case_glob"])
    cfg["base_key_name"] = str(cfg.get("base_key_name", "model.key"))
    cfg["modal_dir_name"] = str(cfg["modal_dir_name"])
    cfg["step_dir_name"] = str(cfg["step_dir_name"])
    cfg["generate_linear_and_nonlinear_decks"] = bool(cfg.get("generate_linear_and_nonlinear_decks", False))
    cfg["output_suffix"] = str(cfg.get("output_suffix", "_nl.k"))
    cfg["output_suffix_nl"] = str(cfg.get("output_suffix_nl", "_nl.k"))
    cfg["output_suffix_lin"] = str(cfg.get("output_suffix_lin", "_lin.k"))
    cfg["patch_implicit_cards"] = bool(cfg.get("patch_implicit_cards", True))

    selected_modes = cfg.get("selected_modes", [])
    if not isinstance(selected_modes, list) or not selected_modes:
        raise ValueError("selected_modes должен быть непустым списком номеров мод.")
    cfg["selected_modes"] = [int(x) for x in selected_modes]

    cfg["model_family"] = str(cfg.get("model_family", "auto")).strip().lower()
    if cfg["model_family"] not in ("auto", "shell", "solid", "mixed"):
        raise ValueError("model_family должен быть auto, shell, solid или mixed.")

    cfg["prescribed_dofs"] = normalize_dofs_or_auto(cfg.get("prescribed_dofs", "auto"), "prescribed_dofs")
    cfg["primary_step_prescribed_mode"] = str(cfg.get("primary_step_prescribed_mode", "auto")).strip().lower()
    if cfg["primary_step_prescribed_mode"] not in ("auto", "full", "normal_only", "translational", "normal_plus_rotations"):
        raise ValueError("primary_step_prescribed_mode должен быть auto, full, normal_only, translational или normal_plus_rotations.")
    cfg["strict_solid_normal_step_guard"] = bool(cfg.get("strict_solid_normal_step_guard", True))
    cfg["min_normal_content_ratio_for_selected_modes"] = float(cfg.get("min_normal_content_ratio_for_selected_modes", 1.0e-6))
    cfg["allow_low_normal_content_modes"] = bool(cfg.get("allow_low_normal_content_modes", False))
    cfg["projection_dofs"] = normalize_dofs_or_auto(cfg.get("projection_dofs", ["UX", "UY", "UZ"]), "projection_dofs")
    if cfg["projection_dofs"] == "auto":
        cfg["projection_dofs"] = ["UX", "UY", "UZ"]

    force_axis = cfg.get("force_normal_axis", None)
    if force_axis is not None:
        force_axis = str(force_axis).upper()
        if force_axis not in ("X", "Y", "Z"):
            raise ValueError("force_normal_axis должен быть X, Y, Z или null.")
    cfg["force_normal_axis"] = force_axis

    cfg["zero_tol"] = float(cfg.get("zero_tol", 1.0e-12))
    cfg["curve_id"] = int(cfg.get("curve_id", 900001))
    cfg["end_time"] = float(cfg.get("end_time", 1.0))
    cfg["single_q_values"] = normalize_positive_number_or_auto(cfg.get("single_q_values", "auto"), "single_q_values")
    cfg["pair_q_values"] = normalize_positive_number_or_auto(cfg.get("pair_q_values", "auto"), "pair_q_values")
    cfg["triple_q_values"] = normalize_positive_number_or_auto(cfg.get("triple_q_values", "auto"), "triple_q_values")
    cfg["generate_dual_mode_decks"] = bool(cfg.get("generate_dual_mode_decks", True))
    raw_dual_p = cfg.get("dual_p_value", "auto")
    if raw_dual_p is None or (isinstance(raw_dual_p, str) and raw_dual_p.strip().lower() == "auto"):
        cfg["dual_p_value"] = "auto"
    else:
        cfg["dual_p_value"] = float(raw_dual_p)
        if not math.isfinite(cfg["dual_p_value"]) or cfg["dual_p_value"] <= 0.0:
            raise ValueError("dual_p_value должен быть конечным числом > 0 или 'auto'.")
    cfg["dual_output_suffix"] = str(cfg.get("dual_output_suffix", ".k"))
    dual_prescribed_mode = str(cfg.get("dual_prescribed_mode", "normal_only")).strip().lower()
    if dual_prescribed_mode not in ("normal_only", "prescribed_dofs"):
        raise ValueError("dual_prescribed_mode должен быть 'normal_only' или 'prescribed_dofs'.")
    cfg["dual_prescribed_mode"] = dual_prescribed_mode

    raw_dual_dofs = cfg.get("dual_prescribed_dofs", None)
    if raw_dual_dofs is None:
        cfg["dual_prescribed_dofs"] = None
    else:
        if not isinstance(raw_dual_dofs, list) or not raw_dual_dofs:
            raise ValueError("dual_prescribed_dofs должен быть null/None или непустым списком DOF.")
        dual_dofs = [str(x).upper() for x in raw_dual_dofs]
        if any(d not in VALID_DOFS for d in dual_dofs):
            raise ValueError("dual_prescribed_dofs может содержать только UX, UY, UZ, ROTX, ROTY, ROTZ.")
        cfg["dual_prescribed_dofs"] = dual_dofs

    raw_set_ids = cfg.get("exclude_node_set_ids", [])
    if raw_set_ids is None:
        raw_set_ids = []
    if not isinstance(raw_set_ids, list):
        raise ValueError("exclude_node_set_ids должен быть списком целых SID.")
    cfg["exclude_node_set_ids"] = [int(x) for x in raw_set_ids]
    cfg["exclude_all_set_node_lists"] = bool(cfg.get("exclude_all_set_node_lists", False))
    cfg["auto_exclude_spc_nodes"] = bool(cfg.get("auto_exclude_spc_nodes", True))
    cfg["auto_exclude_spc_sets"] = bool(cfg.get("auto_exclude_spc_sets", True))
    cfg["auto_exclude_spc_node_cards"] = bool(cfg.get("auto_exclude_spc_node_cards", True))
    cfg["report_excluded_nodes"] = bool(cfg.get("report_excluded_nodes", True))

    cfg["combined_step_q_scaling"] = str(cfg.get("combined_step_q_scaling", "active_count")).strip().lower()
    if cfg["combined_step_q_scaling"] not in ("none", "sqrt_active_count", "active_count", "manual"):
        raise ValueError("combined_step_q_scaling должен быть none, sqrt_active_count, active_count или manual.")
    cfg["include_single_q_values_for_all_interaction_q"] = bool(cfg.get("include_single_q_values_for_all_interaction_q", True))
    cfg["pair_q_factor"] = cfg.get("pair_q_factor", None)
    cfg["triple_q_factor"] = cfg.get("triple_q_factor", None)
    cfg["dual_q_factor"] = float(cfg.get("dual_q_factor", 1.0))
    if not math.isfinite(cfg["dual_q_factor"]) or cfg["dual_q_factor"] <= 0.0:
        raise ValueError("dual_q_factor должен быть конечным числом > 0.")

    # 3D prescribed-motion variant: q can be scaled from the smallest solid
    # element edge rather than from thickness. This keeps the imposed nodal
    # motion much smaller than both the model and element size.
    cfg["solid_q_scale_basis"] = str(cfg.get("solid_q_scale_basis", "min_edge")).strip().lower()
    if cfg["solid_q_scale_basis"] not in ("min_edge", "thickness", "max_span"):
        raise ValueError("solid_q_scale_basis должен быть min_edge, thickness или max_span.")
    cfg["use_full_q_sweep_for_prescribed_step"] = bool(cfg.get("use_full_q_sweep_for_prescribed_step", True))


    cfg["step_load_scheme"] = str(cfg.get("step_load_scheme", "prescribed_motion_step")).strip().lower()
    valid_load_schemes = (
        "prescribed_motion_step",
        "surface_nodal_force_step",
    )
    if cfg["step_load_scheme"] not in valid_load_schemes:
        raise ValueError(
            "step_load_scheme должен быть prescribed_motion_step или surface_nodal_force_step."
        )

    cfg["step_control_basis"] = str(cfg.get("step_control_basis", "q")).strip().lower()
    if cfg["step_control_basis"] not in ("q", "lambda"):
        raise ValueError("step_control_basis должен быть 'q' или 'lambda'.")

    cfg["surface_force_direction"] = str(cfg.get("surface_force_direction", "modal_vector")).strip().lower()
    valid_force_dirs = (
        "modal_vector",
        "modal_normal",
        "surface_normal",
        "dominant_dof",
    )
    if cfg["surface_force_direction"] not in valid_force_dirs:
        raise ValueError(
            "surface_force_direction должен быть modal_vector, modal_normal, surface_normal или dominant_dof."
        )

    cfg["surface_force_node_scope"] = str(cfg.get("surface_force_node_scope", "surface_corner_nodes")).strip().lower()
    valid_force_scopes = (
        "surface_corner_nodes",
        "surface_nodes",
        "corner_nodes",
    )
    if cfg["surface_force_node_scope"] not in valid_force_scopes:
        raise ValueError(
            "surface_force_node_scope должен быть surface_corner_nodes, surface_nodes или corner_nodes."
        )

    cfg["surface_force_use_midside_nodes"] = bool(cfg.get("surface_force_use_midside_nodes", False))
    cfg["surface_force_area_weighted"] = bool(cfg.get("surface_force_area_weighted", True))
    cfg["surface_force_normalization"] = str(cfg.get("surface_force_normalization", "modal_work_unit")).strip().lower()
    if cfg["surface_force_normalization"] not in ("modal_work_unit", "l1_total_force", "none"):
        raise ValueError(
            "surface_force_normalization должен быть modal_work_unit, l1_total_force или none."
        )
    cfg["surface_force_fail_if_empty"] = bool(cfg.get("surface_force_fail_if_empty", True))

    def _lambda_list(name: str, default):
        value = cfg.get(name, default)
        if isinstance(value, str) and value.strip().lower() == "auto":
            value = default
        return ensure_list_of_positive_numbers(list(value), name)

    cfg["single_lambda_values"] = _lambda_list("single_lambda_values", cfg.get("surface_force_single_lambda_values", [1.0]))
    cfg["pair_lambda_values"] = _lambda_list("pair_lambda_values", cfg.get("surface_force_pair_lambda_values", [cfg["single_lambda_values"][0] / 2.0]))
    cfg["triple_lambda_values"] = _lambda_list("triple_lambda_values", cfg.get("surface_force_triple_lambda_values", [cfg["single_lambda_values"][0] / 3.0]))
    cfg["dual_lambda_value"] = float(cfg.get("dual_lambda_value", cfg.get("surface_force_dual_lambda_value", cfg["single_lambda_values"][0])))
    if not math.isfinite(cfg["dual_lambda_value"]) or cfg["dual_lambda_value"] <= 0.0:
        raise ValueError("dual_lambda_value должен быть конечным числом > 0.")

    # Для load-controlled STEP 4-й скрипт обязан брать q из d3plot/fitted FE state,
    # а не из имени файла и не из командного control vector.
    cfg["step_requires_fitted_fe_q"] = bool(cfg.get("step_requires_fitted_fe_q", cfg["step_load_scheme"] != "prescribed_motion_step"))
    cfg["step_expected_reaction_source"] = str(cfg.get(
        "step_expected_reaction_source",
        "applied_load_manifest" if cfg["step_load_scheme"] == "surface_nodal_force_step" else "bndout",
    ))
    return cfg


def field10(text: object) -> str:
    s = str(text)
    if len(s) > FIELD_WIDTH:
        raise ValueError(f"Значение '{s}' не помещается в поле шириной {FIELD_WIDTH}.")
    return s.rjust(FIELD_WIDTH)


def compact_exp_string(x: float, digits: int) -> str:
    s = f"{x:.{digits}E}".upper()
    mant, exp = s.split("E")
    mant = mant.rstrip("0").rstrip(".")
    if mant in ("-0", "+0", "0"):
        mant = "0"
    exp_i = int(exp)
    return f"{mant}E{exp_i}"


def float10_sf(x: float) -> str:
    """
    Форматирует SF для LS-DYNA fixed-width поля 10 символов.

    Важно: маленькие q/SF нельзя сначала писать через fixed decimal,
    потому что 1e-8 превращается в 0.000000 -> 0.0.
    Поэтому для |x| < 1e-4 сразу используем компактную экспоненциальную запись.
    """
    x = float(x)
    if abs(x) <= 1.0e-30:
        return field10("0.0")

    # Малые значения обязательно пишем в E-формате, иначе они округляются в 0.0.
    if abs(x) < 1.0e-4:
        for digits in (5, 4, 3, 2, 1, 0):
            s = compact_exp_string(x, digits)
            if len(s) <= FIELD_WIDTH:
                return field10(s)

    for digits in (6, 5, 4, 3, 2, 1, 0):
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        if s in ("-0", "+0", ""):
            s = "0"
        if "." not in s and "E" not in s.upper():
            s += ".0"
        if len(s) <= FIELD_WIDTH:
            return field10(s)

    for digits in (5, 4, 3, 2, 1, 0):
        s = compact_exp_string(x, digits)
        if len(s) <= FIELD_WIDTH:
            return field10(s)
    raise ValueError(f"Значение SF={x} не помещается в поле шириной {FIELD_WIDTH}.")


def curve_head10(text: object) -> str:
    s = str(text)
    if len(s) > 10:
        raise ValueError(f"Значение '{s}' не помещается в поле шириной 10.")
    return s.rjust(10)


def curve_point20(x: float) -> str:
    x = float(x)
    if abs(x) <= 1.0e-30:
        return "0.0".rjust(20)
    for digits in (6, 5, 4, 3, 2, 1, 0):
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        if "." not in s:
            s += ".0"
        if len(s) <= 20:
            return s.rjust(20)
    return f"{x:.6E}".upper().rjust(20)


def q_to_tag(q_value: float) -> str:
    """
    Compact filename-safe numeric tag.

    Examples:
      1.0                  -> 1
      0.5                  -> 5e-1
      1.6071003336e-08     -> 16071e-12
      5.3567001112e-09     -> 53567e-13

    Exact values are written to step_manifest.json; the filename is only a compact label.
    """
    x = abs(float(q_value))
    if not math.isfinite(x):
        raise ValueError(f"Некорректное значение для имени файла: {q_value}")
    if x <= 1.0e-30:
        return "0"

    s = f"{x:.5e}"
    mant, exp = s.split("e")
    exp_i = int(exp)

    mant = mant.rstrip("0").rstrip(".")
    if "." in mant:
        before, after = mant.split(".")
        digits = before + after
        exp_i -= len(after)
    else:
        digits = mant

    digits = digits.lstrip("0") or "0"
    if exp_i == 0:
        return digits
    return f"{digits}e{exp_i}"


def sign_to_tag(sign: int) -> str:
    return "p" if sign >= 0 else "m"


def detect_mode_number(file_path: Path, text: str) -> int:
    match = FILENAME_MODE_RE.search(file_path.stem)
    if match:
        return int(match.group(1))
    for line in text.splitlines()[:10]:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", line)
        if len(nums) >= 1 and "MODE" in line.upper():
            try:
                return int(round(float(nums[0])))
            except Exception:
                pass
    raise ValueError(f"Не удалось определить номер моды из файла {file_path.name}.")


def parse_mode_file(file_path: Path, config: dict) -> ModeShape:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    raw_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"Пустой файл моды: {file_path}")

    mode_number = detect_mode_number(file_path, text)
    frequency_hz = None
    for line in raw_lines[:10]:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", line)
        if "FREQ_HZ" in line.upper() and len(nums) >= 2:
            try:
                frequency_hz = float(nums[1]); break
            except Exception:
                pass

    node_values_raw = {}
    for line in raw_lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            node = int(round(float(parts[0])))
            x, y, z = map(float, parts[1:4])
            ux, uy, uz = map(float, parts[4:7])
            rotx = float(parts[7]) if len(parts) > 7 else 0.0
            roty = float(parts[8]) if len(parts) > 8 else 0.0
            rotz = float(parts[9]) if len(parts) > 9 else 0.0
        except Exception:
            continue
        node_values_raw[node] = {
            "X": x, "Y": y, "Z": z,
            "UX": ux, "UY": uy, "UZ": uz,
            "ROTX": rotx, "ROTY": roty, "ROTZ": rotz,
        }

    if not node_values_raw:
        raise ValueError(f"В файле {file_path.name} не найдено узловых модальных данных.")

    if config["force_normal_axis"] is not None:
        dominant_dof = "U" + config["force_normal_axis"]
    else:
        maxabs_by_dof = {dof: max(abs(vals[dof]) for vals in node_values_raw.values()) for dof in config["projection_dofs"]}
        dominant_dof = max(maxabs_by_dof, key=maxabs_by_dof.get)

    normalization_value = max(abs(vals[dominant_dof]) for vals in node_values_raw.values())
    if normalization_value <= 1.0e-30:
        raise ValueError(f"В файле {file_path.name} максимум по {dominant_dof} слишком мал для нормировки.")

    node_values_norm = {
        node: {
            "X": vals["X"],
            "Y": vals["Y"],
            "Z": vals["Z"],
            "UX": vals["UX"] / normalization_value,
            "UY": vals["UY"] / normalization_value,
            "UZ": vals["UZ"] / normalization_value,
            "ROTX": vals["ROTX"] / normalization_value,
            "ROTY": vals["ROTY"] / normalization_value,
            "ROTZ": vals["ROTZ"] / normalization_value,
        }
        for node, vals in node_values_raw.items()
    }

    return ModeShape(mode_number, file_path, frequency_hz, dominant_dof, normalization_value, node_values_norm)


def discover_mode_shapes(modal_dir: Path, config: dict) -> Dict[int, ModeShape]:
    result = {}
    for file_path in sorted(modal_dir.glob("*.txt")):
        if "mode" not in file_path.stem.lower():
            continue
        try:
            shape = parse_mode_file(file_path, config)
        except Exception:
            continue
        result[shape.mode_number] = shape
    return result


def build_define_curve_block(curve_id: int, end_time: float) -> List[str]:
    return [
        "*DEFINE_CURVE",
        "$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint",
        f"{curve_head10(curve_id)}{curve_head10(0)}{curve_head10('1.0')}{curve_head10('1.0')}{curve_head10('0.0')}{curve_head10('0.0')}{curve_head10(0)}{curve_head10(0)}",
        "$#                a1                  o1",
        f"{curve_point20(0.0)}{curve_point20(0.0)}",
        f"{curve_point20(end_time)}{curve_point20(1.0)}",
    ]


def parse_set_node_lists_from_key(base_lines: List[str]) -> Dict[int, Set[int]]:
    sets = {}
    i = 0
    n = len(base_lines)
    while i < n:
        up = base_lines[i].strip().upper()
        if not up.startswith("*SET_NODE_LIST"):
            i += 1
            continue
        titled = "TITLE" in up
        i += 1
        if titled and i < n and not base_lines[i].strip().startswith("*"):
            i += 1
        while i < n:
            s = base_lines[i].strip()
            if not s or s.startswith("$"):
                i += 1
                continue
            break
        if i >= n or base_lines[i].strip().startswith("*"):
            continue
        try:
            sid = int(round(float(base_lines[i].split()[0])))
        except Exception:
            i += 1
            continue
        sets.setdefault(sid, set())
        i += 1
        while i < n:
            s = base_lines[i].strip()
            if not s or s.startswith("$"):
                i += 1
                continue
            if s.startswith("*"):
                break
            for token in s.split():
                try:
                    node_id = int(round(float(token)))
                except Exception:
                    continue
                if node_id > 0:
                    sets[sid].add(node_id)
            i += 1
    return sets


def parse_boundary_spc_set_ids_from_key(base_lines: List[str]) -> Set[int]:
    """
    Возвращает SID из *BOUNDARY_SPC_SET.

    Эти узлы нельзя одновременно использовать в *BOUNDARY_PRESCRIBED_MOTION_NODE,
    иначе STEP-deck содержит конфликт: узел закреплен и ему же навязано модальное
    перемещение.
    """
    out: Set[int] = set()
    i = 0
    n = len(base_lines)
    while i < n:
        up = base_lines[i].strip().upper()
        if not up.startswith("*BOUNDARY_SPC_SET"):
            i += 1
            continue
        i += 1
        while i < n:
            s = base_lines[i].strip()
            if not s or s.startswith("$"):
                i += 1
                continue
            if s.startswith("*"):
                break
            parts = s.split()
            if parts:
                try:
                    out.add(int(round(float(parts[0]))))
                except Exception:
                    pass
            i += 1
            # У BOUNDARY_SPC_SET обычно одна строка данных на keyword.
            break
    return out


def parse_boundary_spc_node_ids_from_key(base_lines: List[str]) -> Set[int]:
    """Возвращает NID из *BOUNDARY_SPC_NODE."""
    out: Set[int] = set()
    i = 0
    n = len(base_lines)
    while i < n:
        up = base_lines[i].strip().upper()
        if not up.startswith("*BOUNDARY_SPC_NODE"):
            i += 1
            continue
        i += 1
        while i < n:
            s = base_lines[i].strip()
            if not s or s.startswith("$"):
                i += 1
                continue
            if s.startswith("*"):
                break
            parts = s.split()
            if parts:
                try:
                    out.add(int(round(float(parts[0]))))
                except Exception:
                    pass
            i += 1
    return out


def collect_excluded_nodes(base_lines: List[str], config: dict) -> Set[int]:
    all_sets = parse_set_node_lists_from_key(base_lines)
    out: Set[int] = set()

    if config.get("exclude_all_set_node_lists", False):
        for nodes in all_sets.values():
            out.update(nodes)

    for sid in config.get("exclude_node_set_ids", []):
        out.update(all_sets.get(int(sid), set()))

    if bool(config.get("auto_exclude_spc_nodes", True)):
        if bool(config.get("auto_exclude_spc_sets", True)):
            spc_set_ids = parse_boundary_spc_set_ids_from_key(base_lines)
            for sid in spc_set_ids:
                out.update(all_sets.get(sid, set()))
        if bool(config.get("auto_exclude_spc_node_cards", True)):
            out.update(parse_boundary_spc_node_ids_from_key(base_lines))

    return out


def key_has_keyword(base_lines: List[str], keyword_prefix: str) -> bool:
    prefix = keyword_prefix.upper()
    return any(line.strip().upper().startswith(prefix) for line in base_lines)


def mode_has_rotations(mode_shapes: Dict[int, ModeShape], tol: float = 1.0e-12) -> bool:
    for shape in mode_shapes.values():
        for vals in shape.node_values.values():
            if any(abs(vals.get(dof, 0.0)) > tol for dof in rotational_dofs()):
                return True
    return False


def geometry_spans_from_modes(mode_shapes: Dict[int, ModeShape]) -> Dict[str, float]:
    shape = next(iter(mode_shapes.values()))
    xs = [vals.get("X", float("nan")) for vals in shape.node_values.values()]
    ys = [vals.get("Y", float("nan")) for vals in shape.node_values.values()]
    zs = [vals.get("Z", float("nan")) for vals in shape.node_values.values()]

    if not xs or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)) or not np.all(np.isfinite(zs)):
        raise ValueError(
            "В mode-файлах не найдены корректные координаты X/Y/Z. "
            "Без координат нельзя автоматически определить нормальную ось solid/shell модели."
        )

    spans = {
        "X": max(xs) - min(xs),
        "Y": max(ys) - min(ys),
        "Z": max(zs) - min(zs),
    }

    if max(spans.values()) <= 1.0e-30:
        raise ValueError(
            "Геометрические spans по X/Y/Z равны нулю. "
            "Проверьте parse_mode_file(): координаты X/Y/Z должны сохраняться в ModeShape.node_values."
        )

    return spans



def parse_node_coordinates_from_key(base_lines: List[str]) -> Dict[int, np.ndarray]:
    """Читает координаты *NODE из base .key."""
    coords: Dict[int, np.ndarray] = {}
    i = 0
    n = len(base_lines)
    while i < n:
        up = base_lines[i].strip().upper()
        if not up.startswith("*NODE"):
            i += 1
            continue
        i += 1
        while i < n:
            s = base_lines[i].strip()
            if not s or s.startswith("$"):
                i += 1
                continue
            if s.startswith("*"):
                break
            vals = split_lsdyna_numeric_fields(base_lines[i])
            if len(vals) >= 4:
                try:
                    nid = int(round(float(vals[0])))
                    coords[nid] = np.array([float(vals[1]), float(vals[2]), float(vals[3])], dtype=float)
                except Exception:
                    pass
            i += 1
    return coords


def estimate_solid_min_edge_length_from_modes(
    base_lines: List[str],
    mode_shapes: Dict[int, ModeShape],
    *,
    max_elements: int = 200000,
) -> Optional[float]:
    """
    Оценивает минимальную длину ребра solid-элементов по *NODE/*ELEMENT_SOLID.

    Если координаты *NODE по какой-то причине не прочитались, используется fallback
    по координатам из mode-файлов. Это нужно именно для 3D displacement STEP:
    если модальная форма нормирована так, что max(|U|)=1, то q имеет размерность
    длины. Масштабирование q от min_edge гарантирует, что навязанное поле
    перемещений мало относительно размера КЭ.
    """
    try:
        solid_elements = parse_solid_elements_from_key(base_lines)
    except Exception:
        return None
    if not solid_elements:
        return None

    coords = parse_node_coordinates_from_key(base_lines)
    if not coords:
        base_mode = next(iter(mode_shapes.values()))
        coords = {
            int(nid): np.array([vals["X"], vals["Y"], vals["Z"]], dtype=float)
            for nid, vals in base_mode.node_values.items()
            if all(math.isfinite(float(vals.get(k, float("nan")))) for k in ("X", "Y", "Z"))
        }
    if not coords:
        return None

    min_edge = float("inf")
    checked = 0
    for _, nodes in solid_elements[:max(1, int(max_elements))]:
        corners = element_corner_nodes(nodes)
        if len(corners) < 4:
            continue
        if any(nid not in coords for nid in corners):
            continue
        checked += 1
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                edge = float(np.linalg.norm(coords[corners[j]] - coords[corners[i]]))
                if edge > 1.0e-30:
                    min_edge = min(min_edge, edge)

    if checked <= 0 or not math.isfinite(min_edge):
        return None
    return float(min_edge)


def build_q_settings_from_q_sweep(q_sweep: List[float], config: dict, source: str) -> dict:
    """
    Сохраняет несколько очень малых q для проверки устойчивости STEP-коэффициентов.
    Для pair/triple q уменьшается по той же политике combined_step_q_scaling.
    """
    q_clean = _unique_positive_sorted(list(q_sweep))
    if not q_clean:
        raise ValueError("q_sweep пуст: невозможно сформировать STEP q-settings.")

    selected_count = len(config.get("selected_modes", []))
    pair_factor = _interaction_q_factor(2, config) if selected_count >= 2 else 1.0
    triple_factor = _interaction_q_factor(3, config) if selected_count >= 3 else 1.0

    single_values = list(q_clean)
    pair_values = [q * pair_factor for q in q_clean] if selected_count >= 2 else []
    triple_values = [q * triple_factor for q in q_clean] if selected_count >= 3 else []

    if bool(config.get("include_single_q_values_for_all_interaction_q", True)):
        single_values = single_values + pair_values + triple_values

    q_mid = q_clean[len(q_clean) // 2]
    return {
        "single_q_values": _unique_q_values_preserve_order(single_values),
        "pair_q_values": _unique_q_values_preserve_order(pair_values),
        "triple_q_values": _unique_q_values_preserve_order(triple_values),
        "dual_p_value": float(q_mid) * float(config.get("dual_q_factor", 1.0)),
        "q_auto_source": source + ":full_q_sweep",
        "q_auto_sweep": q_clean,
        "q_scale_basis": str(config.get("solid_q_scale_basis", "min_edge")),
        "combined_step_q_scaling": str(config.get("combined_step_q_scaling", "active_count")),
    }


def detect_model_info(base_lines: List[str], mode_shapes: Dict[int, ModeShape], config: dict) -> dict:
    has_shell = (
        key_has_keyword(base_lines, "*ELEMENT_SHELL")
        or key_has_keyword(base_lines, "*SECTION_SHELL")
        or key_has_keyword(base_lines, "*PART_COMPOSITE")
    )
    has_solid = key_has_keyword(base_lines, "*ELEMENT_SOLID") or key_has_keyword(base_lines, "*SECTION_SOLID")
    rotations_present = mode_has_rotations(mode_shapes)
    spans = geometry_spans_from_modes(mode_shapes)
    max_span = max(spans.values()) if spans else 0.0
    min_span = min((v for v in spans.values() if v > 0.0), default=0.0)
    flat_ratio = min_span / max(max_span, 1.0e-30) if max_span > 0.0 else 0.0
    normal_axis = min(spans, key=spans.get) if spans else "Z"

    requested = str(config.get("model_family", "auto")).lower()
    if requested != "auto":
        family = requested
    elif has_solid and not has_shell:
        family = "solid"
    elif has_shell and not has_solid:
        family = "shell"
    elif rotations_present and has_shell:
        family = "shell"
    elif rotations_present and not has_solid:
        family = "shell"
    elif has_solid:
        family = "solid"
    else:
        family = "shell" if rotations_present else "solid"

    thickness = spans.get(normal_axis, 0.0)
    if thickness <= 1.0e-30:
        thickness = float(config.get("shell_thickness", config.get("thickness_fallback", 1.0)))

    min_solid_edge = estimate_solid_min_edge_length_from_modes(
        base_lines,
        mode_shapes,
        max_elements=int(config.get("preflight_max_sample_elements", 200000)),
    ) if has_solid else None

    return {
        "family": family,
        "has_shell_keywords": has_shell,
        "has_solid_keywords": has_solid,
        "rotations_present": rotations_present,
        "geometry_spans": spans,
        "flat_ratio": flat_ratio,
        "normal_axis_geometry": normal_axis,
        "thickness_estimate": float(thickness),
        "min_solid_edge_length": float(min_solid_edge) if min_solid_edge is not None else None,
    }


def resolve_case_prescribed_dofs(config: dict, model_info: dict) -> List[str]:
    raw = config["prescribed_dofs"]
    if raw != "auto":
        return list(raw)

    normal_dof = "U" + str(model_info.get("normal_axis_geometry", "Z")).upper()
    mode = str(config.get("primary_step_prescribed_mode", "auto")).strip().lower()
    family = str(model_info.get("family", "solid")).lower()

    if mode == "auto":
        # STEP задаёт модальную форму как вектор перемещений.
        # Для solid корректный первичный STEP — полный поступательный
        # вектор UX/UY/UZ. Режим normal_only оставлен только как явный
        # специальный режим для отладки.
        if family == "solid":
            return ["UX", "UY", "UZ"]
        if family == "shell" and model_info.get("rotations_present", False):
            return ["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"]
        return [normal_dof]

    if mode == "normal_only":
        return [normal_dof]
    if mode == "translational":
        return ["UX", "UY", "UZ"]
    if mode == "normal_plus_rotations":
        if model_info.get("rotations_present", False):
            return [normal_dof, "ROTX", "ROTY", "ROTZ"]
        return [normal_dof]
    if mode == "full":
        if family == "shell" and model_info.get("rotations_present", False):
            return ["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"]
        return ["UX", "UY", "UZ"]

    raise ValueError(f"Неизвестный primary_step_prescribed_mode: {mode}")


def _scale_q_values(ratios: List[float], scale: float) -> List[float]:
    return [float(r) * float(scale) for r in ratios]


def _unique_positive_sorted(values: List[float]) -> List[float]:
    out: List[float] = []
    seen = set()
    for value in values:
        v = float(value)
        if not math.isfinite(v) or v <= 0.0:
            continue
        key = round(v, 14)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return sorted(out)


def _resolve_q_sweep_from_scale(config: dict, model_info: dict) -> Tuple[List[float], str]:
    """
    q в STEP — это не подбираемый физический параметр, а масштаб
    вычислительного эксперимента. Чтобы результат не зависел от одного
    выбранного q, генерируется q-sweep. Четвертый скрипт затем сам выбирает
    устойчивую область q по коэффициентам K,G,H.
    """
    family = str(model_info.get("family", "solid")).lower()
    thickness = max(float(model_info.get("thickness_estimate", 1.0)), 1.0e-30)

    if family == "solid":
        basis = str(config.get("solid_q_scale_basis", "min_edge")).strip().lower()
        max_span = max((float(v) for v in model_info.get("geometry_spans", {}).values()), default=thickness)
        min_edge = model_info.get("min_solid_edge_length")

        if basis == "min_edge" and min_edge is not None and math.isfinite(float(min_edge)) and float(min_edge) > 0.0:
            scale = float(min_edge)
            ratios = config.get("solid_q_over_min_edge_sweep", config.get("solid_common_q_over_min_edge", [1.0e-5, 3.0e-5, 1.0e-4]))
            return _unique_positive_sorted(_scale_q_values(list(ratios), scale)), "solid_min_edge_sweep"

        if basis == "max_span" and math.isfinite(float(max_span)) and float(max_span) > 0.0:
            scale = float(max_span)
            ratios = config.get("solid_q_over_max_span_sweep", [1.0e-6, 3.0e-6, 1.0e-5])
            return _unique_positive_sorted(_scale_q_values(list(ratios), scale)), "solid_max_span_sweep"

        ratios = config.get("solid_q_over_thickness_sweep", None)
        if ratios is None:
            ratios = config.get("solid_common_q_over_thickness", [1.0e-5, 3.0e-5, 1.0e-4])
        return _unique_positive_sorted(_scale_q_values(list(ratios), thickness)), "solid_thickness_sweep"

    # Для shell сначала пробуем работать через толщину, если она задана явно
    # или извлечена как fallback. Если пользователь не указал толщину оболочки,
    # используется классический набор q в единицах нормированной модальной формы.
    if "shell_q_over_thickness_sweep" in config:
        ratios = list(config.get("shell_q_over_thickness_sweep", [0.3, 0.7, 1.0]))
        return _unique_positive_sorted(_scale_q_values(ratios, thickness)), "shell_thickness_sweep"

    return _unique_positive_sorted(list(config.get("shell_q_sweep_values", [0.25, 0.5, 1.0]))), "shell_direct_sweep"


def _interaction_q_factor(active_count: int, config: dict) -> float:
    if active_count <= 1:
        return 1.0

    manual_key = "pair_q_factor" if active_count == 2 else "triple_q_factor"
    manual_value = config.get(manual_key, None)
    if manual_value is not None:
        value = float(manual_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{manual_key} должен быть конечным числом > 0.")
        return value

    policy = str(config.get("combined_step_q_scaling", "active_count")).strip().lower()
    if policy == "none":
        return 1.0
    if policy == "sqrt_active_count":
        return 1.0 / math.sqrt(float(active_count))
    if policy in ("active_count", "manual"):
        return 1.0 / float(active_count)
    raise ValueError(f"Неизвестный combined_step_q_scaling: {policy}")


def _unique_q_values_preserve_order(values: List[float]) -> List[float]:
    out: List[float] = []
    seen = set()
    for value in values:
        v = float(value)
        if not math.isfinite(v) or v <= 0.0:
            continue
        key = round(v, 14)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _apply_combined_step_q_scaling(q_base: float, config: dict, source: str) -> dict:
    """
    Делает STEP-нагружения устойчивее.

    q_base берется из real_load FE как характерная одиночная модальная амплитуда.
    Для pair/triple нельзя автоматически задавать тот же q на каждую моду: суммарное
    prescribed displacement поле становится в 2-3 раза жестче одиночного случая и
    может сорвать nonlinear implicit расчет. Поэтому q для взаимодействий уменьшается
    пропорционально числу одновременно активных мод.

    Важно: single_q_values дополняется q_pair и q_triple, потому что формулы STEP
    для pair/triple требуют single +/- реакции при том же q.
    """
    q_base = float(q_base)
    if not math.isfinite(q_base) or q_base <= 0.0:
        raise ValueError("q_base должен быть конечным числом > 0.")

    selected_count = len(config.get("selected_modes", []))
    q_single_main = q_base
    q_pair = q_base * _interaction_q_factor(2, config)
    q_triple = q_base * _interaction_q_factor(3, config)
    q_dual = q_base * float(config.get("dual_q_factor", 1.0))

    if bool(config.get("include_single_q_values_for_all_interaction_q", True)):
        single_values = [q_single_main]
        if selected_count >= 2:
            single_values.append(q_pair)
        if selected_count >= 3:
            single_values.append(q_triple)
    else:
        single_values = [q_single_main]

    pair_values = [q_pair] if selected_count >= 2 else []
    triple_values = [q_triple] if selected_count >= 3 else []

    return {
        "single_q_values": _unique_q_values_preserve_order(single_values),
        "pair_q_values": _unique_q_values_preserve_order(pair_values),
        "triple_q_values": _unique_q_values_preserve_order(triple_values),
        "dual_p_value": float(q_dual),
        "q_auto_source": source,
        "q_base_single": float(q_single_main),
        "q_pair_scaled": float(q_pair),
        "q_triple_scaled": float(q_triple),
        "q_dual_scaled": float(q_dual),
        "combined_step_q_scaling": str(config.get("combined_step_q_scaling", "active_count")),
    }


def resolve_case_q_settings(case_dir: Path, config: dict, model_info: dict) -> dict:
    """
    q для displacement STEP.

    В 3D-prescribed варианте основной путь — очень малые q, масштабированные
    от минимального ребра solid-элемента. FE-калибровка отключена по умолчанию.
    """
    # Явные q из JSON имеют приоритет над auto-масштабированием.
    if config.get("single_q_values") != "auto":
        single = list(config.get("single_q_values", []))
        if not single:
            raise ValueError("single_q_values задан, но список пуст.")
        selected_count = len(config.get("selected_modes", []))
        pair = config.get("pair_q_values")
        triple = config.get("triple_q_values")
        if pair == "auto":
            pair = [q * _interaction_q_factor(2, config) for q in single] if selected_count >= 2 else []
        if triple == "auto":
            triple = [q * _interaction_q_factor(3, config) for q in single] if selected_count >= 3 else []
        dual = config.get("dual_p_value")
        if dual == "auto":
            dual = single[len(single) // 2] * float(config.get("dual_q_factor", 1.0))
        return {
            "single_q_values": _unique_q_values_preserve_order(single),
            "pair_q_values": _unique_q_values_preserve_order(list(pair or [])),
            "triple_q_values": _unique_q_values_preserve_order(list(triple or [])),
            "dual_p_value": float(dual),
            "q_auto_source": "user_config_explicit_q_values",
        }

    q_sweep, source = _resolve_q_sweep_from_scale(config, model_info)
    if not q_sweep:
        raise ValueError("Не удалось автоматически построить q.")

    q0 = q_sweep[len(q_sweep) // 2]
    if (
        str(config.get("step_load_scheme", "prescribed_motion_step")).lower() == "prescribed_motion_step"
        and bool(config.get("use_full_q_sweep_for_prescribed_step", True))
        and not bool(config.get("use_per_mode_q_for_solid", False))
    ):
        return build_q_settings_from_q_sweep(q_sweep, config, source)

    if str(model_info.get("family", "solid")).lower() == "solid" and bool(config.get("use_per_mode_q_for_solid", False)):
        q_by_mode = {str(m): float(q0) for m in config.get("selected_modes", [])}
        return _apply_per_mode_q_settings(
            q_by_mode=q_by_mode,
            config=config,
            source=source + ":per_mode_fallback_equal_q:combined_scaled",
        )
    return _apply_combined_step_q_scaling(
        q_base=q0,
        config=config,
        source=source + ":single_mid_fallback:combined_scaled",
    )


def build_motion_records(
    q_vector,
    mode_shapes,
    prescribed_dofs,
    zero_tol,
    excluded_nodes=None,
    prescribed_nodes: Optional[Set[int]] = None,
):
    base_mode = next(iter(mode_shapes.values()))
    all_nodes = sorted(base_mode.node_values.keys())
    for mode_number, shape in mode_shapes.items():
        if sorted(shape.node_values.keys()) != all_nodes:
            raise ValueError(f"Набор узлов у mode {mode_number} не совпадает с остальными mode-файлами.")
    excluded_nodes = excluded_nodes or set()
    allowed_nodes = set(all_nodes) if prescribed_nodes is None else (set(prescribed_nodes) & set(all_nodes))
    active_nodes = [nid for nid in all_nodes if nid in allowed_nodes and nid not in excluded_nodes]
    target = {nid: {dof: 0.0 for dof in prescribed_dofs} for nid in active_nodes}
    for mode_number, q_value in q_vector.items():
        if abs(q_value) <= 1.0e-30:
            continue
        node_values = mode_shapes[mode_number].node_values
        for nid in active_nodes:
            for dof in prescribed_dofs:
                target[nid][dof] += q_value * node_values[nid][dof]
    dof_map = {"UX": 1, "UY": 2, "UZ": 3, "ROTX": 4, "ROTY": 5, "ROTZ": 6}
    records = []
    for nid in active_nodes:
        for dof in prescribed_dofs:
            value = target[nid][dof]
            if abs(value) <= zero_tol:
                continue
            records.append((nid, dof_map[dof], value))
    return records




def build_dual_motion_records(
    q_vector: Dict[int, float],
    mode_shapes: Dict[int, ModeShape],
    zero_tol: float,
    excluded_nodes: Optional[Set[int]] = None,
    prescribed_nodes: Optional[Set[int]] = None,
    *,
    dual_prescribed_mode: str = "normal_only",
    dual_prescribed_dofs: Optional[List[str]] = None,
) -> List[Tuple[int, int, float]]:
    """
    Dual-deck не должен задавать весь вектор модальной формы (UX, UY, UZ),
    иначе in-plane перемещения будут навязаны кинематически.

    Для выделения dual / in-plane response по умолчанию задаём только
    доминирующий поперечный DOF конкретной моды, а in-plane DOF оставляем
    свободными. Тогда LS-DYNA сам формирует quadratic in-plane field,
    и 4-й скрипт может корректно вычислять dual field:
        w_dual_raw = w(2p) - 2 * w(p)
    """
    active_mode_items = [(mode_number, q_value) for mode_number, q_value in q_vector.items() if abs(q_value) > 1.0e-30]
    if len(active_mode_items) != 1:
        raise ValueError(
            "Dual-deck должен строиться только для одной моды. "
            f"Получено q_vector={q_vector}"
        )

    mode_number, q_value = active_mode_items[0]
    if mode_number not in mode_shapes:
        raise ValueError(f"Для dual-deck не найдена mode shape #{mode_number}.")

    shape = mode_shapes[mode_number]
    if dual_prescribed_mode == "prescribed_dofs":
        prescribed_dofs = list(dual_prescribed_dofs or [])
        if not prescribed_dofs:
            raise ValueError(
                "При dual_prescribed_mode='prescribed_dofs' нужно задать dual_prescribed_dofs."
            )
    else:
        prescribed_dofs = [shape.dominant_dof]

    excluded_nodes = excluded_nodes or set()
    all_nodes = sorted(shape.node_values.keys())
    allowed_nodes = set(all_nodes) if prescribed_nodes is None else (set(prescribed_nodes) & set(all_nodes))
    active_nodes = [nid for nid in all_nodes if nid in allowed_nodes and nid not in excluded_nodes]

    dof_map = {"UX": 1, "UY": 2, "UZ": 3, "ROTX": 4, "ROTY": 5, "ROTZ": 6}
    records: List[Tuple[int, int, float]] = []

    for nid in active_nodes:
        for dof in prescribed_dofs:
            value = q_value * shape.node_values[nid][dof]
            if abs(value) > zero_tol:
                records.append((nid, dof_map[dof], float(value)))

    return records

def build_boundary_prescribed_motion_node_block(motion_records, curve_id):
    lines = []
    for nid, dof_id, sf_value in motion_records:
        lines.append("*BOUNDARY_PRESCRIBED_MOTION_NODE")
        lines.append("$#     nid       dof       vad      lcid        sf       vid     death     birth")
        lines.append(f"{field10(nid)}{field10(dof_id)}{field10(2)}{field10(curve_id)}{float10_sf(sf_value)}{field10(0)}{field10('1.0E28')}{field10('0.0')}")
    return lines


def field10_label(text: object) -> str:
    s = str(text)
    if len(s) > FIELD_WIDTH:
        s = s[:FIELD_WIDTH]
    return s.rjust(FIELD_WIDTH)


def float10_general(x: float) -> str:
    x = float(x)
    if abs(x) <= 1.0e-30:
        return field10("0")

    for digits in (8, 7, 6, 5, 4, 3, 2, 1, 0):
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        if s in ("-0", "+0", ""):
            s = "0"
        if len(s) <= FIELD_WIDTH:
            return field10(s)

    for digits in (4, 3, 2, 1, 0):
        s = compact_exp_string(x, digits)
        if len(s) <= FIELD_WIDTH:
            return field10(s)

    raise ValueError(f"Значение {x} не помещается в поле шириной {FIELD_WIDTH}.")


def field10_value(value) -> str:
    if isinstance(value, int):
        return field10(str(value))
    if isinstance(value, float):
        return float10_general(value)
    try:
        ivalue = int(value)
        if str(ivalue) == str(value):
            return field10(str(ivalue))
    except Exception:
        pass
    try:
        fvalue = float(value)
        return float10_general(fvalue)
    except Exception:
        return field10(str(value))


def fixed_comment_line(names):
    return "$#" + "".join(field10_label(name) for name in names)


def fixed_data_line(values):
    return "".join(field10_value(v) for v in values)


def build_control_implicit_solution_block(params):
    return [
        "*CONTROL_IMPLICIT_SOLUTION",
        fixed_comment_line(["nsolvr", "ilimit", "maxref", "dctol", "ectol", "rctol", "lstol", "abstol"]),
        fixed_data_line([
            params["nsolvr"], params["ilimit"], params["maxref"], params["dctol"],
            params["ectol"], params["rctol"], params["lstol"], params["abstol"]
        ]),
        fixed_comment_line(["dnorm", "diverg", "istif", "nlprint", "nlnorm", "d3itctl", "cpchk", "unused1"]),
        fixed_data_line([
            params["dnorm"], params["diverg"], params["istif"], params["nlprint"],
            params["nlnorm"], params["d3itctl"], params["cpchk"], 0
        ]),
        fixed_comment_line(["arcctl", "arcdir", "arclen", "arcmth", "arcdmp", "arcpsi", "arcalf", "arctim"]),
        fixed_data_line([
            params["arcctl"], params["arcdir"], params["arclen"], params["arcmth"],
            params["arcdmp"], params["arcpsi"], params["arcalf"], params["arctim"]
        ]),
        fixed_comment_line(["lsmtd", "lsdir", "irad", "srad", "awgt", "sred", "kssize"]),
        fixed_data_line([
            params["lsmtd"], params["lsdir"], params["irad"], params["srad"],
            params["awgt"], params["sred"], params["kssize"]
        ]),
    ]


def build_control_implicit_general_block(params):
    return [
        "*CONTROL_IMPLICIT_GENERAL",
        fixed_comment_line(["imflag", "dt0", "imform", "nsbs", "igs", "cnstn", "form", "zero_v"]),
        fixed_data_line([
            params["imflag"], params["dt0"], params["imform"], params["nsbs"],
            params["igs"], params["cnstn"], params["form"], params["zero_v"]
        ]),
    ]


def replace_or_insert_keyword_block(base_lines: List[str], keyword: str, new_block: List[str]) -> List[str]:
    kw = keyword.upper().strip()
    start = None
    end = None
    for i, line in enumerate(base_lines):
        if line.strip().upper() == kw:
            start = i
            j = i + 1
            while j < len(base_lines):
                if base_lines[j].strip().startswith("*"):
                    break
                j += 1
            end = j
            break
    if start is not None:
        return base_lines[:start] + new_block + base_lines[end:]
    end_idx = None
    for i, line in enumerate(base_lines):
        if line.strip().upper().startswith("*END"):
            end_idx = i
            break
    if end_idx is None:
        return list(base_lines) + new_block + ["*END"]
    return base_lines[:end_idx] + new_block + base_lines[end_idx:]


def apply_implicit_patch(base_lines, config, family):
    if not config["patch_implicit_cards"]:
        return list(base_lines)
    if family == "nl":
        sol_params = config["nonlinear_implicit_solution"]
        gen_params = config["nonlinear_implicit_general"]
    elif family == "lin":
        sol_params = config["linear_implicit_solution"]
        gen_params = config["linear_implicit_general"]
    else:
        raise ValueError(f"Неизвестное семейство: {family}")
    patched = list(base_lines)
    if bool(config.get("patch_step_disable_element_erosion", True)):
        patched = patch_control_timestep_erode(patched, int(config.get("patch_step_control_timestep_erode_value", 0)))
    patched = replace_or_insert_keyword_block(patched, "*CONTROL_IMPLICIT_GENERAL", build_control_implicit_general_block(gen_params))
    patched = replace_or_insert_keyword_block(patched, "*CONTROL_IMPLICIT_SOLUTION", build_control_implicit_solution_block(sol_params))
    return patched


def insert_before_end(base_lines: List[str], block_lines: List[str]) -> List[str]:
    end_idx = None
    for i, line in enumerate(base_lines):
        if line.strip().upper().startswith("*END"):
            end_idx = i; break
    if end_idx is None:
        return list(base_lines) + block_lines + ["*END"]
    return list(base_lines[:end_idx]) + block_lines + list(base_lines[end_idx:])


def load_base_key_lines(base_key: Path) -> List[str]:
    return base_key.read_text(encoding="utf-8", errors="ignore").splitlines()


def split_lsdyna_numeric_fields(line: str) -> List[str]:
    """Читает LS-DYNA строки как free-format или fixed-width по 10 символов."""
    s = line.strip()
    if not s or s.startswith("$"):
        return []
    parts = s.split()
    if len(parts) > 1:
        return parts
    # fixed-width fallback
    chunks = [line[i:i + FIELD_WIDTH].strip() for i in range(0, len(line), FIELD_WIDTH)]
    return [c for c in chunks if c]


def parse_solid_elements_from_key(base_lines: List[str]) -> List[Tuple[int, List[int]]]:
    """
    Parser *ELEMENT_SOLID / *ELEMENT_SOLID_*.

    Поддерживаются два распространённых формата LS-DYNA:

    1) Одна строка:
       eid pid n1 n2 n3 n4 ...

    2) Две строки, как часто пишет Workbench LS-DYNA для tet10:
       eid pid
       n1 n2 n3 n4 n5 n6 n7 n8 n9 n10

    Старый parser ошибочно читал вторую строку с узлами как отдельный элемент.
    Из-за этого surface/corner nodes определялись неправильно, и STEP снова
    задавался почти на всю объёмную сетку.
    """
    elements: List[Tuple[int, List[int]]] = []

    def ints_from_line(line: str) -> List[int]:
        vals: List[int] = []
        for token in split_lsdyna_numeric_fields(line):
            try:
                vals.append(int(round(float(token))))
            except Exception:
                pass
        return vals

    def unique_positive(values: List[int]) -> List[int]:
        seen: Set[int] = set()
        out: List[int] = []
        for value in values:
            if value <= 0:
                continue
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    i = 0
    n = len(base_lines)
    while i < n:
        up = base_lines[i].strip().upper()
        if not up.startswith("*ELEMENT_SOLID"):
            i += 1
            continue

        i += 1
        if "TITLE" in up and i < n and not base_lines[i].strip().startswith("*"):
            i += 1

        while i < n:
            s = base_lines[i].strip()
            if not s or s.startswith("$"):
                i += 1
                continue
            if s.startswith("*"):
                break

            vals = ints_from_line(base_lines[i])

            # Free/fixed one-line solid element:
            # eid pid n1 n2 n3 ...
            if len(vals) >= 6:
                eid = vals[0]
                nodes = unique_positive(vals[2:])
                if len(nodes) >= 4:
                    elements.append((eid, nodes))
                i += 1
                continue

            # Workbench two-line solid element:
            # line 1: eid pid
            # line 2: node ids
            if len(vals) >= 2:
                eid = vals[0]
                i += 1

                node_vals: List[int] = []
                while i < n:
                    s2 = base_lines[i].strip()
                    if not s2 or s2.startswith("$"):
                        i += 1
                        continue
                    if s2.startswith("*"):
                        break

                    vals2 = ints_from_line(base_lines[i])

                    # A node line has at least four node ids. The next element
                    # header usually has only eid pid, so do not consume it.
                    if len(vals2) >= 4:
                        node_vals.extend(vals2)
                        i += 1
                    break

                nodes = unique_positive(node_vals)
                if len(nodes) >= 4:
                    elements.append((eid, nodes))
                continue

            i += 1

    return elements


def patch_control_timestep_erode(base_lines: List[str], erode_value: int = 0) -> List[str]:
    """Ставит ERODE в *CONTROL_TIMESTEP в заданное значение, если карта присутствует."""
    out = list(base_lines)
    i = 0
    while i < len(out):
        if out[i].strip().upper().startswith("*CONTROL_TIMESTEP"):
            j = i + 1
            while j < len(out):
                s = out[j].strip()
                if not s or s.startswith("$"):
                    j += 1
                    continue
                if s.startswith("*"):
                    break
                fields = split_lsdyna_numeric_fields(out[j])
                if len(fields) >= 7:
                    fields[6] = str(int(erode_value))
                    out[j] = fixed_data_line(fields)
                return out
        i += 1
    return out


def nodal_displacement_field(
    q_vector: Dict[int, float],
    mode_shapes: Dict[int, ModeShape],
    prescribed_dofs: List[str],
    excluded_nodes: Optional[Set[int]] = None,
    prescribed_nodes: Optional[Set[int]] = None,
) -> Dict[int, Dict[str, float]]:
    base_mode = next(iter(mode_shapes.values()))
    all_nodes = sorted(base_mode.node_values.keys())
    excluded_nodes = excluded_nodes or set()
    allowed_nodes = set(all_nodes) if prescribed_nodes is None else (set(prescribed_nodes) & set(all_nodes))
    active_nodes = [nid for nid in all_nodes if nid in allowed_nodes and nid not in excluded_nodes]
    target = {nid: {dof: 0.0 for dof in ("UX", "UY", "UZ")} for nid in all_nodes}
    for mode_number, q_value in q_vector.items():
        if abs(q_value) <= 1.0e-30:
            continue
        node_values = mode_shapes[mode_number].node_values
        for nid in active_nodes:
            for dof in prescribed_dofs:
                if dof in ("UX", "UY", "UZ"):
                    target[nid][dof] += float(q_value) * float(node_values[nid][dof])
    return target


def mode_node_coordinates(mode_shapes: Dict[int, ModeShape]) -> Dict[int, np.ndarray]:
    shape = next(iter(mode_shapes.values()))
    return {
        nid: np.array([vals["X"], vals["Y"], vals["Z"]], dtype=float)
        for nid, vals in shape.node_values.items()
    }


def tet_signed_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return float(np.linalg.det(np.vstack([b - a, c - a, d - a]).T) / 6.0)


def solid_element_topology(nodes: List[int]) -> str:
    """
    Грубое определение topology по числу узлов.

    Основной целевой случай для текущей задачи — tetra10 из ANSYS/LS-DYNA:
    первые 4 узла являются corner nodes, остальные — midside.
    """
    n = len(nodes)
    if n in (4, 10):
        return "tet"
    if n in (6, 15):
        return "wedge"
    if n in (8, 20):
        return "hex"
    if n >= 10:
        # Безопасный fallback для quadratic tet-like solid.
        return "tet"
    if n >= 8:
        return "hex"
    if n >= 6:
        return "wedge"
    return "unknown"


def element_corner_nodes(nodes: List[int]) -> List[int]:
    topo = solid_element_topology(nodes)
    if topo == "tet":
        return nodes[:4]
    if topo == "wedge":
        return nodes[:6]
    if topo == "hex":
        return nodes[:8]
    return nodes


def element_corner_faces(nodes: List[int]) -> List[Tuple[int, ...]]:
    corners = element_corner_nodes(nodes)
    topo = solid_element_topology(nodes)

    if topo == "tet" and len(corners) >= 4:
        n = corners[:4]
        return [
            (n[0], n[1], n[2]),
            (n[0], n[1], n[3]),
            (n[0], n[2], n[3]),
            (n[1], n[2], n[3]),
        ]

    if topo == "wedge" and len(corners) >= 6:
        n = corners[:6]
        return [
            (n[0], n[1], n[2]),
            (n[3], n[4], n[5]),
            (n[0], n[1], n[4], n[3]),
            (n[1], n[2], n[5], n[4]),
            (n[2], n[0], n[3], n[5]),
        ]

    if topo == "hex" and len(corners) >= 8:
        n = corners[:8]
        return [
            (n[0], n[1], n[2], n[3]),
            (n[4], n[5], n[6], n[7]),
            (n[0], n[1], n[5], n[4]),
            (n[1], n[2], n[6], n[5]),
            (n[2], n[3], n[7], n[6]),
            (n[3], n[0], n[4], n[7]),
        ]

    return []


def collect_solid_corner_nodes(solid_elements: List[Tuple[int, List[int]]]) -> Set[int]:
    out: Set[int] = set()
    for _, nodes in solid_elements:
        out.update(element_corner_nodes(nodes))
    return out


def collect_solid_surface_corner_nodes(solid_elements: List[Tuple[int, List[int]]]) -> Set[int]:
    """
    Находит внешние corner nodes объёмной сетки через грани, встречающиеся один раз.

    Для tetra10 результатом будут только 4 corner nodes на внешних гранях,
    без midside nodes. Это именно нужный режим для более мягкого solid STEP:
    midside/internal nodes остаются свободными.
    """
    face_count: Dict[Tuple[int, ...], int] = {}
    face_original: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    for _, nodes in solid_elements:
        for face in element_corner_faces(nodes):
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1
            face_original[key] = face

    out: Set[int] = set()
    for key, count in face_count.items():
        if count == 1:
            out.update(face_original[key])
    return out


def collect_solid_surface_element_nodes(solid_elements: List[Tuple[int, List[int]]]) -> Set[int]:
    """
    Более широкий surface режим: берёт все узлы элементов, у которых есть внешняя грань.

    Этот режим может включать midside nodes. Для борьбы с неадекватными реакциями
    основной рекомендуемый режим всё равно surface_corner_nodes.
    """
    face_count: Dict[Tuple[int, ...], int] = {}
    face_to_eids: Dict[Tuple[int, ...], List[int]] = {}
    eid_to_nodes: Dict[int, List[int]] = {}

    for eid, nodes in solid_elements:
        eid_to_nodes[eid] = nodes
        for face in element_corner_faces(nodes):
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1
            face_to_eids.setdefault(key, []).append(eid)

    out: Set[int] = set()
    for key, count in face_count.items():
        if count == 1:
            for eid in face_to_eids.get(key, []):
                out.update(eid_to_nodes.get(eid, []))
    return out




def face_area_and_normal(face: Tuple[int, ...], coords: Dict[int, np.ndarray]) -> Tuple[float, np.ndarray]:
    pts = [coords[nid] for nid in face if nid in coords]
    if len(pts) < 3:
        return 0.0, np.zeros(3, dtype=float)

    def tri_area_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, np.ndarray]:
        v = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(v))
        if norm <= 1.0e-30:
            return 0.0, np.zeros(3, dtype=float)
        return 0.5 * norm, v / norm

    if len(pts) == 3:
        return tri_area_normal(pts[0], pts[1], pts[2])

    # quad или poly-face: fan triangulation от первой точки
    area_total = 0.0
    normal_acc = np.zeros(3, dtype=float)
    for k in range(1, len(pts) - 1):
        area, normal = tri_area_normal(pts[0], pts[k], pts[k + 1])
        area_total += area
        normal_acc += normal * area
    norm = float(np.linalg.norm(normal_acc))
    if area_total <= 1.0e-30 or norm <= 1.0e-30:
        return 0.0, np.zeros(3, dtype=float)
    return area_total, normal_acc / norm


def collect_solid_external_faces(solid_elements: List[Tuple[int, List[int]]]) -> List[Tuple[int, ...]]:
    face_count: Dict[Tuple[int, ...], int] = {}
    face_original: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    for _, nodes in solid_elements:
        for face in element_corner_faces(nodes):
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1
            face_original[key] = face
    return [face_original[key] for key, count in face_count.items() if count == 1]


def build_surface_load_data(
    *,
    mode_shapes: Dict[int, ModeShape],
    solid_elements: List[Tuple[int, List[int]]],
    excluded_nodes: Set[int],
    config: dict,
) -> dict:
    """
    Строит геометрию для нового load-controlled STEP.

    Главный принцип: силы прикладываются только к внешним surface/corner nodes.
    Внутренние solid-узлы не получают ни prescribed motion, ни nodal force.
    """
    if not solid_elements:
        if bool(config.get("surface_force_fail_if_empty", True)):
            raise RuntimeError(
                "surface_nodal_force_step требует распознанные *ELEMENT_SOLID. "
                "solid_elements пуст. Проверьте base model.key и parser *ELEMENT_SOLID."
            )
        base_mode = next(iter(mode_shapes.values()))
        nodes = sorted(set(base_mode.node_values.keys()) - set(excluded_nodes))
        return {
            "surface_nodes": set(nodes),
            "node_area": {int(n): 1.0 for n in nodes},
            "node_normal": {int(n): np.array([0.0, 1.0, 0.0], dtype=float) for n in nodes},
            "external_face_count": 0,
            "note": "fallback_all_mode_nodes_no_solid_elements",
        }

    coords = mode_node_coordinates(mode_shapes)
    all_mode_nodes = set(next(iter(mode_shapes.values())).node_values.keys())
    corner_nodes = collect_solid_corner_nodes(solid_elements)
    surface_corner_nodes = collect_solid_surface_corner_nodes(solid_elements)
    surface_element_nodes = collect_solid_surface_element_nodes(solid_elements)

    scope = str(config.get("surface_force_node_scope", "surface_corner_nodes")).strip().lower()
    if scope == "surface_corner_nodes":
        selected = set(surface_corner_nodes)
    elif scope == "surface_nodes":
        selected = set(surface_element_nodes)
        if not bool(config.get("surface_force_use_midside_nodes", False)):
            selected &= set(corner_nodes)
    elif scope == "corner_nodes":
        selected = set(corner_nodes)
    else:
        raise ValueError(f"Неизвестный surface_force_node_scope: {scope}")

    selected = (set(int(n) for n in selected) & all_mode_nodes) - set(excluded_nodes)
    if not selected and bool(config.get("surface_force_fail_if_empty", True)):
        raise RuntimeError(
            "После выбора surface force nodes и исключения SPC-узлов не осталось узлов для нагрузки.\n"
            f"surface_force_node_scope={scope}\n"
            f"surface_corner_node_count={len(surface_corner_nodes)}\n"
            f"surface_element_node_count={len(surface_element_nodes)}\n"
            f"excluded_node_count={len(excluded_nodes)}"
        )

    node_area: Dict[int, float] = {int(n): 0.0 for n in selected}
    normal_acc: Dict[int, np.ndarray] = {int(n): np.zeros(3, dtype=float) for n in selected}
    external_faces = collect_solid_external_faces(solid_elements)

    for face in external_faces:
        area, normal = face_area_and_normal(face, coords)
        if area <= 1.0e-30:
            continue
        face_nodes = [int(n) for n in face if int(n) in selected]
        if not face_nodes:
            continue
        share = area / float(len(face_nodes))
        for nid in face_nodes:
            node_area[nid] = node_area.get(nid, 0.0) + share
            normal_acc[nid] = normal_acc.get(nid, np.zeros(3, dtype=float)) + normal * share

    positive_areas = [a for a in node_area.values() if a > 1.0e-30]
    fallback_area = float(np.median(positive_areas)) if positive_areas else 1.0
    node_normal: Dict[int, np.ndarray] = {}
    for nid in selected:
        if node_area.get(nid, 0.0) <= 1.0e-30:
            node_area[nid] = fallback_area
        nvec = normal_acc.get(nid, np.zeros(3, dtype=float))
        nrm = float(np.linalg.norm(nvec))
        if nrm <= 1.0e-30:
            node_normal[nid] = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            node_normal[nid] = nvec / nrm

    return {
        "surface_nodes": set(selected),
        "node_area": node_area,
        "node_normal": node_normal,
        "external_face_count": len(external_faces),
        "surface_force_node_scope": scope,
        "surface_corner_node_count": len(surface_corner_nodes),
        "surface_element_node_count": len(surface_element_nodes),
        "corner_node_count": len(corner_nodes),
        "surface_load_node_count": len(selected),
        "area_sum": float(sum(node_area.values())),
    }


def vector_component_to_records(total: Dict[int, np.ndarray], zero_tol: float) -> List[Tuple[int, int, float]]:
    records: List[Tuple[int, int, float]] = []
    for nid in sorted(total):
        vec = np.asarray(total[nid], dtype=float)
        for dof_id, value in enumerate(vec[:3], start=1):
            if abs(float(value)) > zero_tol:
                records.append((int(nid), int(dof_id), float(value)))
    return records


def build_surface_nodal_force_basis_for_mode(
    *,
    mode_number: int,
    mode_shapes: Dict[int, ModeShape],
    surface_data: dict,
    config: dict,
) -> Dict[int, np.ndarray]:
    """
    Возвращает nodal force basis f_m для одной моды.

    По умолчанию используется f_i = A_i * phi_i и нормировка
    phi_m^T f_m = 1. Тогда control value lambda_m имеет смысл
    обобщенной силы по этой моде. Для load-controlled STEP это важнее,
    чем сохранять старую интерпретацию q как заданного перемещения.
    """
    shape = mode_shapes[int(mode_number)]
    nodes: Set[int] = set(surface_data["surface_nodes"])
    node_area: Dict[int, float] = surface_data["node_area"]
    node_normal: Dict[int, np.ndarray] = surface_data["node_normal"]
    direction = str(config.get("surface_force_direction", "modal_vector")).lower()
    area_weighted = bool(config.get("surface_force_area_weighted", True))

    raw: Dict[int, np.ndarray] = {}
    for nid in sorted(nodes):
        vals = shape.node_values.get(nid)
        if vals is None:
            continue
        phi = np.array([float(vals["UX"]), float(vals["UY"]), float(vals["UZ"])], dtype=float)
        normal = np.asarray(node_normal.get(nid, np.array([0.0, 1.0, 0.0], dtype=float)), dtype=float)
        nn = float(np.linalg.norm(normal))
        if nn <= 1.0e-30:
            normal = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            normal = normal / nn

        if direction == "modal_vector":
            vec = phi
        elif direction in ("modal_normal", "surface_normal"):
            vec = float(np.dot(phi, normal)) * normal
        elif direction == "dominant_dof":
            vec = np.zeros(3, dtype=float)
            dom = str(shape.dominant_dof).upper()
            idx = {"UX": 0, "UY": 1, "UZ": 2}.get(dom, None)
            if idx is not None:
                vec[idx] = phi[idx]
        else:
            raise ValueError(f"Неизвестный surface_force_direction: {direction}")

        weight = float(node_area.get(nid, 1.0)) if area_weighted else 1.0
        raw[nid] = weight * vec

    norm_mode = str(config.get("surface_force_normalization", "modal_work_unit")).lower()
    denom = 1.0
    if norm_mode == "modal_work_unit":
        work = 0.0
        for nid, fvec in raw.items():
            vals = shape.node_values[nid]
            phi = np.array([float(vals["UX"]), float(vals["UY"]), float(vals["UZ"])], dtype=float)
            work += float(np.dot(phi, fvec))
        denom = work
    elif norm_mode == "l1_total_force":
        denom = sum(float(np.linalg.norm(v, ord=1)) for v in raw.values())
    elif norm_mode == "none":
        denom = 1.0
    else:
        raise ValueError(f"Неизвестный surface_force_normalization: {norm_mode}")

    if abs(denom) <= 1.0e-30:
        # fallback: пусть хотя бы суммарный L1 масштаб будет единичным
        denom = sum(float(np.linalg.norm(v, ord=1)) for v in raw.values())
    if abs(denom) <= 1.0e-30:
        raise RuntimeError(f"Нулевая surface force basis для mode {mode_number}.")

    return {nid: np.asarray(v, dtype=float) / float(denom) for nid, v in raw.items()}


def build_surface_nodal_force_records(
    lambda_vector: Dict[int, float],
    mode_shapes: Dict[int, ModeShape],
    surface_data: dict,
    config: dict,
    zero_tol: float,
) -> Tuple[List[Tuple[int, int, float]], dict]:
    total: Dict[int, np.ndarray] = {
        int(nid): np.zeros(3, dtype=float)
        for nid in surface_data["surface_nodes"]
    }
    basis_meta: Dict[str, dict] = {}

    for mode_number, lam in sorted(lambda_vector.items()):
        lam = float(lam)
        if abs(lam) <= 1.0e-30:
            continue
        basis = build_surface_nodal_force_basis_for_mode(
            mode_number=int(mode_number),
            mode_shapes=mode_shapes,
            surface_data=surface_data,
            config=config,
        )
        l1 = 0.0
        l2_sq = 0.0
        for nid, vec in basis.items():
            total[nid] = total.get(nid, np.zeros(3, dtype=float)) + lam * vec
            l1 += float(np.linalg.norm(vec, ord=1))
            l2_sq += float(np.dot(vec, vec))
        basis_meta[str(mode_number)] = {
            "lambda": lam,
            "basis_l1": l1,
            "basis_l2": math.sqrt(l2_sq),
            "basis_node_count": len(basis),
        }

    records = vector_component_to_records(total, zero_tol)
    abs_values = [abs(v) for _, _, v in records]
    meta = {
        "load_record_count": len(records),
        "loaded_node_count": len({nid for nid, _, _ in records}),
        "max_abs_nodal_force_component": max(abs_values) if abs_values else 0.0,
        "sum_abs_nodal_force_components": sum(abs_values),
        "basis_meta": basis_meta,
        "surface_data_summary": {
            k: (len(v) if isinstance(v, set) else v)
            for k, v in surface_data.items()
            if k not in ("node_area", "node_normal", "surface_nodes")
        },
    }
    return records, meta


def build_load_node_point_block(force_records: List[Tuple[int, int, float]], curve_id: int) -> List[str]:
    lines: List[str] = []
    for nid, dof_id, sf_value in force_records:
        lines.append("*LOAD_NODE_POINT")
        lines.append("$#    node       dof      lcid        sf       cid        m1        m2        m3")
        lines.append(
            f"{field10(nid)}{field10(dof_id)}{field10(curve_id)}{float10_sf(sf_value)}"
            f"{field10(0)}{field10(0)}{field10(0)}{field10(0)}"
        )
    return lines


def _control_vector_to_case_name(vector: Dict[int, float], value_prefix: str = "q") -> str:
    """
    Builds compact, parseable STEP case names.

    Equal-amplitude cases use the canonical compact form:
      single_m1p_q1e-8
      pair_m1p_m2m_q5e-9
      triple_m1p_m2p_m3m_q3e-9

    If amplitudes differ by mode, the name stays unambiguous:
      pair_m1p_q1e-8_m2m_q3e-8

    Exact values are always stored in step_manifest.json.
    """
    active = [(int(m), float(v)) for m, v in sorted(vector.items()) if abs(float(v)) > 1.0e-30]
    prefix = {1: "single", 2: "pair", 3: "triple"}.get(len(active), "step")
    if not active:
        return f"{prefix}_{value_prefix}0"

    abs_values = [abs(v) for _, v in active]
    ref = max(abs_values)
    same_abs = all(abs(v - abs_values[0]) <= max(1.0e-30, 1.0e-8 * ref) for v in abs_values)

    if same_abs:
        modes_part = "_".join(f"m{m}{sign_to_tag(1 if v >= 0 else -1)}" for m, v in active)
        return f"{prefix}_{modes_part}_{value_prefix}{q_to_tag(abs_values[0])}"

    parts = [prefix]
    for mode, value in active:
        parts.append(f"m{mode}{sign_to_tag(1 if value >= 0 else -1)}_{value_prefix}{q_to_tag(abs(value))}")
    return "_".join(parts)


def qvec_to_lambda_case_name(lambda_vector: Dict[int, float]) -> str:
    return _control_vector_to_case_name(lambda_vector, value_prefix="l")


def resolve_case_lambda_settings(config: dict) -> dict:
    selected_modes = [int(x) for x in config.get("selected_modes", [])]
    single = list(config.get("single_lambda_values", [1.0]))
    pair = list(config.get("pair_lambda_values", [single[0] / 2.0])) if len(selected_modes) >= 2 else []
    triple = list(config.get("triple_lambda_values", [single[0] / 3.0])) if len(selected_modes) >= 3 else []
    dual = float(config.get("dual_lambda_value", single[0]))
    return {
        "single_q_values": single,
        "pair_q_values": pair,
        "triple_q_values": triple,
        "dual_p_value": dual,
        "control_basis": "lambda",
        "q_auto_source": "user_config_lambda_values",
        "requires_fitted_fe_q": True,
    }

def resolve_prescribed_nodes_for_step(
    *,
    mode_shapes: Dict[int, ModeShape],
    model_info: dict,
    solid_elements: List[Tuple[int, List[int]]],
    excluded_nodes: Set[int],
    config: dict,
) -> Tuple[Optional[Set[int]], dict]:
    """
    Возвращает множество узлов, на которые будет наложен STEP prescribed motion.

    None означает legacy-режим: все узлы mode-файла, кроме excluded_nodes.
    Для solid по умолчанию возвращает только внешние corner nodes.
    """
    family = str(model_info.get("family", "solid")).lower()
    base_mode = next(iter(mode_shapes.values()))
    all_mode_nodes = set(base_mode.node_values.keys())

    if family != "solid":
        return None, {
            "enabled": False,
            "scope": "all_free_nodes",
            "reason": "non_solid_model",
            "all_mode_node_count": len(all_mode_nodes),
            "excluded_node_count": len(excluded_nodes),
            "prescribed_node_count": len(all_mode_nodes - excluded_nodes),
        }

    scope = str(config.get("solid_step_prescription_scope", "surface_corner_nodes")).strip().lower()
    if scope == "auto":
        scope = "surface_corner_nodes"

    if scope == "all_free_nodes":
        return None, {
            "enabled": False,
            "scope": "all_free_nodes",
            "reason": "legacy_all_free_nodes_requested",
            "all_mode_node_count": len(all_mode_nodes),
            "excluded_node_count": len(excluded_nodes),
            "prescribed_node_count": len(all_mode_nodes - excluded_nodes),
        }

    if not solid_elements:
        if bool(config.get("solid_fail_if_no_surface_nodes", True)):
            raise RuntimeError(
                "solid_step_prescription_scope требует *ELEMENT_SOLID, но solid_elements не распознаны."
            )
        return None, {
            "enabled": False,
            "scope": "all_free_nodes",
            "reason": "no_solid_elements_fallback",
            "all_mode_node_count": len(all_mode_nodes),
            "excluded_node_count": len(excluded_nodes),
            "prescribed_node_count": len(all_mode_nodes - excluded_nodes),
        }

    corner_nodes = collect_solid_corner_nodes(solid_elements)
    surface_corner_nodes = collect_solid_surface_corner_nodes(solid_elements)
    surface_element_nodes = collect_solid_surface_element_nodes(solid_elements)

    if scope == "corner_nodes":
        selected = corner_nodes
    elif scope == "surface_nodes":
        selected = surface_element_nodes
        if not bool(config.get("solid_prescribe_midside_nodes", False)):
            selected = selected & corner_nodes
            scope = "surface_nodes_corner_only"
    elif scope == "surface_corner_nodes":
        selected = surface_corner_nodes
    else:
        raise ValueError(f"Неизвестный solid_step_prescription_scope: {scope}")

    selected = set(int(n) for n in selected)
    selected_in_modes = selected & all_mode_nodes
    selected_free = selected_in_modes - set(excluded_nodes)

    if not selected_free and bool(config.get("solid_fail_if_no_surface_nodes", True)):
        raise RuntimeError(
            "После выбора solid surface/corner nodes и исключения SPC-узлов не осталось prescribed nodes.\n"
            f"scope={scope}\n"
            f"surface_corner_node_count={len(surface_corner_nodes)}\n"
            f"corner_node_count={len(corner_nodes)}\n"
            f"excluded_node_count={len(excluded_nodes)}"
        )

    info = {
        "enabled": True,
        "scope": scope,
        "all_mode_node_count": len(all_mode_nodes),
        "excluded_node_count": len(excluded_nodes),
        "solid_element_count": len(solid_elements),
        "solid_corner_node_count": len(corner_nodes),
        "solid_surface_corner_node_count": len(surface_corner_nodes),
        "solid_surface_element_node_count": len(surface_element_nodes),
        "selected_node_count_before_mode_filter": len(selected),
        "selected_node_count_in_modes": len(selected_in_modes),
        "prescribed_node_count": len(selected_free),
        "internal_nodes_prescribed": False,
        "midside_nodes_prescribed": bool(config.get("solid_prescribe_midside_nodes", False)),
    }
    return selected_free, info


def element_tet_decomposition(corners: List[int]) -> List[Tuple[int, int, int, int]]:
    if len(corners) >= 8:
        n = corners[:8]
        return [
            (n[0], n[1], n[3], n[4]),
            (n[1], n[2], n[3], n[6]),
            (n[1], n[3], n[4], n[6]),
            (n[1], n[4], n[5], n[6]),
            (n[3], n[4], n[6], n[7]),
        ]
    if len(corners) >= 6:
        n = corners[:6]
        return [(n[0], n[1], n[2], n[3]), (n[1], n[2], n[4], n[5]), (n[1], n[2], n[3], n[5])]
    if len(corners) >= 4:
        n = corners[:4]
        return [(n[0], n[1], n[2], n[3])]
    return []


def evaluate_step_preflight(
    *,
    q_vector: Dict[int, float],
    mode_shapes: Dict[int, ModeShape],
    prescribed_dofs: List[str],
    excluded_nodes: Set[int],
    solid_elements: List[Tuple[int, List[int]]],
    model_info: dict,
    config: dict,
    prescribed_nodes: Optional[Set[int]] = None,
) -> dict:
    coords0 = mode_node_coordinates(mode_shapes)
    disp = nodal_displacement_field(q_vector, mode_shapes, prescribed_dofs, excluded_nodes, prescribed_nodes)
    h = max(float(model_info.get("thickness_estimate", 1.0)), 1.0e-30)

    max_disp = 0.0
    max_comp = 0.0
    for nid, d in disp.items():
        v = np.array([d["UX"], d["UY"], d["UZ"]], dtype=float)
        max_disp = max(max_disp, float(np.linalg.norm(v)))
        max_comp = max(max_comp, float(np.max(np.abs(v)))) if v.size else max_comp

    min_edge = float("inf")
    min_vol_ratio = float("inf")
    min_signed_ratio = float("inf")
    bad_volume_count = 0
    checked_elements = 0
    max_elements = int(config.get("preflight_max_sample_elements", 200000))

    for eid, nodes in solid_elements[:max_elements]:
        corners = element_corner_nodes(nodes)
        if any(nid not in coords0 for nid in corners):
            continue
        checked_elements += 1
        pts0 = {nid: coords0[nid] for nid in corners}
        pts1 = {
            nid: coords0[nid] + np.array([disp[nid]["UX"], disp[nid]["UY"], disp[nid]["UZ"]], dtype=float)
            for nid in corners
        }
        # edge metric
        for a_i in range(len(corners)):
            for b_i in range(a_i + 1, len(corners)):
                e = float(np.linalg.norm(pts0[corners[b_i]] - pts0[corners[a_i]]))
                if e > 1.0e-30:
                    min_edge = min(min_edge, e)

        for tet in element_tet_decomposition(corners):
            try:
                v0 = tet_signed_volume(pts0[tet[0]], pts0[tet[1]], pts0[tet[2]], pts0[tet[3]])
                v1 = tet_signed_volume(pts1[tet[0]], pts1[tet[1]], pts1[tet[2]], pts1[tet[3]])
            except Exception:
                continue
            if abs(v0) <= 1.0e-30:
                continue
            ratio_abs = abs(v1) / abs(v0)
            ratio_signed = v1 / v0
            min_vol_ratio = min(min_vol_ratio, ratio_abs)
            min_signed_ratio = min(min_signed_ratio, ratio_signed)
            if ratio_signed <= 0.0:
                bad_volume_count += 1

    if not math.isfinite(min_edge):
        min_edge = float("nan")
    if not math.isfinite(min_vol_ratio):
        min_vol_ratio = float("nan")
    if not math.isfinite(min_signed_ratio):
        min_signed_ratio = float("nan")

    max_motion_over_min_edge = max_disp / min_edge if math.isfinite(min_edge) and min_edge > 0 else float("nan")
    max_disp_over_h = max_disp / h

    reasons: List[str] = []
    if max_disp_over_h > float(config.get("preflight_max_disp_over_thickness", 0.08)):
        reasons.append(f"max_disp_over_h>{float(config.get('preflight_max_disp_over_thickness', 0.08)):.3e}")
    if math.isfinite(max_motion_over_min_edge) and max_motion_over_min_edge > float(config.get("preflight_max_motion_over_min_edge", 0.20)):
        reasons.append(f"max_motion_over_min_edge>{float(config.get('preflight_max_motion_over_min_edge', 0.20)):.3e}")
    if math.isfinite(min_vol_ratio) and min_vol_ratio < float(config.get("preflight_min_tet_volume_ratio", 0.05)):
        reasons.append(f"min_volume_ratio<{float(config.get('preflight_min_tet_volume_ratio', 0.05)):.3e}")
    if bad_volume_count > 0:
        reasons.append("signed_volume_flip")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "max_disp": max_disp,
        "max_component_disp": max_comp,
        "max_disp_over_thickness": max_disp_over_h,
        "min_edge_length": min_edge,
        "max_motion_over_min_edge": max_motion_over_min_edge,
        "checked_solid_elements": checked_elements,
        "min_abs_volume_ratio": min_vol_ratio,
        "min_signed_volume_ratio": min_signed_ratio,
        "bad_volume_count": bad_volume_count,
    }


def adapt_q_vector_by_preflight(
    *,
    q_vector: Dict[int, float],
    mode_shapes: Dict[int, ModeShape],
    prescribed_dofs: List[str],
    excluded_nodes: Set[int],
    solid_elements: List[Tuple[int, List[int]]],
    model_info: dict,
    config: dict,
    prescribed_nodes: Optional[Set[int]] = None,
) -> Tuple[Dict[int, float], dict]:
    family = str(model_info.get("family", "solid")).lower()
    if family != "solid" or not bool(config.get("enable_solid_step_preflight", True)) or not solid_elements:
        return dict(q_vector), {"enabled": False, "ok": True, "scale": 1.0, "attempts": 0}

    scale = 1.0
    shrink = float(config.get("preflight_shrink_factor", 0.5))
    max_steps = int(config.get("preflight_max_shrink_steps", 12))
    last_metrics = None
    for attempt in range(max_steps + 1):
        candidate = {int(m): float(q) * scale for m, q in q_vector.items()}
        metrics = evaluate_step_preflight(
            q_vector=candidate,
            mode_shapes=mode_shapes,
            prescribed_dofs=prescribed_dofs,
            excluded_nodes=excluded_nodes,
            solid_elements=solid_elements,
            model_info=model_info,
            config=config,
            prescribed_nodes=prescribed_nodes,
        )
        metrics.update({"enabled": True, "scale": scale, "attempts": attempt})
        last_metrics = metrics
        if metrics["ok"]:
            return candidate, metrics
        if not bool(config.get("preflight_allow_auto_shrink", True)):
            break
        scale *= shrink

    if bool(config.get("preflight_reject_on_bad_geometry", True)):
        raise RuntimeError(
            "STEP-поле отвергнуто preflight-проверкой геометрии solid-элементов.\n"
            f"q_vector={q_vector}\n"
            f"last_metrics={json.dumps(last_metrics, ensure_ascii=False, indent=2)}"
        )
    return {int(m): float(q) * scale for m, q in q_vector.items()}, last_metrics or {"enabled": True, "ok": False}


def qvec_to_case_name(q_vector: Dict[int, float]) -> str:
    return _control_vector_to_case_name(q_vector, value_prefix="q")


def build_step_case_defs(selected_modes: List[int], q_settings: dict, config: dict) -> List[Tuple[str, Dict[int, float]]]:
    """Создает классические STEP-поля. Для solid поддерживает индивидуальные q_r."""
    q_by_mode_raw = q_settings.get("q_values_by_mode")
    if isinstance(q_by_mode_raw, dict):
        q_by_mode = {int(k): float(v) for k, v in q_by_mode_raw.items()}
        case_defs: List[Tuple[str, Dict[int, float]]] = []
        single_scales = [1.0]
        if bool(config.get("include_single_q_values_for_all_interaction_q", True)):
            if len(selected_modes) >= 2:
                single_scales.append(_interaction_q_factor(2, config))
            if len(selected_modes) >= 3:
                single_scales.append(_interaction_q_factor(3, config))
        # single +/- for each mode and all needed interaction scales
        seen_single = set()
        for mode in selected_modes:
            q0 = q_by_mode.get(int(mode))
            if q0 is None:
                raise ValueError(f"Нет q_values_by_mode для mode {mode}")
            for sc in single_scales:
                q = q0 * sc
                key = (mode, round(q, 16))
                if key in seen_single:
                    continue
                seen_single.add(key)
                for sign in (-1, 1):
                    qv = {int(mode): sign * q}
                    case_defs.append((qvec_to_case_name(qv), qv))
        # pair +/- combinations with per-mode amplitudes
        if len(selected_modes) >= 2:
            pair_scale = _interaction_q_factor(2, config)
            for mode_a, mode_b in itertools.combinations(selected_modes, 2):
                for signs in itertools.product((-1, 1), repeat=2):
                    qv = {
                        int(mode_a): signs[0] * q_by_mode[int(mode_a)] * pair_scale,
                        int(mode_b): signs[1] * q_by_mode[int(mode_b)] * pair_scale,
                    }
                    case_defs.append((qvec_to_case_name(qv), qv))
        # triple +/- combinations with per-mode amplitudes
        if len(selected_modes) >= 3:
            triple_scale = _interaction_q_factor(3, config)
            for mode_a, mode_b, mode_c in itertools.combinations(selected_modes, 3):
                for signs in itertools.product((-1, 1), repeat=3):
                    qv = {
                        int(mode_a): signs[0] * q_by_mode[int(mode_a)] * triple_scale,
                        int(mode_b): signs[1] * q_by_mode[int(mode_b)] * triple_scale,
                        int(mode_c): signs[2] * q_by_mode[int(mode_c)] * triple_scale,
                    }
                    case_defs.append((qvec_to_case_name(qv), qv))
        return case_defs

    # legacy scalar-q path
    case_defs = []
    case_defs.extend(build_single_cases(selected_modes, q_settings["single_q_values"]))
    if len(selected_modes) >= 2:
        case_defs.extend(build_pair_cases(selected_modes, q_settings["pair_q_values"]))
    if len(selected_modes) >= 3:
        case_defs.extend(build_triple_cases(selected_modes, q_settings["triple_q_values"]))
    return case_defs


def build_dual_case_defs(selected_modes: List[int], q_settings: dict, config: dict) -> List[Tuple[str, Dict[int, float]]]:
    q_by_mode_raw = q_settings.get("dual_p_values_by_mode") or q_settings.get("q_values_by_mode")
    if isinstance(q_by_mode_raw, dict):
        out: List[Tuple[str, Dict[int, float]]] = []
        dual_factor = float(config.get("dual_q_factor", 1.0))
        for mode in selected_modes:
            p = float(q_by_mode_raw[str(mode)] if str(mode) in q_by_mode_raw else q_by_mode_raw[int(mode)]) * dual_factor
            out.append((f"dual_mode{mode}_p_q{q_to_tag(p)}", {int(mode): p}))
            out.append((f"dual_mode{mode}_2p_q{q_to_tag(2.0 * p)}", {int(mode): 2.0 * p}))
        return out
    return build_dual_cases(selected_modes, q_settings["dual_p_value"])


def build_single_cases(selected_modes, q_values):
    cases = []
    for mode in selected_modes:
        for q in q_values:
            for sign in (-1, 1):
                cases.append((f"single_m{mode}{sign_to_tag(sign)}_q{q_to_tag(q)}", {mode: sign * q}))
    return cases


def build_pair_cases(selected_modes, q_values):
    cases = []
    for mode_a, mode_b in itertools.combinations(selected_modes, 2):
        for q in q_values:
            for signs in itertools.product((-1, 1), repeat=2):
                qvec = {mode_a: signs[0] * q, mode_b: signs[1] * q}
                name = f"pair_m{mode_a}{sign_to_tag(signs[0])}_m{mode_b}{sign_to_tag(signs[1])}_q{q_to_tag(q)}"
                cases.append((name, qvec))
    return cases


def build_triple_cases(selected_modes, q_values):
    cases = []
    for mode_a, mode_b, mode_c in itertools.combinations(selected_modes, 3):
        for q in q_values:
            for signs in itertools.product((-1, 1), repeat=3):
                qvec = {mode_a: signs[0] * q, mode_b: signs[1] * q, mode_c: signs[2] * q}
                name = f"triple_m{mode_a}{sign_to_tag(signs[0])}_m{mode_b}{sign_to_tag(signs[1])}_m{mode_c}{sign_to_tag(signs[2])}_q{q_to_tag(q)}"
                cases.append((name, qvec))
    return cases


def build_dual_cases(selected_modes, dual_p_value):
    cases = []
    for mode in selected_modes:
        cases.append((f"dual_mode{mode}_p", {mode: 1.0 * dual_p_value}))
        cases.append((f"dual_mode{mode}_2p", {mode: 2.0 * dual_p_value}))
    return cases


def mode_normal_content_ratio(shape: ModeShape, normal_dof: str) -> float:
    transl_max = max(
        max(abs(vals[dof]) for vals in shape.node_values.values())
        for dof in ("UX", "UY", "UZ")
    )
    normal_max = max(abs(vals[normal_dof]) for vals in shape.node_values.values())
    return float(normal_max / max(transl_max, 1.0e-30))


def validate_primary_step_setup(
    *,
    case_dir: Path,
    config: dict,
    model_info: dict,
    prescribed_dofs: List[str],
    mode_shapes: Dict[int, ModeShape],
    selected_modes: List[int],
) -> Dict[str, object]:
    normal_axis = str(model_info.get("normal_axis_geometry", "Z")).upper()
    normal_dof = "U" + normal_axis
    family = str(model_info.get("family", "solid")).lower()

    ratios = {
        str(m): mode_normal_content_ratio(mode_shapes[m], normal_dof)
        for m in selected_modes
    }

    if family == "solid" and bool(config.get("strict_solid_normal_step_guard", True)):
        mode = str(config.get("primary_step_prescribed_mode", "auto")).lower()
        if mode == "normal_only":
            if prescribed_dofs != [normal_dof]:
                raise RuntimeError(
                    f"{case_dir.name}: solid STEP в режиме normal_only должен задаваться только по {normal_dof}, "
                    f"но получено prescribed_dofs={prescribed_dofs}."
                )
        else:
            # Для объемной постановки не допускаем ROT*-DOF и требуем, чтобы
            # нормальный DOF входил в заданное модальное поле.
            if any(str(d).upper().startswith("ROT") for d in prescribed_dofs):
                raise RuntimeError(
                    f"{case_dir.name}: solid STEP не должен содержать вращательные DOF: {prescribed_dofs}."
                )
            if normal_dof not in prescribed_dofs:
                raise RuntimeError(
                    f"{case_dir.name}: solid STEP не содержит нормальный DOF {normal_dof}: {prescribed_dofs}."
                )

    min_ratio = float(config.get("min_normal_content_ratio_for_selected_modes", 1.0e-6))
    low = {m: r for m, r in ratios.items() if r < min_ratio}
    if low and not bool(config.get("allow_low_normal_content_modes", False)):
        raise RuntimeError(
            f"{case_dir.name}: среди selected_modes есть формы почти без нормальной компоненты {normal_dof}: {low}. "
            "Для transverse STEP/ROM выбирайте изгибные формы с заметной нормальной компонентой "
            "или включите allow_low_normal_content_modes=true только если это осознанно."
        )

    return {
        "normal_axis": normal_axis,
        "normal_dof": normal_dof,
        "normal_content_ratio_by_mode": ratios,
        "strict_solid_normal_step_guard": bool(config.get("strict_solid_normal_step_guard", True)),
    }


def generate_step_cases_for_bc(case_dir: Path, config: dict) -> List[Path]:
    modal_dir = case_dir / config["modal_dir_name"]
    step_dir = case_dir / config["step_dir_name"]
    base_key = case_dir / config["base_key_name"]
    if not modal_dir.exists():
        raise FileNotFoundError(f"Не найдена папка мод: {modal_dir}")
    if not base_key.exists():
        raise FileNotFoundError(f"Не найден base key: {base_key}")

    discovered = discover_mode_shapes(modal_dir, config)
    if not discovered:
        raise FileNotFoundError(f"В папке {modal_dir} не найдены mode-файлы.")
    selected_modes = [m for m in config["selected_modes"] if m in discovered]
    if not selected_modes:
        raise ValueError(f"Для случая {case_dir.name} не найдено ни одной выбранной моды из {config['selected_modes']}.")

    mode_shapes = {m: discovered[m] for m in selected_modes}
    base_key_lines = load_base_key_lines(base_key)
    excluded_nodes = collect_excluded_nodes(base_key_lines, config)
    solid_elements = parse_solid_elements_from_key(base_key_lines)

    model_info = detect_model_info(base_key_lines, mode_shapes, config)
    prescribed_dofs = resolve_case_prescribed_dofs(config, model_info)
    setup_validation = validate_primary_step_setup(
        case_dir=case_dir,
        config=config,
        model_info=model_info,
        prescribed_dofs=prescribed_dofs,
        mode_shapes=mode_shapes,
        selected_modes=selected_modes,
    )
    step_load_scheme = str(config.get("step_load_scheme", "surface_nodal_force_step")).lower()
    if step_load_scheme == "surface_nodal_force_step":
        q_settings = resolve_case_lambda_settings(config)
    else:
        q_settings = resolve_case_q_settings(case_dir, config, model_info)

    prescribed_nodes, prescribed_node_info = resolve_prescribed_nodes_for_step(
        mode_shapes=mode_shapes,
        model_info=model_info,
        solid_elements=solid_elements,
        excluded_nodes=excluded_nodes,
        config=config,
    )

    surface_load_data = None
    if step_load_scheme == "surface_nodal_force_step":
        surface_load_data = build_surface_load_data(
            mode_shapes=mode_shapes,
            solid_elements=solid_elements,
            excluded_nodes=excluded_nodes,
            config=config,
        )

    if config["report_excluded_nodes"]:
        print(f"[INFO] {case_dir.name}: model_family = {model_info['family']}")
        print(f"[INFO] {case_dir.name}: geometry_spans = {model_info['geometry_spans']}")
        print(f"[INFO] {case_dir.name}: thickness_estimate = {model_info['thickness_estimate']:.6g}")
        print(f"[INFO] {case_dir.name}: normal_axis = {setup_validation['normal_axis']}")
        print(f"[INFO] {case_dir.name}: normal_dof = {setup_validation['normal_dof']}")
        print(f"[INFO] {case_dir.name}: prescribed_dofs = {prescribed_dofs}")
        print(f"[INFO] {case_dir.name}: normal_content_ratio_by_mode = {setup_validation['normal_content_ratio_by_mode']}")
        print(f"[INFO] {case_dir.name}: step_load_scheme = {step_load_scheme}")
        if step_load_scheme == "surface_nodal_force_step":
            print(f"[INFO] {case_dir.name}: lambda single/pair/triple/dual = {q_settings['single_q_values']} / {q_settings['pair_q_values']} / {q_settings['triple_q_values']} / {q_settings['dual_p_value']}")
            if isinstance(surface_load_data, dict):
                print(f"[INFO] {case_dir.name}: surface load nodes = {len(surface_load_data.get('surface_nodes', []))}; external faces = {surface_load_data.get('external_face_count')}; area_sum = {surface_load_data.get('area_sum'):.6g}")
        elif isinstance(q_settings.get("q_values_by_mode"), dict):
            print(f"[INFO] {case_dir.name}: q_values_by_mode = {q_settings['q_values_by_mode']}")
            print(f"[INFO] {case_dir.name}: pair/triple scale = {q_settings.get('pair_scale')} / {q_settings.get('triple_scale')}")
        else:
            print(f"[INFO] {case_dir.name}: q single/pair/triple/dual = {q_settings['single_q_values']} / {q_settings['pair_q_values']} / {q_settings['triple_q_values']} / {q_settings['dual_p_value']}")
        print(f"[INFO] {case_dir.name}: excluded support nodes = {len(excluded_nodes)}")
        print(f"[INFO] {case_dir.name}: parsed solid elements = {len(solid_elements)}")
        print(f"[INFO] {case_dir.name}: solid_step_prescription_scope = {prescribed_node_info.get('scope')}")
        print(f"[INFO] {case_dir.name}: prescribed STEP nodes = {prescribed_node_info.get('prescribed_node_count')}")
        if config["exclude_node_set_ids"]:
            print(f"[INFO] {case_dir.name}: exclude_node_set_ids = {config['exclude_node_set_ids']}")
        if config["exclude_all_set_node_lists"]:
            print(f"[INFO] {case_dir.name}: using ALL *SET_NODE_LIST cards")

    step_dir.mkdir(parents=True, exist_ok=True)

    step_case_defs = build_step_case_defs(selected_modes, q_settings, config)
    dual_case_defs = build_dual_case_defs(selected_modes, q_settings, config) if config["generate_dual_mode_decks"] else []

    generated_files: List[Path] = []
    manifest = {
        "case": case_dir.name,
        "method": step_load_scheme,
        "note": (
            "surface_nodal_force_step is load-controlled. It does not prescribe q. "
            "4_Generate_ROM must reconstruct q_fe from each STEP d3plot/final FE state and "
            "must use lambda_vector/applied_load as the generalized force."
            if step_load_scheme == "surface_nodal_force_step" else
            "classic prescribed-motion STEP; q_vector is the commanded displacement vector."
        ),
        "selected_modes": selected_modes,
        "model_info": model_info,
        "prescribed_dofs": prescribed_dofs,
        "setup_validation": setup_validation,
        "primary_step_prescribed_mode_resolved": config.get("primary_step_prescribed_mode", "auto"),
        "step_load_scheme": step_load_scheme,
        "step_control_basis": "lambda" if step_load_scheme == "surface_nodal_force_step" else "q",
        "requires_fitted_fe_q": bool(config.get("step_requires_fitted_fe_q", step_load_scheme != "prescribed_motion_step")),
        "expected_reaction_source": str(config.get("step_expected_reaction_source", "applied_load_manifest")),
        "q_settings": q_settings,
        "excluded_node_count": len(excluded_nodes),
        "solid_element_count": len(solid_elements),
        "prescribed_node_info": prescribed_node_info,
        "surface_load_info": (
            {
                "surface_force_direction": config.get("surface_force_direction"),
                "surface_force_node_scope": config.get("surface_force_node_scope"),
                "surface_force_area_weighted": config.get("surface_force_area_weighted"),
                "surface_force_normalization": config.get("surface_force_normalization"),
                "surface_load_node_count": len(surface_load_data.get("surface_nodes", [])) if isinstance(surface_load_data, dict) else 0,
                "external_face_count": surface_load_data.get("external_face_count") if isinstance(surface_load_data, dict) else 0,
                "area_sum": surface_load_data.get("area_sum") if isinstance(surface_load_data, dict) else None,
            }
            if step_load_scheme == "surface_nodal_force_step" else None
        ),
        "cases": [],
        "dual_cases": [],
    }

    # STEP single/pair/triple decks
    for raw_case_name, raw_control_vector in step_case_defs:
        if step_load_scheme == "surface_nodal_force_step":
            if surface_load_data is None:
                raise RuntimeError("surface_nodal_force_step выбран, но surface_load_data не построен.")
            lambda_vector = {int(k): float(v) for k, v in raw_control_vector.items()}
            case_name = qvec_to_lambda_case_name(lambda_vector)
            force_records, force_meta = build_surface_nodal_force_records(
                lambda_vector=lambda_vector,
                mode_shapes=mode_shapes,
                surface_data=surface_load_data,
                config=config,
                zero_tol=float(config["zero_tol"]),
            )
            if not force_records:
                continue

            step_block = []
            step_block.extend(build_define_curve_block(config["curve_id"], config["end_time"]))
            step_block.extend(build_load_node_point_block(force_records, config["curve_id"]))

            entry_common = {
                "name": case_name,
                "raw_name": raw_case_name,
                "q_vector": None,
                "lambda_vector": {str(k): float(v) for k, v in sorted(lambda_vector.items())},
                "control_vector": {str(k): float(v) for k, v in sorted(lambda_vector.items())},
                "control_basis": "lambda",
                "requires_fitted_fe_q": True,
                "expected_reaction_source": str(config.get("step_expected_reaction_source", "applied_load_manifest")),
                "basis_type": "surface_node_force",
                "basis_norm": str(config.get("surface_force_normalization", "modal_work_unit")),
                "surface_force_direction": str(config.get("surface_force_direction", "modal_vector")),
                "surface_force_node_scope": str(config.get("surface_force_node_scope", "surface_corner_nodes")),
                "load_record_count": len(force_records),
                "load_meta": force_meta,
                "preflight": {"enabled": False, "reason": "load_controlled_no_prescribed_displacement"},
            }
        else:
            raw_q_vector = {int(k): float(v) for k, v in raw_control_vector.items()}
            q_vector, preflight = adapt_q_vector_by_preflight(
                q_vector=raw_q_vector,
                mode_shapes=mode_shapes,
                prescribed_dofs=prescribed_dofs,
                excluded_nodes=excluded_nodes,
                solid_elements=solid_elements,
                model_info=model_info,
                config=config,
                prescribed_nodes=prescribed_nodes,
            )
            case_name = qvec_to_case_name(q_vector)
            motion_records = build_motion_records(
                q_vector,
                mode_shapes,
                prescribed_dofs,
                config["zero_tol"],
                excluded_nodes,
                prescribed_nodes=prescribed_nodes,
            )
            if not motion_records:
                continue

            step_block = []
            step_block.extend(build_define_curve_block(config["curve_id"], config["end_time"]))
            step_block.extend(build_boundary_prescribed_motion_node_block(motion_records, config["curve_id"]))

            entry_common = {
                "name": case_name,
                "raw_name": raw_case_name,
                "q_vector": {str(k): float(v) for k, v in sorted(q_vector.items())},
                "raw_q_vector": {str(k): float(v) for k, v in sorted(raw_q_vector.items())},
                "control_basis": "q",
                "requires_fitted_fe_q": False,
                "expected_reaction_source": "bndout",
                "preflight": preflight,
                "prescribed_dofs": prescribed_dofs,
                "motion_record_count": len(motion_records),
                "prescribed_node_info": prescribed_node_info,
            }

        if config["generate_linear_and_nonlinear_decks"]:
            base_nl = apply_implicit_patch(base_key_lines, config, "nl")
            out_path_nl = step_dir / f"{case_name}{config['output_suffix_nl']}"
            out_path_nl.write_text("\n".join(insert_before_end(base_nl, step_block)) + "\n", encoding="utf-8")
            generated_files.append(out_path_nl)
            manifest["cases"].append(dict(entry_common, branch="nl", file=str(out_path_nl.name)))

            base_lin = apply_implicit_patch(base_key_lines, config, "lin")
            out_path_lin = step_dir / f"{case_name}{config['output_suffix_lin']}"
            out_path_lin.write_text("\n".join(insert_before_end(base_lin, step_block)) + "\n", encoding="utf-8")
            generated_files.append(out_path_lin)
            manifest["cases"].append(dict(entry_common, branch="lin", file=str(out_path_lin.name)))
        else:
            base_one = apply_implicit_patch(base_key_lines, config, "nl")
            out_path = step_dir / f"{case_name}{config['output_suffix']}"
            out_path.write_text("\n".join(insert_before_end(base_one, step_block)) + "\n", encoding="utf-8")
            generated_files.append(out_path)
            manifest["cases"].append(dict(entry_common, branch="nl", file=str(out_path.name)))

    dual_prescribed_mode_for_case = config["dual_prescribed_mode"]
    dual_prescribed_dofs_for_case = config["dual_prescribed_dofs"]
    if dual_prescribed_mode_for_case == "normal_only":
        dual_prescribed_mode_for_case = "prescribed_dofs"
        dual_prescribed_dofs_for_case = [setup_validation["normal_dof"]]

    for case_name, q_vector in dual_case_defs:
        motion_records = build_dual_motion_records(
            q_vector,
            mode_shapes,
            config["zero_tol"],
            excluded_nodes,
            prescribed_nodes=prescribed_nodes,
            dual_prescribed_mode=dual_prescribed_mode_for_case,
            dual_prescribed_dofs=dual_prescribed_dofs_for_case,
        )
        if not motion_records:
            continue

        dual_block = []
        dual_block.extend(build_define_curve_block(config["curve_id"], config["end_time"]))
        dual_block.extend(build_boundary_prescribed_motion_node_block(motion_records, config["curve_id"]))

        base_dual = apply_implicit_patch(base_key_lines, config, "nl")
        out_path_dual = step_dir / f"{case_name}{config['dual_output_suffix']}"
        out_path_dual.write_text("\n".join(insert_before_end(base_dual, dual_block)) + "\n", encoding="utf-8")
        generated_files.append(out_path_dual)
        manifest["dual_cases"].append({
            "name": case_name,
            "file": str(out_path_dual.name),
            "q_vector": {str(k): float(v) for k, v in sorted(q_vector.items())},
            "dual_prescribed_mode": dual_prescribed_mode_for_case,
            "dual_prescribed_dofs": dual_prescribed_dofs_for_case,
            "motion_record_count": len(motion_records),
            "prescribed_node_info": prescribed_node_info,
        })

    (step_dir / "auto_step_generation_info.json").write_text(
        json.dumps({
            "case": case_dir.name,
            "selected_modes": selected_modes,
            "model_info": model_info,
            "prescribed_dofs": prescribed_dofs,
            "setup_validation": setup_validation,
            "primary_step_prescribed_mode_resolved": config.get("primary_step_prescribed_mode", "auto"),
            "step_load_scheme": step_load_scheme,
            "step_control_basis": "lambda" if step_load_scheme == "surface_nodal_force_step" else "q",
            "requires_fitted_fe_q": bool(config.get("step_requires_fitted_fe_q", step_load_scheme != "prescribed_motion_step")),
            "q_settings": q_settings,
            "surface_load_info": manifest.get("surface_load_info"),
            "excluded_node_count": len(excluded_nodes),
            "solid_element_count": len(solid_elements),
            "solid_element_parser": "two_line_workbench_supported",
            "prescribed_node_info": prescribed_node_info,
            "manifest": str(config.get("step_manifest_filename", "step_manifest.json")),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if bool(config.get("write_step_manifest", True)):
        (step_dir / str(config.get("step_manifest_filename", "step_manifest.json"))).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return generated_files


# ---------------------------------------------------------------------
# Запуск real_load FE внутри 2-го скрипта для калибровки q
# ---------------------------------------------------------------------

def run_generate_k_files_stage(config: dict, base_dir: Path | None = None) -> None:
    config = normalize_config(config)
    base_dir = Path(base_dir) if base_dir is not None else script_dir()
    exports_root = Path(config["exports_root"])
    if not exports_root.is_absolute():
        exports_root = (base_dir / exports_root).resolve()
    if not exports_root.exists():
        raise FileNotFoundError(f"Не найдена папка exports_root: {exports_root}")
    case_dirs = sorted([p for p in exports_root.glob(config["case_glob"]) if p.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f"В папке {exports_root} не найдены случаи по маске {config['case_glob']}")

    all_generated = []
    print(f"[INFO] exports_root = {exports_root}")
    print(f"[INFO] cases found  = {len(case_dirs)}")
    print(f"[INFO] mode         = {'lin+nl' if config['generate_linear_and_nonlinear_decks'] else 'single-family'}")
    print(f"[INFO] base key      = {config['base_key_name']}")
    print(f"[INFO] patch implicit= {config['patch_implicit_cards']}")
    print(f"[INFO] dual decks    = {config['generate_dual_mode_decks']}")
    if config["generate_dual_mode_decks"]:
        print(f"[INFO] dual p value  = {config['dual_p_value']}")
        print(f"[INFO] dual mode     = {config['dual_prescribed_mode']}")
        if config["dual_prescribed_mode"] == "prescribed_dofs":
            print(f"[INFO] dual dofs     = {config['dual_prescribed_dofs']}")

    for case_dir in case_dirs:
        print(f"[INFO] Processing {case_dir.name} ...")
        generated = generate_step_cases_for_bc(case_dir, config)
        all_generated.extend(generated)
        print(f"[INFO] Generated {len(generated)} STEP decks in {case_dir / config['step_dir_name']}")

    print(f"[INFO] Total generated files: {len(all_generated)}")
    for path in all_generated:
        print(path)
