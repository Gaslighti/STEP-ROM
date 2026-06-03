
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np



REACTION_FILE_RE = re.compile(r"^(single|pair|triple)_(.+?)_(lin|nl)\.txt$", re.IGNORECASE)
MODE_TOKEN_RE = re.compile(r"m(?P<mode>\d+)(?P<sign>[pm])", re.IGNORECASE)


@dataclass
class ModeData:
    mode_number: int
    freq_hz: float
    node_values: Dict[int, Dict[str, float]]
    ref_node: Optional[int]
    dom_dof: Optional[str]
    normalization_value: float


@dataclass
class DualModeData:
    source_mode_number: int
    p_value: float
    raw_norm: float
    coeff: float
    node_values: Dict[int, Dict[str, float]]


@dataclass
class ReactionSnapshot:
    time_value: float
    node_values: Dict[int, Dict[str, float]]


@dataclass
class StepExperiment:
    family: str
    name_base: str
    q_vector: np.ndarray
    q_abs: float
    sign_code: Tuple[int, ...]
    lin_file: Optional[Path] = None
    nl_file: Optional[Path] = None
    lin_force: Optional[np.ndarray] = None
    nl_force: Optional[np.ndarray] = None
    gamma_force: Optional[np.ndarray] = None


@dataclass
class FEState:
    index: int
    time_value: float
    load_pct: float
    field: Dict[int, Dict[str, float]]
    w_value: float


def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def deep_update(base: dict, new: dict) -> dict:
    result = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден config: {config_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(cfg, dict):
        raise ValueError(f"Файл конфигурации должен содержать JSON-объект: {config_path}")

    return cfg


def fmt(x: float) -> str:
    return f"{float(x):.12e}"


def translational_dofs() -> List[str]:
    return ["UX", "UY", "UZ"]


def rotational_dofs() -> List[str]:
    return ["ROTX", "ROTY", "ROTZ"]


def all_dofs(use_rotations: bool) -> List[str]:
    return translational_dofs() + (rotational_dofs() if use_rotations else [])


def estimate_modal_normal_dof(
    modes: Dict[int, ModeData],
    selected_modes: Sequence[int],
    reference_modes: Optional[Sequence[int]] = None,
) -> Tuple[str, Dict[str, int], Dict[str, float]]:
    dofs = ["UX", "UY", "UZ"]
    votes = {d: 0 for d in dofs}
    scores = {d: 0.0 for d in dofs}

    if reference_modes is None:
        ref_list = list(selected_modes)
    else:
        ref_list = [m for m in reference_modes if m in modes]
        if not ref_list:
            ref_list = list(selected_modes)

    for mode_num in ref_list:
        md = modes[mode_num]
        local = {
            dof: max(abs(vals[dof]) for vals in md.node_values.values())
            for dof in dofs
        }
        best = max(local, key=local.get)
        votes[best] += 1
        for dof in dofs:
            scores[dof] += float(local[dof])

    best_vote = max(votes.values())
    cands = [d for d in dofs if votes[d] == best_vote]
    if len(cands) == 1:
        return cands[0], votes, scores

    best = max(cands, key=lambda d: scores[d])
    return best, votes, scores


def estimate_geometry_normal_dof(
    modes: Dict[int, ModeData],
    selected_modes: Sequence[int],
) -> Tuple[str, Dict[str, float], float]:
    first_mode = modes[list(selected_modes)[0]]
    nodes = sorted(first_mode.node_values.keys())

    xs = np.array([first_mode.node_values[n]["X"] for n in nodes], dtype=float)
    ys = np.array([first_mode.node_values[n]["Y"] for n in nodes], dtype=float)
    zs = np.array([first_mode.node_values[n]["Z"] for n in nodes], dtype=float)

    spans = {
        "X": float(xs.max() - xs.min()) if xs.size else 0.0,
        "Y": float(ys.max() - ys.min()) if ys.size else 0.0,
        "Z": float(zs.max() - zs.min()) if zs.size else 0.0,
    }

    axis = min(spans, key=spans.get)
    other_max = max(v for k, v in spans.items() if k != axis)
    flat_ratio = spans[axis] / max(other_max, 1.0e-30)

    return "U" + axis, spans, float(flat_ratio)


def resolve_normal_dof(
    cfg: dict,
    modes: Dict[int, ModeData],
    selected_modes: Sequence[int],
) -> Tuple[str, Dict[str, object]]:
    raw = str(cfg.get("normal_dof", "auto")).upper().strip()
    if raw in {"UX", "UY", "UZ"}:
        return raw, {
            "source": "config",
            "modal_votes": None,
            "modal_scores": None,
            "geometry_spans": None,
            "flat_ratio": None,
        }

    geometry_axis = cfg.get("geometry_normal_axis", None)
    if geometry_axis is not None:
        geometry_axis = str(geometry_axis).upper().strip()
        if geometry_axis not in {"X", "Y", "Z"}:
            raise ValueError("geometry_normal_axis должен быть X, Y, Z или null.")
        return "U" + geometry_axis, {
            "source": "geometry_override",
            "modal_votes": None,
            "modal_scores": None,
            "geometry_spans": None,
            "flat_ratio": None,
        }

    reference_modes = cfg.get("auto_normal_reference_modes", None)
    modal_dof, modal_votes, modal_scores = estimate_modal_normal_dof(
        modes=modes,
        selected_modes=selected_modes,
        reference_modes=reference_modes,
    )
    geom_dof, geom_spans, flat_ratio = estimate_geometry_normal_dof(
        modes=modes,
        selected_modes=selected_modes,
    )

    flat_ratio_max = float(cfg.get("auto_normal_flat_ratio_max", 0.08))
    prefer = str(cfg.get("auto_normal_prefer", "modal")).lower().strip()

    if geom_dof == modal_dof:
        resolved = modal_dof
        source = "auto_modal_and_geometry_agree"
    elif flat_ratio <= flat_ratio_max and prefer == "geometry":
        resolved = geom_dof
        source = "auto_geometry"
    else:
        resolved = modal_dof
        source = "auto_modal"

    return resolved, {
        "source": source,
        "modal_votes": modal_votes,
        "modal_scores": modal_scores,
        "geometry_spans": geom_spans,
        "flat_ratio": flat_ratio,
        "modal_dof": modal_dof,
        "geometry_dof": geom_dof,
    }



def parse_mode_file(path: Path) -> ModeData:
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [line.rstrip() for line in raw_lines if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Файл моды слишком короткий: {path}")

    info_line = lines[1]
    nums = re.findall(r"[+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?", info_line)
    if len(nums) < 2:
        raise ValueError(f"Не удалось распарсить строку параметров в {path}: {info_line}")

    mode_number = int(round(float(nums[0])))
    freq_hz = float(nums[1])

    tokens = info_line.split()
    dom_dof = None
    ref_node = None
    for tok in tokens:
        t = tok.upper()
        if t in {"UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"}:
            dom_dof = t
            break
    if dom_dof is not None:
        idx = tokens.index(dom_dof)
        if idx + 1 < len(tokens):
            try:
                ref_node = int(round(float(tokens[idx + 1])))
            except Exception:
                ref_node = None

    node_values_raw: Dict[int, Dict[str, float]] = {}
    for line in lines[3:]:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            node = int(round(float(parts[0])))
            node_values_raw[node] = {
                "X": float(parts[1]),
                "Y": float(parts[2]),
                "Z": float(parts[3]),
                "UX": float(parts[4]),
                "UY": float(parts[5]),
                "UZ": float(parts[6]),
                "ROTX": float(parts[7]),
                "ROTY": float(parts[8]),
                "ROTZ": float(parts[9]),
            }
        except Exception:
            continue

    if not node_values_raw:
        raise ValueError(f"Не удалось извлечь узловые данные моды: {path}")

    if dom_dof is not None and dom_dof in {"UX", "UY", "UZ"}:
        normalization_dof = dom_dof
    else:
        transl_max = {
            dof: max(abs(vals[dof]) for vals in node_values_raw.values())
            for dof in ("UX", "UY", "UZ")
        }
        normalization_dof = max(transl_max, key=transl_max.get)

    normalization_value = max(abs(vals[normalization_dof]) for vals in node_values_raw.values())
    if normalization_value <= 1.0e-30:
        raise ValueError(
            f"Слишком малый коэффициент нормировки для mode {mode_number} "
            f"по DOF {normalization_dof} в файле {path}"
        )

    node_values: Dict[int, Dict[str, float]] = {}
    for node, vals in node_values_raw.items():
        node_values[node] = {
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

    return ModeData(
        mode_number=mode_number,
        freq_hz=freq_hz,
        node_values=node_values,
        ref_node=ref_node,
        dom_dof=dom_dof if dom_dof is not None else normalization_dof,
        normalization_value=normalization_value,
    )


def discover_modes(modal_dir: Path, selected_modes: Sequence[int]) -> Dict[int, ModeData]:
    discovered: Dict[int, ModeData] = {}
    search_roots = []

    if modal_dir.exists():
        search_roots.append(modal_dir)

    case_dir = modal_dir.parent
    if case_dir.exists() and case_dir not in search_roots:
        search_roots.append(case_dir)

    candidate_paths: List[Path] = []
    for root in search_roots:
        candidate_paths.extend(root.glob("*.txt"))
        candidate_paths.extend(root.rglob("*mode*_step*.txt"))
        candidate_paths.extend(root.rglob("*mode*.txt"))

    uniq: List[Path] = []
    seen = set()
    for p in candidate_paths:
        rp = p.resolve()
        if p.is_file() and rp not in seen:
            seen.add(rp)
            uniq.append(p)

    for path in sorted(uniq, key=lambda p: p.name.lower()):
        try:
            mode = parse_mode_file(path)
        except Exception:
            continue
        if mode.mode_number in selected_modes and mode.mode_number not in discovered:
            discovered[mode.mode_number] = mode

    missing = [m for m in selected_modes if m not in discovered]
    if missing:
        raise FileNotFoundError(f"Не найдены/не распознаны файлы мод для modes={missing}. Поиск: {search_roots}")
    return discovered


def qtag_to_float_for_step_name(qtag: str) -> float:
    """
    Converts q-tags from STEP reaction filenames to float.

    Supported forms:
      1p0              -> 1.0
      0p5              -> 0.5
      1p6071e-08       -> 1.6071e-08
      16071e-12        -> 1.6071e-08
      1.6071e-08       -> 1.6071e-08
    """
    s = str(qtag).strip().lower()
    if not s:
        raise ValueError("Пустой qtag в имени STEP-файла.")

    # Historical file names used p as decimal separator.
    # Compact names may already contain normal scientific notation.
    s = s.replace("p", ".")

    try:
        value = float(s)
    except Exception as exc:
        raise ValueError(f"Не удалось преобразовать qtag='{qtag}' в число.") from exc

    if not math.isfinite(value):
        raise ValueError(f"Некорректный qtag='{qtag}': значение не конечно.")

    return float(value)


def parse_experiment_name(stem: str, selected_modes: Sequence[int]) -> Tuple[str, np.ndarray, float, Tuple[int, ...]]:
    """
    Parses STEP reaction filenames produced by old and new generators.

    Supported canonical compact names:
      single_m1p_q16071e-12_nl.txt
      pair_m1p_m2m_q16071e-12_nl.txt
      triple_m1p_m2p_m3m_q107134e-13_nl.txt

    Supported old/common-q names:
      single_m1p_q1p0_nl.txt
      pair_m1p_m2m_q0p5_nl.txt
      triple_m1p_m2p_m3m_q0p25_nl.txt

    Supported per-mode-q names:
      pair_m1p_q16071e-12_m2m_q53567e-13_nl.txt
      triple_m1p_q1p07134e-08_m2p_q1p07134e-08_m3m_q1p07134e-08_nl.txt
    """
    m = REACTION_FILE_RE.match(stem + ".txt")
    if m is None:
        raise ValueError(f"Имя файла не соответствует STEP-шаблону: {stem}")

    family = m.group(1).lower()
    middle = m.group(2)
    selected_modes = list(selected_modes)

    q_vec = np.zeros(len(selected_modes), dtype=float)
    sign_code = [0] * len(selected_modes)

    # ------------------------------------------------------------------
    # New/per-mode form:
    #   pair_m1p_q16071e-12_m2m_q53567e-13_nl
    #   triple_m1p_q1p07134e-08_m2p_q1p07134e-08_m3m_q1p07134e-08_nl
    # ------------------------------------------------------------------
    per_mode_token_re = re.compile(
        r"(?:^|_)m(?P<mode>\d+)(?P<sign>[pm])_q(?P<qtag>[0-9pP.+\-Ee]+)(?=_m\d+[pm]_q|$)",
        re.IGNORECASE,
    )
    # Treat as per-mode-q only when the name really contains q after more than
    # one mode. Canonical compact names such as pair_m1p_m2m_q... have only
    # one final common q and must be handled by the common-q branch below.
    per_mode_matches = list(per_mode_token_re.finditer(middle)) if middle.lower().count("_q") > 1 else []
    if per_mode_matches:
        q_abs_values: List[float] = []

        for mt in per_mode_matches:
            mode_num = int(mt.group("mode"))
            sign = 1.0 if mt.group("sign").lower() == "p" else -1.0
            q_abs_i = abs(qtag_to_float_for_step_name(mt.group("qtag")))
            q_abs_values.append(q_abs_i)

            if mode_num in selected_modes:
                idx = selected_modes.index(mode_num)
                q_vec[idx] = sign * q_abs_i
                sign_code[idx] = int(sign)

        if not np.any(np.abs(q_vec) > 0.0):
            raise ValueError(f"Не удалось восстановить q-вектор из имени файла: {stem}")

        active_q_values = [q for q in q_abs_values if q > 0.0]
        if not active_q_values:
            raise ValueError(f"В имени STEP-файла нет положительного q: {stem}")

        q_abs = float(np.mean(active_q_values))
        spread = max(active_q_values) - min(active_q_values)
        ref = max(active_q_values)

        # Current canonical STEP extraction assumes one common q for pair/triple.
        # If the generator ever writes truly different q per active mode, fail
        # loudly instead of silently producing wrong G/H coefficients.
        if spread > max(1.0e-14, 1.0e-6 * ref):
            raise ValueError(
                "В одном STEP-файле найдены разные q по активным модам. "
                "Текущий canonical postprocessor ожидает одинаковый q для pair/triple.\n"
                f"file={stem}\nq_values={active_q_values}"
            )

        return family, q_vec, q_abs, tuple(sign_code)

    # ------------------------------------------------------------------
    # Common-q form:
    #   pair_m1p_m2m_q16071e-12_nl
    #   single_m1p_q1p0_nl
    # ------------------------------------------------------------------
    q_match = re.search(r"_q(?P<qtag>[0-9pP.+\-Ee]+)$", middle, re.IGNORECASE)
    if q_match is None:
        raise ValueError(f"В имени STEP-файла не найден блок q: {stem}")

    q_abs = abs(qtag_to_float_for_step_name(q_match.group("qtag")))
    modes_part = middle[: q_match.start()]

    for token in modes_part.split("_"):
        mt = MODE_TOKEN_RE.fullmatch(token)
        if mt is None:
            continue

        mode_num = int(mt.group("mode"))
        sign = 1.0 if mt.group("sign").lower() == "p" else -1.0

        if mode_num in selected_modes:
            idx = selected_modes.index(mode_num)
            q_vec[idx] = sign * q_abs
            sign_code[idx] = int(sign)

    if not np.any(np.abs(q_vec) > 0.0):
        raise ValueError(f"Не удалось восстановить q-вектор из имени файла: {stem}")

    return family, q_vec, q_abs, tuple(sign_code)

def _read_manifest_file(manifest_path: Path) -> dict:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_fe_manifest(fe_job_dir: Path, cfg: dict) -> dict:
    manifest_path = fe_job_dir / str(cfg.get("fe_manifest_filename", "fe_export_manifest.json"))
    if manifest_path.exists():
        return _read_manifest_file(manifest_path)
    return {}


def resolve_result_file(fe_job_dir: Path, cfg: dict, logical_name: str, required: bool = True) -> Optional[Path]:
    candidates: List[Path] = []

    manifest = load_fe_manifest(fe_job_dir, cfg)
    files_map = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    if logical_name in files_map:
        try:
            candidates.append(Path(files_map[logical_name]))
        except Exception:
            pass

    candidates.append(fe_job_dir / logical_name)

    if logical_name == "d3plot":
        candidates.append(fe_job_dir / "d3plot01")
    elif logical_name == "bndout":
        candidates.append(fe_job_dir / "spcforc")
        candidates.append(fe_job_dir / "ncforc")
        candidates.append(fe_job_dir / "rcforc")

    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        try:
            rc = c.resolve()
        except Exception:
            rc = c
        if str(rc) not in seen:
            seen.add(str(rc))
            uniq.append(c)

    for c in uniq:
        if c.exists():
            return c

    if required:
        raise FileNotFoundError(f"Не найден файл результата '{logical_name}' в {fe_job_dir}")
    return None


def _configure_lasso_logging() -> None:
    try:
        logging.getLogger("lasso").setLevel(logging.ERROR)
        logging.getLogger("lasso.dyna").setLevel(logging.ERROR)
        logging.getLogger("lasso.dyna.d3plot").setLevel(logging.ERROR)
    except Exception:
        pass


def parse_bndout_all_snapshots(path: Path) -> List[ReactionSnapshot]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    snapshots: List[ReactionSnapshot] = []
    current_time: Optional[float] = None
    current_nodes: Dict[int, Dict[str, float]] = {}

    def flush_snapshot():
        nonlocal current_time, current_nodes
        if current_time is not None and current_nodes:
            snapshots.append(ReactionSnapshot(time_value=current_time, node_values=current_nodes))
        current_time = None
        current_nodes = {}

    def try_extract_time(line: str) -> Optional[float]:
        compact = re.sub(r"\s+", "", line).lower()
        if "outputt=" in compact:
            m = re.search(r"t=([+\-0-9eE\.]+)", compact)
            if m:
                return float(m.group(1))
        return None

    node_re_full = re.compile(
        r"nd#\s*(\d+)\s+"
        r"xforce=\s*([+\-0-9Ee\.]+)\s+"
        r"yforce=\s*([+\-0-9Ee\.]+)\s+"
        r"zforce=\s*([+\-0-9Ee\.]+)\s+"
        r"energy=\s*([+\-0-9Ee\.]+)\s+"
        r"setid\s*=\s*(\d+)\s+"
        r"xmoment=\s*([+\-0-9Ee\.]+)\s+"
        r"ymoment=\s*([+\-0-9Ee\.]+)\s+"
        r"zmoment=\s*([+\-0-9Ee\.]+)",
        re.IGNORECASE,
    )


    node_re_basic = re.compile(
        r"nd#\s*(\d+)\s+"
        r"xforce=\s*([+\-0-9Ee\.]+)\s+"
        r"yforce=\s*([+\-0-9Ee\.]+)\s+"
        r"zforce=\s*([+\-0-9Ee\.]+)\s+"
        r"energy=\s*([+\-0-9Ee\.]+)",
        re.IGNORECASE,
    )

    for line in lines:
        tval = try_extract_time(line)
        if tval is not None:
            flush_snapshot()
            current_time = tval
            current_nodes = {}
            continue

        if current_time is None:
            continue

        nm = node_re_full.search(line)
        if nm:
            node = int(nm.group(1))
            current_nodes[node] = {
                "FX": float(nm.group(2)),
                "FY": float(nm.group(3)),
                "FZ": float(nm.group(4)),
                "MX": float(nm.group(7)),
                "MY": float(nm.group(8)),
                "MZ": float(nm.group(9)),
            }
            continue

        nm = node_re_basic.search(line)
        if nm:
            node = int(nm.group(1))
            current_nodes[node] = {
                "FX": float(nm.group(2)),
                "FY": float(nm.group(3)),
                "FZ": float(nm.group(4)),
                "MX": 0.0,
                "MY": 0.0,
                "MZ": 0.0,
            }
            continue

    flush_snapshot()

    if not snapshots:
        raise ValueError(
            f"В файле {path} не найдено ни одного временного блока bndout. "
            f"Поддерживаются форматы: full reaction block и pressure boundary condition forces."
        )
    return snapshots


def parse_bndout_last_snapshot(path: Path) -> ReactionSnapshot:
    return parse_bndout_all_snapshots(path)[-1]


def build_phi_vector_for_mode(mode: ModeData, node_order: Sequence[int], use_rotations: bool) -> np.ndarray:
    values: List[float] = []
    for node in node_order:
        nd = mode.node_values[node]
        for dof in translational_dofs():
            values.append(nd[dof])
        if use_rotations:
            for dof in rotational_dofs():
                values.append(nd[dof])
    return np.array(values, dtype=float)


def build_phi_vector_from_field(field: Dict[int, Dict[str, float]], node_order: Sequence[int], use_rotations: bool) -> np.ndarray:
    values: List[float] = []
    for node in node_order:
        nd = field[node]
        for dof in translational_dofs():
            values.append(nd.get(dof, 0.0))
        if use_rotations:
            for dof in rotational_dofs():
                values.append(nd.get(dof, 0.0))
    return np.array(values, dtype=float)


def build_reaction_vector(snapshot: ReactionSnapshot, node_order: Sequence[int], reaction_sign: float, use_rotations: bool) -> np.ndarray:
    values: List[float] = []
    for node in node_order:
        nd = snapshot.node_values[node]
        values.extend([
            reaction_sign * nd["FX"],
            reaction_sign * nd["FY"],
            reaction_sign * nd["FZ"],
        ])
        if use_rotations:
            values.extend([
                reaction_sign * nd["MX"],
                reaction_sign * nd["MY"],
                reaction_sign * nd["MZ"],
            ])
    return np.array(values, dtype=float)


def compute_generalized_force_vector(
    modes: Dict[int, ModeData],
    snapshot: ReactionSnapshot,
    selected_modes: Sequence[int],
    reaction_sign: float,
    use_rotations: bool,
) -> np.ndarray:
    common_nodes = set(snapshot.node_values.keys())
    for m in selected_modes:
        common_nodes &= set(modes[m].node_values.keys())

    if not common_nodes:
        raise ValueError("Нет общих узлов между bndout и модальными формами.")

    node_order = sorted(common_nodes)
    reaction_vec = build_reaction_vector(snapshot, node_order, reaction_sign, use_rotations)

    out = np.zeros(len(selected_modes), dtype=float)
    for i, mode_num in enumerate(selected_modes):
        phi = build_phi_vector_for_mode(modes[mode_num], node_order, use_rotations)
        out[i] = float(np.dot(phi, reaction_vec))
    return out


def compute_tangent_generalized_force_vector(
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    snapshot: ReactionSnapshot,
    q_state: np.ndarray,
    selected_modes: Sequence[int],
    reaction_sign: float,
    use_rotations: bool,
    use_dual: bool,
) -> np.ndarray:
    common_nodes = set(snapshot.node_values.keys())
    for m in selected_modes:
        common_nodes &= set(modes[m].node_values.keys())
    if use_dual:
        for m in selected_modes:
            common_nodes &= set(dual_modes[m].node_values.keys())

    if not common_nodes:
        raise ValueError("Нет общих узлов между силовым файлом и базисом для tangent generalized force.")

    node_order = sorted(common_nodes)
    dofs = all_dofs(use_rotations)
    reaction_vec = build_reaction_vector(snapshot, node_order, reaction_sign, use_rotations)

    V = build_tangent_projection_matrix(
        q=q_state,
        modes=modes,
        dual_modes=dual_modes,
        selected_modes=selected_modes,
        node_order=node_order,
        dofs=dofs,
        use_dual=use_dual,
    )
    return V.T @ reaction_vec


def discover_step_experiments(
    reactions_dir: Path,
    modes: Dict[int, ModeData],
    selected_modes: Sequence[int],
    reaction_sign: float,
    use_rotations: bool,
) -> Dict[str, StepExperiment]:
    experiments: Dict[str, StepExperiment] = {}

    for path in sorted(reactions_dir.glob("*.txt")):
        stem = path.stem
        try:
            family, q_vec, q_abs, sign_code = parse_experiment_name(stem, selected_modes)
        except Exception:
            continue

        if stem.endswith("_lin"):
            name_base = stem[:-4]
            branch = "lin"
        elif stem.endswith("_nl"):
            name_base = stem[:-3]
            branch = "nl"
        else:
            continue

        snap = parse_bndout_last_snapshot(path)
        force_vec = compute_generalized_force_vector(
            modes=modes,
            snapshot=snap,
            selected_modes=selected_modes,
            reaction_sign=reaction_sign,
            use_rotations=use_rotations,
        )

        exp = experiments.get(name_base)
        if exp is None:
            exp = StepExperiment(
                family=family,
                name_base=name_base,
                q_vector=q_vec,
                q_abs=q_abs,
                sign_code=sign_code,
            )
            experiments[name_base] = exp

        if branch == "lin":
            exp.lin_file = path
            exp.lin_force = force_vec
        else:
            exp.nl_file = path
            exp.nl_force = force_vec

    for exp in experiments.values():
        if exp.lin_force is not None and exp.nl_force is not None:
            exp.gamma_force = exp.nl_force - exp.lin_force

    if not experiments:
        raise FileNotFoundError(f"В {reactions_dir} не найдены STEP reaction-файлы (*.txt).")
    return experiments


def build_experiment_index(experiments: Dict[str, StepExperiment], selected_modes: Sequence[int]) -> Dict[Tuple, StepExperiment]:
    idx: Dict[Tuple, StepExperiment] = {}
    selected_modes = list(selected_modes)

    for exp in experiments.values():
        active = [(selected_modes[i], int(math.copysign(1.0, exp.q_vector[i]))) for i in range(len(selected_modes)) if abs(exp.q_vector[i]) > 0.0]
        q = exp.q_abs
        active_sorted = tuple(sorted(active, key=lambda x: x[0]))

        if exp.family == "single" and len(active_sorted) == 1:
            idx[("single", active_sorted[0][0], active_sorted[0][1], q)] = exp
        elif exp.family == "pair" and len(active_sorted) == 2:
            idx[("pair", active_sorted[0][0], active_sorted[0][1], active_sorted[1][0], active_sorted[1][1], q)] = exp
        elif exp.family == "triple" and len(active_sorted) == 3:
            idx[(
                "triple",
                active_sorted[0][0], active_sorted[0][1],
                active_sorted[1][0], active_sorted[1][1],
                active_sorted[2][0], active_sorted[2][1],
                q
            )] = exp

    return idx


def avg_vectors(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Пустой список векторов для усреднения.")
    return np.mean(np.vstack(vectors), axis=0)


def extract_canonical_step_coefficients(
    experiments: Dict[str, StepExperiment],
    selected_modes: Sequence[int],
) -> Dict[str, object]:
    selected_modes = list(selected_modes)
    n = len(selected_modes)
    idx = build_experiment_index(experiments, selected_modes)

    K = np.zeros((n, n), dtype=float)
    G = np.zeros((n, n, n), dtype=float)
    H = np.zeros((n, n, n, n), dtype=float)

    single_gamma_pos: Dict[Tuple[int, float], np.ndarray] = {}
    single_gamma_neg: Dict[Tuple[int, float], np.ndarray] = {}

    q_values_by_mode: Dict[int, List[float]] = {m: [] for m in selected_modes}
    for key in idx:
        if key[0] == "single":
            _, r, _sgn, q = key
            q_values_by_mode[r].append(q)
    for r in q_values_by_mode:
        q_values_by_mode[r] = sorted(set(q_values_by_mode[r]))

    for r in selected_modes:
        K_cols = []
        Grr_rows = []
        Hrrr_rows = []

        for q in q_values_by_mode[r]:
            ep = idx.get(("single", r, +1, q))
            em = idx.get(("single", r, -1, q))
            if ep is None or em is None or ep.gamma_force is None or em.gamma_force is None or ep.lin_force is None or em.lin_force is None:
                raise ValueError(f"Для mode {r} и q={q} отсутствуют полные single +/- STEP данные.")

            single_gamma_pos[(r, q)] = ep.gamma_force.copy()
            single_gamma_neg[(r, q)] = em.gamma_force.copy()

            k_col = 0.5 * (ep.lin_force / q - em.lin_force / q)
            K_cols.append(k_col)

            g_rr = (ep.gamma_force + em.gamma_force) / (2.0 * q * q)
            h_rrr = (ep.gamma_force - em.gamma_force) / (2.0 * q * q * q)
            Grr_rows.append(g_rr)
            Hrrr_rows.append(h_rrr)

        ir = selected_modes.index(r)
        K[:, ir] = avg_vectors(K_cols)
        G[:, ir, ir] = avg_vectors(Grr_rows)
        H[:, ir, ir, ir] = avg_vectors(Hrrr_rows)

    q_values_by_pair: Dict[Tuple[int, int], List[float]] = {}
    for key in idx:
        if key[0] == "pair":
            _, r, _sr, s, _ss, q = key
            q_values_by_pair.setdefault((r, s), []).append(q)
    for p in q_values_by_pair:
        q_values_by_pair[p] = sorted(set(q_values_by_pair[p]))

    for (r, s), qvals in q_values_by_pair.items():
        ir = selected_modes.index(r)
        is_ = selected_modes.index(s)

        g_rs_all = []
        h_rrs_all = []
        h_rss_all = []

        for q in qvals:
            e_pp = idx.get(("pair", r, +1, s, +1, q))
            e_mp = idx.get(("pair", r, -1, s, +1, q))
            e_pm = idx.get(("pair", r, +1, s, -1, q))

            if any(e is None or e.gamma_force is None for e in [e_pp, e_mp, e_pm]):
                raise ValueError(f"Для пары ({r},{s}) и q={q} отсутствуют нужные pair STEP данные.")

            sr_p = single_gamma_pos[(r, q)]
            sr_m = single_gamma_neg[(r, q)]
            ss_p = single_gamma_pos[(s, q)]
            ss_m = single_gamma_neg[(s, q)]

            b_pp = e_pp.gamma_force - sr_p - ss_p
            b_mp = e_mp.gamma_force - sr_m - ss_p
            b_pm = e_pm.gamma_force - sr_p - ss_m

            h_rrs = (b_pp + b_mp) / (6.0 * q**3)
            h_rss = (b_pp + b_pm) / (6.0 * q**3)
            g_rs = (b_pp - 3.0 * q**3 * h_rrs - 3.0 * q**3 * h_rss) / (2.0 * q**2)

            g_rs_all.append(g_rs)
            h_rrs_all.append(h_rrs)
            h_rss_all.append(h_rss)

        g_rs_avg = avg_vectors(g_rs_all)
        h_rrs_avg = avg_vectors(h_rrs_all)
        h_rss_avg = avg_vectors(h_rss_all)

        G[:, ir, is_] = g_rs_avg
        G[:, is_, ir] = g_rs_avg

        H[:, ir, ir, is_] = h_rrs_avg
        H[:, ir, is_, ir] = h_rrs_avg
        H[:, is_, ir, ir] = h_rrs_avg

        H[:, ir, is_, is_] = h_rss_avg
        H[:, is_, ir, is_] = h_rss_avg
        H[:, is_, is_, ir] = h_rss_avg

    q_values_by_triple: Dict[Tuple[int, int, int], List[float]] = {}
    for key in idx:
        if key[0] == "triple":
            _, r, _sr, s, _ss, t, _st, q = key
            q_values_by_triple.setdefault((r, s, t), []).append(q)
    for triad in q_values_by_triple:
        q_values_by_triple[triad] = sorted(set(q_values_by_triple[triad]))

    for (r, s, t), qvals in q_values_by_triple.items():
        ir = selected_modes.index(r)
        is_ = selected_modes.index(s)
        it = selected_modes.index(t)

        h_rst_all = []

        for q in qvals:
            e_ppp = idx.get(("triple", r, +1, s, +1, t, +1, q))
            if e_ppp is None or e_ppp.gamma_force is None:
                raise ValueError(f"Для тройки ({r},{s},{t}) и q={q} отсутствует triple +++ STEP данные.")

            gr = single_gamma_pos[(r, q)]
            gs = single_gamma_pos[(s, q)]
            gt = single_gamma_pos[(t, q)]

            pair_rs = 2.0 * G[:, ir, is_] * q**2 + 3.0 * H[:, ir, ir, is_] * q**3 + 3.0 * H[:, ir, is_, is_] * q**3
            pair_rt = 2.0 * G[:, ir, it] * q**2 + 3.0 * H[:, ir, ir, it] * q**3 + 3.0 * H[:, ir, it, it] * q**3
            pair_st = 2.0 * G[:, is_, it] * q**2 + 3.0 * H[:, is_, is_, it] * q**3 + 3.0 * H[:, is_, it, it] * q**3

            residual = e_ppp.gamma_force - gr - gs - gt - pair_rs - pair_rt - pair_st
            h_rst = residual / (6.0 * q**3)
            h_rst_all.append(h_rst)

        h_rst_avg = avg_vectors(h_rst_all)
        perms = [
            (ir, is_, it), (ir, it, is_),
            (is_, ir, it), (is_, it, ir),
            (it, ir, is_), (it, is_, ir),
        ]
        for a, b, c in perms:
            H[:, a, b, c] = h_rst_avg

    return {
        "K": K,
        "G": G,
        "H": H,
        "experiments_index_size": len(idx),
    }


def evaluate_canonical_gamma(q: np.ndarray, G: np.ndarray, H: np.ndarray) -> np.ndarray:
    n = len(q)
    out = np.zeros(n, dtype=float)
    for l in range(n):
        val = 0.0
        for r in range(n):
            for s in range(n):
                val += G[l, r, s] * q[r] * q[s]
        for r in range(n):
            for s in range(n):
                for t in range(n):
                    val += H[l, r, s, t] * q[r] * q[s] * q[t]
        out[l] = val
    return out


# ---------------------------------------------------------------------
# D3PLOT reading: lasso-python backend
# ---------------------------------------------------------------------
def _load_lasso_d3plot_symbols():
    try:
        _configure_lasso_logging()
        from lasso.dyna import D3plot, ArrayType
        return D3plot, ArrayType
    except Exception as exc:
        raise ImportError(
            "Не удалось импортировать lasso-python из текущего Python-окружения.\n"
            "Установите пакет lasso-python в то же окружение, которым запускается скрипт,\n"
            "например:\n"
            "python -m pip install lasso-python\n"
            "И проверьте импорт командой:\n"
            "python -c \"from lasso.dyna import D3plot, ArrayType; print('OK')\""
        ) from exc


def _try_lasso_array(arrays, array_type, *names):
    tried = []
    for name in names:
        tried.append(name)
        # direct string key
        try:
            return arrays[name]
        except Exception:
            pass
        # ArrayType attribute
        try:
            attr = getattr(array_type, name)
            return arrays[attr]
        except Exception:
            pass
    raise KeyError(f"Не удалось найти массив в d3plot по ключам {tried}")


def read_d3plot_states_lasso(d3plot_path: Path) -> List[Tuple[float, Dict[int, Dict[str, float]]]]:
    D3plot, ArrayType = _load_lasso_d3plot_symbols()

    lasso_logger = logging.getLogger("lasso.dyna.d3plot")
    old_level = lasso_logger.level
    lasso_logger.setLevel(logging.ERROR)

    try:
        d3 = D3plot(str(d3plot_path))
        arrays = d3.arrays
    finally:
        lasso_logger.setLevel(old_level)

    node_ids = np.asarray(_try_lasso_array(arrays, ArrayType, "node_ids", "node_id")).astype(int).reshape(-1)
    node_coords = np.asarray(_try_lasso_array(arrays, ArrayType, "node_coordinates", "node_coordinate"))
    node_disp = np.asarray(_try_lasso_array(arrays, ArrayType, "node_displacement", "node_displacements"))
    try:
        times = np.asarray(_try_lasso_array(arrays, ArrayType, "timesteps", "global_timesteps", "time"))
    except Exception:
        times = np.arange(node_disp.shape[0], dtype=float)

    # geometry arrays can be (nNodes, 3), displacement usually (nStates, nNodes, 3)
    if node_coords.ndim != 2 or node_coords.shape[1] < 3:
        raise ValueError(f"Некорректная форма node_coordinates: {node_coords.shape}")
    if node_disp.ndim != 3 or node_disp.shape[2] < 3:
        raise ValueError(f"Некорректная форма node_displacement: {node_disp.shape}")

    n_states = node_disp.shape[0]
    if times.size != n_states:
        if times.size == 1:
            times = np.linspace(0.0, float(times[0]), n_states)
        else:
            times = np.arange(n_states, dtype=float)

    states = []
    for ist in range(n_states):
        field: Dict[int, Dict[str, float]] = {}
        disp_state = node_disp[ist]
        for idx, nid in enumerate(node_ids):
            field[int(nid)] = {
                "UX": float(disp_state[idx, 0]),
                "UY": float(disp_state[idx, 1]),
                "UZ": float(disp_state[idx, 2]),
                "X": float(node_coords[idx, 0]),
                "Y": float(node_coords[idx, 1]),
                "Z": float(node_coords[idx, 2]),
            }
        states.append((float(times[ist]), field))
    return states


def read_d3plot_states(path: Path, backend: str = "auto") -> List[Tuple[float, Dict[int, Dict[str, float]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден d3plot: {path}")

    if backend in ("auto", "lasso"):
        return read_d3plot_states_lasso(path)

    raise ValueError(f"Неподдерживаемый d3plot backend: {backend}")


def apply_d3plot_reference_state(
    raw_states: List[Tuple[float, Dict[int, Dict[str, float]]]],
    cfg: dict,
) -> Tuple[List[Tuple[float, Dict[int, Dict[str, float]]]], dict]:
    """
    Converts d3plot displacement states to increments from a reference state.

    Why this is needed:
    some LS-DYNA/lasso exports may contain an initial offset or deformed/absolute
    nodal positions in the displacement-like array. If this offset is not removed,
    FE and ROM curves start from a non-zero W/h value and may look horizontal.
    """
    if not raw_states:
        return raw_states, {
            "mode": "empty",
            "enabled": False,
            "reference_index": None,
            "reference_time": None,
            "max_abs_reference_component": 0.0,
        }

    mode = str(cfg.get("d3plot_reference_state", "first")).strip().lower()

    if mode in ("none", "zero", "absolute", "off", "false"):
        return raw_states, {
            "mode": mode,
            "enabled": False,
            "reference_index": None,
            "reference_time": None,
            "max_abs_reference_component": 0.0,
        }

    if mode in ("auto", "first", "initial", "first_state"):
        ref_index = 0
    else:
        try:
            ref_index = int(mode)
        except Exception as exc:
            raise ValueError(
                "d3plot_reference_state должен быть 'first'/'auto'/'none' "
                "или целым индексом состояния."
            ) from exc

    if ref_index < 0:
        ref_index = len(raw_states) + ref_index
    if ref_index < 0 or ref_index >= len(raw_states):
        raise IndexError(
            f"d3plot_reference_state={mode} вне диапазона d3plot states: "
            f"0..{len(raw_states) - 1}"
        )

    reference_time, reference_field = raw_states[ref_index]
    out: List[Tuple[float, Dict[int, Dict[str, float]]]] = []
    max_ref = 0.0

    for _nid, vals in reference_field.items():
        for dof in translational_dofs():
            max_ref = max(max_ref, abs(float(vals.get(dof, 0.0))))

    for t, field in raw_states:
        new_field: Dict[int, Dict[str, float]] = {}
        for nid, vals in field.items():
            ref_vals = reference_field.get(nid, {})
            new_vals = dict(vals)
            for dof in translational_dofs():
                new_vals[dof] = float(vals.get(dof, 0.0)) - float(ref_vals.get(dof, 0.0))
            new_field[int(nid)] = new_vals
        out.append((float(t), new_field))

    return out, {
        "mode": mode,
        "enabled": True,
        "reference_index": int(ref_index),
        "reference_time": float(reference_time),
        "max_abs_reference_component": float(max_ref),
    }


def compute_w_measure_from_field(field: Dict[int, Dict[str, float]], normal_dof: str, thickness: float, measure: str) -> float:
    if measure != "wmax_over_h":
        raise NotImplementedError(f"Неподдерживаемый measure: {measure}")
    wmax = max(abs(v.get(normal_dof, 0.0)) for v in field.values()) if field else 0.0
    return wmax / thickness


def build_fe_states_from_d3plot(fe_job_dir: Path, cfg: dict) -> List[FEState]:
    d3plot_name = str(cfg.get("d3plot_filename", "d3plot"))
    d3plot_path = resolve_result_file(fe_job_dir, cfg, d3plot_name, required=True)
    raw_states = read_d3plot_states(d3plot_path, backend=str(cfg.get("d3plot_backend", "auto")))

    if not raw_states:
        raise ValueError(f"Из {d3plot_path} не получено ни одного состояния.")

    raw_states, ref_meta = apply_d3plot_reference_state(raw_states, cfg)
    cfg["_d3plot_reference_meta"] = ref_meta

    times = np.array([s[0] for s in raw_states], dtype=float)
    t0 = float(times[0])
    t1 = float(times[-1])
    if abs(t1 - t0) < 1.0e-30:
        alphas = np.linspace(0.0, 1.0, len(raw_states))
    else:
        alphas = (times - t0) / (t1 - t0)

    out: List[FEState] = []
    for idx, ((t, field), alpha) in enumerate(zip(raw_states, alphas), start=1):
        out.append(FEState(
            index=idx,
            time_value=float(t),
            load_pct=float(alpha * 100.0),
            field=field,
            w_value=compute_w_measure_from_field(
                field=field,
                normal_dof=str(cfg["normal_dof"]).upper(),
                thickness=float(cfg["thickness"]),
                measure=str(cfg["w_measure"]),
            ),
        ))
    return out


def interpolate_vector_path(x_src: np.ndarray, y_src: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    # y_src shape (n_src, dim)
    dim = y_src.shape[1]
    out = np.zeros((x_target.size, dim), dtype=float)
    for j in range(dim):
        out[:, j] = np.interp(x_target, x_src, y_src[:, j])
    return out


def build_fe_generalized_force_path_from_text_result(
    fe_job_dir: Path,
    result_name: str,
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    cfg: dict,
    target_times: np.ndarray,
    fe_state_times: np.ndarray,
    fe_qs: Sequence[np.ndarray],
) -> np.ndarray:
    result_path = resolve_result_file(fe_job_dir, cfg, result_name, required=True)
    snapshots = parse_bndout_all_snapshots(result_path)
    src_times = np.array([s.time_value for s in snapshots], dtype=float)

    q_src = interpolate_vector_path(
        fe_state_times,
        np.vstack([np.asarray(q, dtype=float) for q in fe_qs]),
        src_times,
    )

    qext = np.zeros((len(snapshots), len(selected_modes)), dtype=float)
    use_tangent = bool(cfg.get("fe_generalized_force_use_tangent_basis", True))
    use_dual = bool(cfg.get("use_dual_modes", True))

    for i, snap in enumerate(snapshots):
        if use_tangent and use_dual and dual_modes:
            qext[i, :] = compute_tangent_generalized_force_vector(
                modes=modes,
                dual_modes=dual_modes,
                snapshot=snap,
                q_state=q_src[i, :],
                selected_modes=selected_modes,
                reaction_sign=float(cfg["reaction_sign"]),
                use_rotations=bool(cfg["use_rotations_in_generalized_force"]),
                use_dual=True,
            )
        else:
            qext[i, :] = compute_generalized_force_vector(
                modes=modes,
                snapshot=snap,
                selected_modes=selected_modes,
                reaction_sign=float(cfg["reaction_sign"]),
                use_rotations=bool(cfg["use_rotations_in_generalized_force"]),
            )

    return interpolate_vector_path(src_times, qext, target_times)


def build_fe_generalized_force_path_from_projected_equilibrium(
    fe_qs: Sequence[np.ndarray],
    K: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    qext = np.zeros((len(fe_qs), len(fe_qs[0])), dtype=float)
    for i, q in enumerate(fe_qs):
        qext[i, :] = K @ q + evaluate_canonical_gamma(q, G, H)
    return qext


def _max_vector_path_norm(path_values: np.ndarray) -> float:
    arr = np.asarray(path_values, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        return float(np.linalg.norm(arr))
    return float(max(np.linalg.norm(row) for row in arr))


def _validate_external_force_path(
    qext: np.ndarray,
    source: str,
    fe_qs: Optional[Sequence[np.ndarray]],
    K: Optional[np.ndarray],
    G: Optional[np.ndarray],
    H: Optional[np.ndarray],
    cfg: dict,
) -> Tuple[bool, str]:
    """
    Checks whether a force path read from bndout/spcforc/ncforc/rcforc is usable.

    In practice LS-DYNA text outputs can be present and parsed but still contain
    zero useful generalized load for this ROM projection. If such a zero path is
    accepted, Newton solves K*q + G(q,q) + H(q,q,q) = 0 and ROM remains a
    horizontal zero line. In auto mode we therefore reject a practically zero
    external path and fall back to projected_equilibrium.
    """
    qext_norm = _max_vector_path_norm(qext)
    abs_tol = float(cfg.get("force_path_min_abs_norm", 1.0e-30))
    rel_tol = float(cfg.get("force_path_min_relative_norm", 1.0e-8))

    if qext_norm <= abs_tol:
        if fe_qs is None or K is None or G is None or H is None:
            return False, (
                f"{source}: generalized force path is zero "
                f"(max_norm={qext_norm:.12e})"
            )

    if fe_qs is not None and K is not None and G is not None and H is not None:
        projected = build_fe_generalized_force_path_from_projected_equilibrium(fe_qs, K, G, H)
        projected_norm = _max_vector_path_norm(projected)

        if projected_norm > abs_tol:
            ratio = qext_norm / projected_norm
            if ratio < rel_tol:
                return False, (
                    f"{source}: generalized force path is almost zero relative to "
                    f"projected_equilibrium "
                    f"(source_max_norm={qext_norm:.12e}, "
                    f"projected_max_norm={projected_norm:.12e}, "
                    f"ratio={ratio:.12e}, limit={rel_tol:.12e})"
                )

    return True, (
        f"{source}: accepted generalized force path "
        f"(max_norm={qext_norm:.12e})"
    )


def build_fe_generalized_force_path(
    fe_job_dir: Path,
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    cfg: dict,
    target_times: np.ndarray,
    fe_qs: Optional[Sequence[np.ndarray]] = None,
    K: Optional[np.ndarray] = None,
    G: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    source_setting = str(cfg.get("fe_generalized_force_source", "auto")).lower()
    auto_mode = source_setting == "auto"

    if auto_mode:
        sources = list(cfg.get(
            "fe_generalized_force_sources",
            ["bndout", "spcforc", "ncforc", "rcforc", "projected_equilibrium"],
        ))
    else:
        sources = [source_setting]

    validate_auto_paths = bool(cfg.get("validate_external_force_path", True))

    errors: List[str] = []
    diagnostics: List[str] = []

    for source in sources:
        source = str(source).lower()

        if source == "projected_equilibrium":
            if fe_qs is None or K is None or G is None or H is None:
                errors.append("projected_equilibrium: нужны fe_qs, K, G, H")
                continue

            qext = build_fe_generalized_force_path_from_projected_equilibrium(fe_qs, K, G, H)
            diagnostics.append(
                f"projected_equilibrium: used "
                f"(max_norm={_max_vector_path_norm(qext):.12e})"
            )
            cfg["_force_path_diagnostics"] = diagnostics
            return qext, "projected_equilibrium"

        try:
            if fe_qs is None:
                errors.append(f"{source}: нужны fe_qs для построения generalized force path")
                continue

            qext = build_fe_generalized_force_path_from_text_result(
                fe_job_dir=fe_job_dir,
                result_name=source,
                modes=modes,
                dual_modes=dual_modes,
                selected_modes=selected_modes,
                cfg=cfg,
                target_times=target_times,
                fe_state_times=target_times,
                fe_qs=fe_qs,
            )

            if auto_mode and validate_auto_paths:
                ok, msg = _validate_external_force_path(
                    qext=qext,
                    source=source,
                    fe_qs=fe_qs,
                    K=K,
                    G=G,
                    H=H,
                    cfg=cfg,
                )
                diagnostics.append(msg)
                if not ok:
                    errors.append(msg)
                    continue
            else:
                diagnostics.append(
                    f"{source}: used without auto validation "
                    f"(max_norm={_max_vector_path_norm(qext):.12e})"
                )

            cfg["_force_path_diagnostics"] = diagnostics
            return qext, source

        except Exception as exc:
            errors.append(f"{source}: {exc}")
            diagnostics.append(f"{source}: failed: {exc}")
            continue

    cfg["_force_path_diagnostics"] = diagnostics
    raise RuntimeError(
        "Не удалось построить путь внешней обобщённой нагрузки FE. Проверенные источники:\n"
        + "\n".join(errors)
    )


def discover_dual_mode_dirs(case_dir: Path, selected_modes: Sequence[int], cfg: dict) -> Dict[int, Tuple[Path, Path]]:
    dual_root = case_dir / str(cfg["dual_results_dir_name"])
    if not dual_root.exists():
        raise FileNotFoundError(f"Не найдена папка dual_fields: {dual_root}")

    prefix = str(cfg.get("dual_job_prefix", "dual_mode"))
    result: Dict[int, Tuple[Path, Path]] = {}

    for m in selected_modes:
        p_dir = dual_root / f"{prefix}{m}_p"
        d2_dir = dual_root / f"{prefix}{m}_2p"
        if not p_dir.exists():
            raise FileNotFoundError(f"Не найдена dual-папка: {p_dir}")
        if not d2_dir.exists():
            raise FileNotFoundError(f"Не найдена dual-папка: {d2_dir}")
        result[m] = (p_dir, d2_dir)
    return result


def build_dual_modes(
    case_dir: Path,
    modes: Dict[int, ModeData],
    selected_modes: Sequence[int],
    cfg: dict,
) -> Dict[int, DualModeData]:
    if not bool(cfg.get("use_dual_modes", True)):
        return {}

    p_value = float(cfg["dual_p_value"])
    dual_dirs = discover_dual_mode_dirs(case_dir, selected_modes, cfg)
    normal_dof = str(cfg["normal_dof"]).upper()
    inplane_dofs = [dof for dof in translational_dofs() if dof != normal_dof]

    dual_modes: Dict[int, DualModeData] = {}
    for m in selected_modes:
        p_dir, d2_dir = dual_dirs[m]
        d3plot_name = str(cfg.get("d3plot_filename", "d3plot"))
        p_d3plot = resolve_result_file(p_dir, cfg, d3plot_name, required=True)
        d2_d3plot = resolve_result_file(d2_dir, cfg, d3plot_name, required=True)
        p_states = read_d3plot_states(p_d3plot, backend=str(cfg.get("d3plot_backend", "auto")))
        d2_states = read_d3plot_states(d2_d3plot, backend=str(cfg.get("d3plot_backend", "auto")))
        if not p_states or not d2_states:
            raise ValueError(f"Не удалось прочитать dual states для mode {m}")

        _, w1 = p_states[int(cfg["d3plot_state_index"])]
        _, w2 = d2_states[int(cfg["d3plot_state_index"])]

        common_nodes = set(w1.keys()) & set(w2.keys()) & set(modes[m].node_values.keys())
        if not common_nodes:
            raise ValueError(f"Нет общих узлов при построении dual mode {m}")

        node_order = sorted(common_nodes)

        raw_field: Dict[int, Dict[str, float]] = {}
        for nid in node_order:
            vals: Dict[str, float] = {}
            for dof in translational_dofs():
                vals[dof] = float(w2[nid].get(dof, 0.0) - 2.0 * w1[nid].get(dof, 0.0))
            raw_field[nid] = vals

        if bool(cfg.get("dual_zero_normal_dof", True)):
            for nid in node_order:
                raw_field[nid][normal_dof] = 0.0

        if bool(cfg.get("dual_remove_inplane_mean", True)):
            for dof in inplane_dofs:
                mean_val = float(np.mean([raw_field[nid][dof] for nid in node_order]))
                for nid in node_order:
                    raw_field[nid][dof] -= mean_val

        if bool(cfg.get("dual_remove_affine_inplane", True)):
            remove_affine_inplane_component(raw_field, modes[m], node_order, inplane_dofs)

        if bool(cfg.get("dual_remove_linear_projection", True)) and inplane_dofs:
            raw_vec = flatten_field(raw_field, node_order, inplane_dofs)
            Phi_ip = build_linear_projection_matrix(modes, selected_modes, node_order, inplane_dofs)
            try:
                a, *_ = np.linalg.lstsq(Phi_ip, raw_vec, rcond=None)
                raw_vec = raw_vec - Phi_ip @ a
                cleaned = write_flat_field(raw_vec, node_order, inplane_dofs)
                for nid in node_order:
                    for dof in inplane_dofs:
                        raw_field[nid][dof] = cleaned[nid][dof]
            except np.linalg.LinAlgError:
                pass

        # raw_norm should be taken after all cleaning
        scale_candidates = [abs(raw_field[nid][dof]) for nid in node_order for dof in inplane_dofs]
        raw_norm = max(scale_candidates) if scale_candidates else 0.0
        if raw_norm < 1.0e-12:
            raise ValueError(
                f"Слишком малый raw_norm для dual mode {m}. "
                f"Проверьте dual-прогоны и граничные условия."
            )

        node_values: Dict[int, Dict[str, float]] = {}
        for nid in node_order:
            node_values[nid] = {}
            for dof in translational_dofs():
                node_values[nid][dof] = raw_field[nid].get(dof, 0.0) / raw_norm
            for dof in rotational_dofs():
                node_values[nid][dof] = 0.0

        coeff = raw_norm / (2.0 * p_value * p_value)
        dual_modes[m] = DualModeData(
            source_mode_number=m,
            p_value=p_value,
            raw_norm=raw_norm,
            coeff=coeff,
            node_values=node_values,
        )

    return dual_modes


def build_linear_projection_matrix(
    modes: Dict[int, ModeData],
    selected_modes: Sequence[int],
    node_order: Sequence[int],
    dofs: Sequence[str],
) -> np.ndarray:
    cols = []
    for mode_num in selected_modes:
        col = []
        mode = modes[mode_num]
        for node in node_order:
            vals = mode.node_values[node]
            for dof in dofs:
                col.append(vals[dof])
        cols.append(np.array(col, dtype=float))
    return np.column_stack(cols)


def build_dual_matrix(
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    node_order: Sequence[int],
    dofs: Sequence[str],
) -> np.ndarray:
    cols = []
    for mode_num in selected_modes:
        dm = dual_modes[mode_num]
        col = []
        for node in node_order:
            vals = dm.node_values[node]
            for dof in dofs:
                col.append(vals.get(dof, 0.0))
        cols.append(np.array(col, dtype=float))
    return np.column_stack(cols)


def flatten_field(field: Dict[int, Dict[str, float]], node_order: Sequence[int], dofs: Sequence[str]) -> np.ndarray:
    out = []
    for node in node_order:
        vals = field[node]
        for dof in dofs:
            out.append(vals.get(dof, 0.0))
    return np.array(out, dtype=float)


def write_flat_field(field_vec: np.ndarray, node_order: Sequence[int], dofs: Sequence[str]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    k = 0
    for node in node_order:
        vals: Dict[str, float] = {}
        for dof in dofs:
            vals[dof] = float(field_vec[k])
            k += 1
        out[node] = vals
    return out



def remove_affine_inplane_component(
    raw_field: Dict[int, Dict[str, float]],
    mode: ModeData,
    node_order: Sequence[int],
    inplane_dofs: Sequence[str],
) -> None:
    """
    Remove best-fit affine in-plane drift:
        u_dof ~= a0 + a1 * X + a2 * Z
    This catches rigid translation / linear drift that otherwise produces
    artificial raw_norm ~ O(structure length).
    """
    if not inplane_dofs:
        return
    A = []
    for nid in node_order:
        md = mode.node_values[nid]
        A.append([1.0, float(md["X"]), float(md["Z"])])
    A = np.asarray(A, dtype=float)

    for dof in inplane_dofs:
        y = np.asarray([raw_field[nid][dof] for nid in node_order], dtype=float)
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            y_fit = A @ coeffs
            y_clean = y - y_fit
            for nid, val in zip(node_order, y_clean):
                raw_field[nid][dof] = float(val)
        except np.linalg.LinAlgError:
            pass


def fit_fe_q_vector(
    fe_field: Dict[int, Dict[str, float]],
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    cfg: dict,
    q_prev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Stable FE->q projection.
    By default fit only the transverse normal displacement on the primary basis.
    This avoids sign-flipping spurious in-plane contamination from dual fields.
    """
    fit_space = str(cfg.get("fe_q_fit_space", "primary_normal")).lower()
    normal_dof = str(cfg["normal_dof"]).upper()

    common_nodes = set(fe_field.keys())
    for m in selected_modes:
        common_nodes &= set(modes[m].node_values.keys())
    if fit_space == "dual_manifold" and bool(cfg.get("use_dual_modes", True)):
        for m in selected_modes:
            common_nodes &= set(dual_modes[m].node_values.keys())
    if not common_nodes:
        raise ValueError("Нет общих узлов между FE полем и базисом для FE->q projection.")

    node_order = sorted(common_nodes)
    reg = float(cfg.get("fe_q_regularization", 1.0e-8))
    smooth = float(cfg.get("fe_q_smoothness_weight", 1.0e-4))

    if fit_space == "primary_normal":
        dofs = [normal_dof]
        Phi = build_linear_projection_matrix(modes, selected_modes, node_order, dofs)
        u_fe = flatten_field(fe_field, node_order, dofs)
        A = Phi.T @ Phi + reg * np.eye(len(selected_modes))
        b = Phi.T @ u_fe
        if q_prev is not None:
            A = A + smooth * np.eye(len(selected_modes))
            b = b + smooth * q_prev
        q = np.linalg.solve(A, b)
        u_rec = Phi @ q
        rel = float(np.linalg.norm(u_fe - u_rec) / max(np.linalg.norm(u_fe), 1.0e-30))
        return q, rel

    if fit_space == "primary_trans":
        dofs = list(cfg.get("fe_projection_dofs", ["UX", "UY", "UZ"]))
        Phi = build_linear_projection_matrix(modes, selected_modes, node_order, dofs)
        u_fe = flatten_field(fe_field, node_order, dofs)
        A = Phi.T @ Phi + reg * np.eye(len(selected_modes))
        b = Phi.T @ u_fe
        if q_prev is not None:
            A = A + smooth * np.eye(len(selected_modes))
            b = b + smooth * q_prev
        q = np.linalg.solve(A, b)
        u_rec = Phi @ q
        rel = float(np.linalg.norm(u_fe - u_rec) / max(np.linalg.norm(u_fe), 1.0e-30))
        return q, rel

    # fallback to old dual manifold fit
    return fit_fe_field_to_dual_manifold(
        fe_field=fe_field,
        modes=modes,
        dual_modes=dual_modes,
        selected_modes=selected_modes,
        projection_dofs=list(cfg["fe_projection_dofs"]),
        dual_enabled=bool(cfg.get("use_dual_modes", True)),
    )



def build_tangent_projection_matrix(
    q: np.ndarray,
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    node_order: Sequence[int],
    dofs: Sequence[str],
    use_dual: bool,
) -> np.ndarray:
    Phi = build_linear_projection_matrix(modes, selected_modes, node_order, dofs)
    if not use_dual or not dual_modes:
        return Phi

    Psi_dual = build_dual_matrix(dual_modes, selected_modes, node_order, dofs)
    dual_coeffs = np.array([dual_modes[m].coeff for m in selected_modes], dtype=float)

    V = Phi.copy()
    for i in range(len(selected_modes)):
        V[:, i] += 2.0 * q[i] * dual_coeffs[i] * Psi_dual[:, i]
    return V


def dual_manifold_displacement(
    q: np.ndarray,
    Phi: np.ndarray,
    Psi_dual: Optional[np.ndarray],
    dual_coeffs: Optional[np.ndarray],
) -> np.ndarray:
    u = Phi @ q
    if Psi_dual is not None and dual_coeffs is not None:
        u = u + Psi_dual @ (dual_coeffs * (q ** 2))
    return u


def fit_fe_field_to_dual_manifold(
    fe_field: Dict[int, Dict[str, float]],
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    projection_dofs: Sequence[str],
    dual_enabled: bool,
) -> Tuple[np.ndarray, float]:
    common_nodes = set(fe_field.keys())
    for m in selected_modes:
        common_nodes &= set(modes[m].node_values.keys())
    if dual_enabled:
        for m in selected_modes:
            common_nodes &= set(dual_modes[m].node_values.keys())

    if not common_nodes:
        raise ValueError("Нет общих узлов между FE полем и модальным/dual базисом.")

    node_order = sorted(common_nodes)
    Phi = build_linear_projection_matrix(modes, selected_modes, node_order, projection_dofs)
    u_fe = flatten_field(fe_field, node_order, projection_dofs)

    if not dual_enabled:
        q, *_ = np.linalg.lstsq(Phi, u_fe, rcond=None)
        u_rec = Phi @ q
        rel = float(np.linalg.norm(u_fe - u_rec) / max(np.linalg.norm(u_fe), 1.0e-30))
        return q, rel

    Psi_dual = build_dual_matrix(dual_modes, selected_modes, node_order, projection_dofs)
    dual_coeffs = np.array([dual_modes[m].coeff for m in selected_modes], dtype=float)

    q, *_ = np.linalg.lstsq(Phi, u_fe, rcond=None)

    for _ in range(40):
        u_rec = dual_manifold_displacement(q, Phi, Psi_dual, dual_coeffs)
        r = u_rec - u_fe

        J = Phi.copy()
        for i in range(len(selected_modes)):
            J[:, i] += 2.0 * q[i] * dual_coeffs[i] * Psi_dual[:, i]

        try:
            dq = np.linalg.solve(J.T @ J, -(J.T @ r))
        except np.linalg.LinAlgError:
            dq, *_ = np.linalg.lstsq(J, -r, rcond=None)

        q = q + dq
        if float(np.linalg.norm(dq)) < 1.0e-12:
            break

    u_rec = dual_manifold_displacement(q, Phi, Psi_dual, dual_coeffs)
    rel = float(np.linalg.norm(u_fe - u_rec) / max(np.linalg.norm(u_fe), 1.0e-30))
    return q, rel


def reconstruct_w_value(
    q: np.ndarray,
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    normal_dof: str,
    thickness: float,
    use_dual: bool,
) -> float:
    common_nodes = set(modes[selected_modes[0]].node_values.keys())
    for m in selected_modes[1:]:
        common_nodes &= set(modes[m].node_values.keys())
    if use_dual:
        for m in selected_modes:
            common_nodes &= set(dual_modes[m].node_values.keys())

    wmax = 0.0
    for node in common_nodes:
        val = 0.0
        for i, mode_num in enumerate(selected_modes):
            val += q[i] * modes[mode_num].node_values[node][normal_dof]
        if use_dual:
            for i, mode_num in enumerate(selected_modes):
                dm = dual_modes[mode_num]
                val += (q[i] ** 2) * dm.coeff * dm.node_values[node].get(normal_dof, 0.0)
        wmax = max(wmax, abs(val))
    return wmax / thickness


def residual_static_rom(q: np.ndarray, qext: np.ndarray, K: np.ndarray, G: np.ndarray, H: np.ndarray) -> np.ndarray:
    return K @ q + evaluate_canonical_gamma(q, G, H) - qext


def finite_difference_jacobian(fun, q: np.ndarray, eps: float) -> np.ndarray:
    n = len(q)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        dq = np.zeros(n, dtype=float)
        dq[i] = eps
        fp = fun(q + dq)
        fm = fun(q - dq)
        J[:, i] = (fp - fm) / (2.0 * eps)
    return J


def solve_rom_path_from_force_path(
    fe_states: Sequence[FEState],
    qext_path: np.ndarray,
    K: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    modes: Dict[int, ModeData],
    dual_modes: Dict[int, DualModeData],
    selected_modes: Sequence[int],
    cfg: dict,
) -> List[dict]:
    q_prev = np.zeros(len(selected_modes), dtype=float)
    out = []
    use_dual = bool(cfg.get("use_dual_modes", True))

    for step, qext in zip(fe_states, qext_path):
        q = q_prev.copy()

        def fun(x):
            return residual_static_rom(x, qext, K, G, H)

        converged = False
        for _ in range(int(cfg["newton_max_iter"])):
            r = fun(q)
            if float(np.linalg.norm(r)) <= float(cfg["newton_tol"]):
                converged = True
                break
            J = finite_difference_jacobian(fun, q, float(cfg["fd_eps"]))
            try:
                dq = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, -r, rcond=None)
            q = q + dq
            if float(np.linalg.norm(dq)) <= float(cfg["newton_tol"]):
                converged = True
                break

        q_prev = q.copy()
        out.append({
            "set_id": step.index,
            "load_pct": step.load_pct,
            "time_value": step.time_value,
            "q_rom": q.copy(),
            "w_value_rom": reconstruct_w_value(
                q=q,
                modes=modes,
                dual_modes=dual_modes,
                selected_modes=selected_modes,
                normal_dof=str(cfg["normal_dof"]).upper(),
                thickness=float(cfg["thickness"]),
                use_dual=use_dual,
            ),
            "converged": converged,
        })

    return out


def write_step_summary_csv(path: Path, experiments: Sequence[StepExperiment], selected_modes: Sequence[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        header = ["experiment", "family", "q_abs", "sign_code"] + [f"q_m{m}" for m in selected_modes]
        header += [f"F_lin_m{m}" for m in selected_modes] + [f"Gamma_m{m}" for m in selected_modes]
        w.writerow(header)

        for exp in sorted(experiments, key=lambda e: e.name_base):
            row = [
                exp.name_base,
                exp.family,
                fmt(exp.q_abs),
                json.dumps(exp.sign_code, ensure_ascii=False),
            ]
            row += [fmt(x) for x in exp.q_vector]
            row += [fmt(x) for x in (exp.lin_force if exp.lin_force is not None else np.full(len(selected_modes), np.nan))]
            row += [fmt(x) for x in (exp.gamma_force if exp.gamma_force is not None else np.full(len(selected_modes), np.nan))]
            w.writerow(row)


def write_curve_csv(
    path: Path,
    fe_states: Sequence[FEState],
    fe_qs: Sequence[np.ndarray],
    fe_proj_errs: Sequence[float],
    qext_path: np.ndarray,
    rom_path: Sequence[dict],
) -> None:
    rom_map = {p["set_id"]: p for p in rom_path}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "set_id", "time", "load_pct",
            "fe_w", "fe_proj_rel_l2_err",
            "fe_q_vector",
            "qext_vector",
            "rom_q_vector", "rom_w", "rom_converged"
        ])
        for step, q_fe, err, qext in zip(fe_states, fe_qs, fe_proj_errs, qext_path):
            rp = rom_map[step.index]
            w.writerow([
                step.index,
                fmt(step.time_value),
                fmt(step.load_pct),
                fmt(step.w_value),
                fmt(err),
                json.dumps([float(x) for x in q_fe], ensure_ascii=False),
                json.dumps([float(x) for x in qext], ensure_ascii=False),
                json.dumps([float(x) for x in rp["q_rom"]], ensure_ascii=False),
                fmt(rp["w_value_rom"]),
                str(bool(rp["converged"])),
            ])


def make_plot(path: Path, fe_states: Sequence[FEState], rom_path: Sequence[dict], title: str) -> None:
    x_fe = [step.load_pct for step in fe_states]
    y_fe = [step.w_value for step in fe_states]

    x_rom = [p["load_pct"] for p in rom_path]
    y_rom = [p["w_value_rom"] for p in rom_path]

    plt.figure(figsize=(10, 6))
    plt.plot(x_fe, y_fe, marker="o", linewidth=2, label="FE nonlinear static")
    plt.plot(x_rom, y_rom, marker="^", linewidth=2, label="STEP ROM")
    plt.xlabel("Load factor, %")
    plt.ylabel("W / h")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_fe_plot(path: Path, fe_states: Sequence[FEState], title: str) -> None:
    x_fe = [step.load_pct for step in fe_states]
    y_fe = [step.w_value for step in fe_states]

    plt.figure(figsize=(10, 6))
    plt.plot(x_fe, y_fe, marker="o", linewidth=2, label="FE nonlinear static")
    plt.xlabel("Load factor, %")
    plt.ylabel("W / h")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_rom_plot(path: Path, rom_path: Sequence[dict], title: str) -> None:
    x_rom = [p["load_pct"] for p in rom_path]
    y_rom = [p["w_value_rom"] for p in rom_path]

    plt.figure(figsize=(10, 6))
    plt.plot(x_rom, y_rom, marker="^", linewidth=2, label="STEP ROM")
    plt.xlabel("Load factor, %")
    plt.ylabel("W / h")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def process_bc(case_dir: Path, cfg: dict) -> Dict[str, str]:
    selected_modes = list(cfg["selected_modes"])
    modal_dir = case_dir / cfg["modal_dir_name"]
    reactions_dir = case_dir / cfg["reactions_dir_name"]
    rom_dir = case_dir / cfg["rom_dir_name"]
    rom_dir.mkdir(parents=True, exist_ok=True)

    modes = discover_modes(modal_dir, selected_modes)

    local_cfg = dict(cfg)
    resolved_normal_dof, normal_meta = resolve_normal_dof(local_cfg, modes, selected_modes)
    local_cfg["normal_dof"] = resolved_normal_dof

    dual_modes = build_dual_modes(case_dir, modes, selected_modes, local_cfg)

    experiments_map = discover_step_experiments(
        reactions_dir=reactions_dir,
        modes=modes,
        selected_modes=selected_modes,
        reaction_sign=float(cfg["reaction_sign"]),
        use_rotations=bool(cfg["use_rotations_in_generalized_force"]),
    )
    experiments = list(experiments_map.values())

    step_coeffs = extract_canonical_step_coefficients(experiments_map, selected_modes)
    K = step_coeffs["K"]
    G = step_coeffs["G"]
    H = step_coeffs["H"]

    summary_csv = rom_dir / cfg["step_summary_csv_filename"]
    coeff_json = rom_dir / cfg["coefficients_json_filename"]
    report_txt = rom_dir / cfg["report_filename"]
    curve_csv = rom_dir / cfg["curve_csv_filename"]
    curve_png = rom_dir / cfg["plot_filename"]
    fe_curve_png = rom_dir / str(cfg.get("fe_plot_filename", "fe_static_curve.png"))
    rom_curve_png = rom_dir / str(cfg.get("rom_plot_filename", "rom_static_curve.png"))

    write_step_summary_csv(summary_csv, experiments, selected_modes)

    fe_states: List[FEState] = []
    fe_qs: List[np.ndarray] = []
    fe_proj_errs: List[float] = []
    rom_path: List[dict] = []
    qext_path: Optional[np.ndarray] = None
    qext_source: Optional[str] = None
    qext_source = "not_used"

    if bool(cfg.get("compare_with_fe", True)):
        fe_job_dir = case_dir / str(cfg["fe_reference_dir_name"]) / str(cfg["fe_reference_job_name"])
        if not fe_job_dir.exists():
            raise FileNotFoundError(f"Не найдена папка FE reference job: {fe_job_dir}")

        fe_states = build_fe_states_from_d3plot(fe_job_dir, local_cfg)
        fe_times = np.array([s.time_value for s in fe_states], dtype=float)

        q_prev = np.zeros(len(selected_modes), dtype=float)
        for step in fe_states:
            q_fe, rel_err = fit_fe_q_vector(
                fe_field=step.field,
                modes=modes,
                dual_modes=dual_modes,
                selected_modes=selected_modes,
                cfg=local_cfg,
                q_prev=q_prev,
            )
            fe_qs.append(q_fe)
            fe_proj_errs.append(rel_err)
            q_prev = q_fe.copy()


        if str(local_cfg.get("fe_generalized_force_source", "auto")).lower() == "auto":
            if fe_proj_errs and float(np.mean(fe_proj_errs)) > float(local_cfg.get("fe_projection_error_threshold", 0.5)):
                local_cfg["fe_generalized_force_source"] = "projected_equilibrium"

        qext_path, qext_source = build_fe_generalized_force_path(
            fe_job_dir=fe_job_dir,
            modes=modes,
            dual_modes=dual_modes,
            selected_modes=selected_modes,
            cfg=local_cfg,
            target_times=fe_times,
            fe_qs=fe_qs,
            K=K,
            G=G,
            H=H,
        )

        rom_path = solve_rom_path_from_force_path(
            fe_states=fe_states,
            qext_path=qext_path,
            K=K,
            G=G,
            H=H,
            modes=modes,
            dual_modes=dual_modes,
            selected_modes=selected_modes,
            cfg=local_cfg,
        )

        write_curve_csv(curve_csv, fe_states, fe_qs, fe_proj_errs, qext_path, rom_path)
        make_plot(curve_png, fe_states, rom_path, str(cfg["plot_title"]))
        make_fe_plot(fe_curve_png, fe_states, str(cfg.get("fe_plot_title", "FE nonlinear static displacement curve")))
        make_rom_plot(rom_curve_png, rom_path, str(cfg.get("rom_plot_title", "STEP ROM static displacement curve")))

    dual_info = {
        str(m): {
            "raw_norm": dual_modes[m].raw_norm,
            "coeff": dual_modes[m].coeff,
            "p_value": dual_modes[m].p_value,
        }
        for m in dual_modes
    }
    coeff_json.write_text(
        json.dumps({
            "selected_modes": list(selected_modes),
            "normal_dof_requested": cfg.get("normal_dof", "auto"),
            "normal_dof_resolved": resolved_normal_dof,
            "normal_dof_meta": normal_meta,
            "d3plot_reference_meta": local_cfg.get("_d3plot_reference_meta", {}),
            "force_path_diagnostics": local_cfg.get("_force_path_diagnostics", []),
            "K": K.tolist(),
            "G": G.tolist(),
            "H": H.tolist(),
            "dual_modes": dual_info,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    lines: List[str] = []
    lines.append("Static dual-enhanced canonical STEP postprocessor")
    lines.append("=" * 100)
    lines.append(f"CASE DIR                                = {case_dir}")
    lines.append(f"SELECTED MODES                          = {selected_modes}")
    lines.append(f"REACTIONS DIR                           = {reactions_dir}")
    lines.append(f"MODAL DIR                               = {modal_dir}")
    lines.append(f"use_rotations_in_generalized_force      = {bool(cfg['use_rotations_in_generalized_force'])}")
    lines.append(f"use_dual_modes                          = {bool(cfg['use_dual_modes'])}")
    lines.append(f"d3plot_backend                          = {cfg['d3plot_backend']}")
    ref_meta = local_cfg.get("_d3plot_reference_meta", {})
    if ref_meta:
        lines.append(f"d3plot_reference_state                 = {ref_meta.get('mode')}")
        lines.append(f"d3plot_reference_enabled               = {ref_meta.get('enabled')}")
        lines.append(f"d3plot_reference_index                 = {ref_meta.get('reference_index')}")
        lines.append(f"d3plot_reference_time                  = {ref_meta.get('reference_time')}")
        lines.append(f"d3plot_reference_max_abs_component     = {ref_meta.get('max_abs_reference_component')}")
    lines.append(f"fe_generalized_force_source            = {qext_source}")
    lines.append(f"fe_q_fit_space                        = {cfg.get('fe_q_fit_space', 'primary_normal')}")
    lines.append(f"normal_dof_requested                  = {cfg.get('normal_dof', 'auto')}")
    lines.append(f"normal_dof_resolved                   = {resolved_normal_dof}")
    lines.append(f"normal_dof_source                     = {normal_meta.get('source')}")
    if normal_meta.get("modal_votes") is not None:
        lines.append(f"modal_votes                           = {normal_meta['modal_votes']}")
    if normal_meta.get("modal_scores") is not None:
        lines.append(f"modal_scores                          = {normal_meta['modal_scores']}")
    if normal_meta.get("geometry_spans") is not None:
        lines.append(f"geometry_spans                        = {normal_meta['geometry_spans']}")
    if normal_meta.get("flat_ratio") is not None:
        lines.append(f"flat_ratio                            = {normal_meta['flat_ratio']:.12e}")
    lines.append("")

    lines.append("IMPORTANT")
    lines.append("-" * 100)
    lines.append("1) Коэффициенты K,G,H извлечены по канонической STEP-логике только для transverse modes.")
    lines.append("2) Dual / in-plane modes построены из d3plot-пар dual_mode_i_p и dual_mode_i_2p как w(2p)-2*w(p).")
    lines.append("3) Dual / in-plane fields additionally cleaned from uniform in-plane drift and linear projection onto the primary basis.")
    lines.append("4) Статическая ROM решается по transverse STEP-уравнениям с внешней generalized force path.")
    lines.append("   Для FE nodal-force files generalized force проецируется на tangent basis du/dq, а не только на Phi.")
    lines.append("   Источник пути нагрузки: bndout, а если его не удалось распарсить — projected_equilibrium")
    lines.append("   по FE-проекции q_FE(t). Low-load calibration не используется.")
    lines.append("5) Для полного 6-mode STEP на расширенном базисе потребовались бы отдельные STEP-эксперименты")
    lines.append("   и для dual basis functions.")
    lines.append("")

    lines.append("MODES")
    lines.append("-" * 100)
    for m in selected_modes:
        md = modes[m]
        lines.append(
            f"mode={m:3d}  freq={md.freq_hz:.8f} Hz  nodes={len(md.node_values):6d}  "
            f"dom_dof={md.dom_dof}  ref_node={md.ref_node}  norm={md.normalization_value:.12e}"
        )
    lines.append("")

    if dual_modes:
        lines.append("DUAL MODES")
        lines.append("-" * 100)
        for m in selected_modes:
            dm = dual_modes[m]
            lines.append(
                f"dual({m}) raw_norm={dm.raw_norm:.12e}  coeff={dm.coeff:.12e}  p={dm.p_value:.12e}"
            )
        lines.append("")

    lines.append("LINEAR MATRIX K")
    lines.append("-" * 100)
    for i, mi in enumerate(selected_modes):
        row = "  ".join(f"{K[i, j]: .6e}" for j in range(len(selected_modes)))
        lines.append(f"row for output mode {mi}: {row}")
    lines.append("")

    if fe_states:
        lines.append("FE / ROM COMPARISON")
        lines.append("-" * 100)
        lines.append(f"fe_state_count                          = {len(fe_states)}")
        lines.append(f"max FE projection rel L2 error         = {max(fe_proj_errs):.12e}")
        lines.append(f"mean FE projection rel L2 error        = {float(np.mean(fe_proj_errs)):.12e}")
        lines.append(f"ROM converged steps                    = {sum(1 for p in rom_path if p['converged'])}")
        lines.append(f"FE generalized force source            = {qext_source}")
        if local_cfg.get("_force_path_diagnostics"):
            lines.append("force_path_diagnostics                 =")
            for diag in local_cfg.get("_force_path_diagnostics", []):
                lines.append(f"  - {diag}")
        lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(f"summary csv                            = {summary_csv}")
    lines.append(f"coefficients json                      = {coeff_json}")
    if fe_states:
        lines.append(f"curve csv                              = {curve_csv}")
        lines.append(f"curve png                              = {curve_png}")
        lines.append(f"fe curve png                           = {fe_curve_png}")
        lines.append(f"rom curve png                          = {rom_curve_png}")
    lines.append(f"report txt                             = {report_txt}")
    lines.append("")
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "summary_csv": str(summary_csv),
        "coeff_json": str(coeff_json),
        "report_txt": str(report_txt),
    }
    if fe_states:
        result["curve_csv"] = str(curve_csv)
        result["curve_png"] = str(curve_png)
        result["fe_curve_png"] = str(fe_curve_png)
        result["rom_curve_png"] = str(rom_curve_png)
        result["qext_source"] = str(qext_source)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Static dual-enhanced canonical STEP postprocessor using STEP reactions + dual d3plot + FE reference d3plot/bndout."
    )
    parser.add_argument(
        "--config",
        default="4_Generate_ROM.json",
        help="JSON-конфиг рядом со скриптом."
    )
    args = parser.parse_args()

    base_dir = script_dir()
    cfg = load_config(base_dir / args.config)

    exports_root = Path(cfg["exports_root"])
    if not exports_root.is_absolute():
        exports_root = (base_dir / exports_root).resolve()

    bc_count = int(cfg["bc_count"])
    results = {}

    for i in range(1, bc_count + 1):
        bc_name = str(cfg["bc_name_pattern"]).format(index=i)
        case_dir = exports_root / bc_name
        if not case_dir.exists():
            raise FileNotFoundError(f"Не найдена папка случая: {case_dir}")
        print(f"[INFO] Processing {bc_name}")
        results[bc_name] = process_bc(case_dir, cfg)

    summary_path = exports_root / "step_rom_dual_static_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
