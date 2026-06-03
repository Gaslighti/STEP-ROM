from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import builtins
import sys


def _safe_console_text(value) -> str:
    text = str(value)

    # Удаляем управляющие символы, которые иногда ломают Windows/PyCharm console.
    text = text.replace("\x00", "")
    text = "".join(
        ch if ch in "\n\r\t" or ord(ch) >= 32 else "?"
        for ch in text
    )

    # Одинокие surrogate-символы и невалидные символы превращаем в безопасный текст.
    try:
        text = text.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="replace")
    except Exception:
        text = repr(text)

    return text


def safe_print(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    text = sep.join(_safe_console_text(arg) for arg in args) + end

    try:
        builtins.print(text, end="", flush=flush)
        return
    except (OSError, UnicodeEncodeError, ValueError):
        pass

    # Если консоль PyCharm/Windows отвалилась, не останавливаем LS-DYNA batch.
    # Пишем сообщение в fallback-log рядом со скриптом.
    try:
        fallback_path = Path(__file__).resolve().with_name("3_Ls_dyna_run_configs_console_fallback.log")
        with fallback_path.open("a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(text)
    except Exception:
        pass


DEFAULT_CONFIG = "3_Ls_dyna_run_configs.json"


@dataclass
class Job:
    bc_name: str
    case_dir: Path
    name: str
    source_k: Path
    job_dir: Path
    job_k: Path
    origin_dir: Path
    job_type: str               # "step" | "dual" | "real"
    step_dir: Optional[Path] = None
    run_bat: Optional[Path] = None


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Файл конфигурации должен содержать JSON-объект: {config_path}")

    return cfg


def windows_quote(arg: str) -> str:
    arg = str(arg)
    if any(ch in arg for ch in ' \t"&()[]{}=;,+!^'):
        return f'"{arg}"'
    return arg


def ensure_clean_dir(path_obj: Path, overwrite: bool) -> None:
    if path_obj.exists():
        if overwrite:
            shutil.rmtree(path_obj)
        else:
            raise FileExistsError(f"Папка уже существует: {path_obj}")
    path_obj.mkdir(parents=True, exist_ok=True)


def discover_bc_case_dirs(cfg: dict, base_dir: Path) -> List[tuple[str, Path, Path]]:
    exports_root = (base_dir / cfg["exports_root"]).resolve()
    if not exports_root.exists():
        raise FileNotFoundError(f"Не найдена папка exports_root: {exports_root}")

    bc_count = int(cfg["bc_count"])
    if bc_count < 1:
        raise ValueError("bc_count должен быть >= 1")

    out: List[tuple[str, Path, Path]] = []
    for i in range(1, bc_count + 1):
        bc_name = str(cfg["bc_name_pattern"]).format(index=i)
        case_dir = exports_root / bc_name
        step_dir = case_dir / cfg["step_subdir_name"]
        if not case_dir.exists():
            raise FileNotFoundError(f"Не найдена папка случая {bc_name}: {case_dir}")
        if not step_dir.exists():
            raise FileNotFoundError(f"Не найдена папка step для {bc_name}: {step_dir}")
        out.append((bc_name, case_dir, step_dir))
    return out


def discover_kfiles(step_dir: Path, file_glob: str) -> List[Path]:
    return sorted(
        [p for p in step_dir.glob(file_glob) if p.is_file()],
        key=lambda p: p.name.lower()
    )


def build_command(cfg: dict, job: Job) -> List[str]:
    solver_exe = str(cfg["solver_exe"])
    solver_args = cfg.get("solver_args", [])
    if not solver_exe:
        raise ValueError("В config не задан solver_exe")

    values = {
        "bc_name": job.bc_name,
        "jobname": job.name,
        "jobdir": str(job.job_dir),
        "kfile": str(job.job_k),
        "kfile_name": job.job_k.name,
        "kfile_stem": job.job_k.stem,
    }

    cmd = [solver_exe]
    for arg in solver_args:
        cmd.append(str(arg).format(**values))
    return cmd


def create_run_bat(cfg: dict, job: Job) -> Path:
    bat_path = job.job_dir / cfg["run_bat_name"]
    command = build_command(cfg, job)

    lines = [
        "@echo off",
        "setlocal",
        f'cd /d "{job.job_dir}"',
        " ".join(windows_quote(a) for a in command),
    ]
    if bool(cfg.get("bat_pause_on_error", False)):
        lines += [
            "if errorlevel 1 (",
            "  echo Solver returned errorlevel %errorlevel%",
            "  pause",
            ")"
        ]
    lines.append("exit /b %errorlevel%")

    bat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bat_path


def classify_step_job(cfg: dict, kfile: Path) -> str:
    prefix = str(cfg.get("dual_job_prefix", "dual_mode")).lower()
    stem = kfile.stem.lower()
    if stem.startswith(prefix):
        return "dual"
    return "step"


def prepare_one_job_dir(
    cfg: dict,
    *,
    bc_name: str,
    case_dir: Path,
    source_k: Path,
    job_root: Path,
    job_type: str,
    step_dir: Optional[Path],
    copy_instead_of_move: bool,
) -> Job:
    job_name = source_k.stem
    job_dir = job_root / job_name
    ensure_clean_dir(job_dir, overwrite=bool(cfg["overwrite_job_dirs"]))

    job_k = job_dir / source_k.name
    if copy_instead_of_move:
        shutil.copy2(str(source_k), str(job_k))
    else:
        if str(cfg["organize_mode"]).lower() == "move":
            shutil.move(str(source_k), str(job_k))
        else:
            shutil.copy2(str(source_k), str(job_k))

    job = Job(
        bc_name=bc_name,
        case_dir=case_dir,
        name=job_name,
        source_k=source_k,
        job_dir=job_dir,
        job_k=job_k,
        origin_dir=source_k.parent,
        job_type=job_type,
        step_dir=step_dir,
    )

    job.run_bat = create_run_bat(cfg, job)

    manifest = {
        "bc_name": job.bc_name,
        "job_name": job.name,
        "job_type": job.job_type,
        "source_k": str(job.source_k),
        "job_dir": str(job.job_dir),
        "job_k": str(job.job_k),
        "run_bat": str(job.run_bat),
        "solver_exe": str(cfg["solver_exe"]),
        "solver_args": cfg.get("solver_args", []),
    }
    (job_dir / cfg["manifest_name"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return job


def organize_jobs(cfg: dict, base_dir: Path) -> List[Job]:
    all_jobs: List[Job] = []

    for bc_name, case_dir, step_dir in discover_bc_case_dirs(cfg, base_dir):
        # STEP + DUAL jobs from exports/bc_xx/step
        step_job_root = step_dir / cfg["job_root_name"]
        step_job_root.mkdir(parents=True, exist_ok=True)

        kfiles = discover_kfiles(step_dir, cfg["file_glob"])
        if not kfiles:
            safe_print(f"[WARN] В {step_dir} не найдено .k файлов")
        else:
            for kfile in kfiles:
                job_type = classify_step_job(cfg, kfile)
                job = prepare_one_job_dir(
                    cfg,
                    bc_name=bc_name,
                    case_dir=case_dir,
                    source_k=kfile,
                    job_root=step_job_root,
                    job_type=job_type,
                    step_dir=step_dir,
                    copy_instead_of_move=False,
                )
                all_jobs.append(job)


        if bool(cfg.get("process_real_load_key", True)):
            real_k = case_dir / str(cfg.get("real_load_filename", "real_load.key"))
            if real_k.exists() and real_k.is_file():
                real_job_root = case_dir / str(cfg.get("real_job_root_name", "real_jobs"))
                real_job_root.mkdir(parents=True, exist_ok=True)
                job = prepare_one_job_dir(
                    cfg,
                    bc_name=bc_name,
                    case_dir=case_dir,
                    source_k=real_k,
                    job_root=real_job_root,
                    job_type="real",
                    step_dir=None,
                    copy_instead_of_move=bool(cfg.get("copy_real_load_key_instead_of_move", True)),
                )
                all_jobs.append(job)
            else:
                safe_print(f"[WARN] Для {bc_name} не найден real_load.key: {real_k}")

    if not all_jobs:
        raise RuntimeError("Не удалось подготовить ни одной job-папки.")
    return all_jobs


def discover_existing_jobs(cfg: dict, base_dir: Path) -> List[Job]:
    all_jobs: List[Job] = []

    for bc_name, case_dir, step_dir in discover_bc_case_dirs(cfg, base_dir):
        step_job_root = step_dir / cfg["job_root_name"]
        if step_job_root.exists():
            for job_dir in sorted([p for p in step_job_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
                kfiles = sorted(job_dir.glob("*.k"))
                if not kfiles:
                    continue
                job_type = classify_step_job(cfg, kfiles[0])
                run_bat = job_dir / cfg["run_bat_name"]
                all_jobs.append(
                    Job(
                        bc_name=bc_name,
                        case_dir=case_dir,
                        name=job_dir.name,
                        source_k=kfiles[0],
                        job_dir=job_dir,
                        job_k=kfiles[0],
                        origin_dir=job_dir,
                        job_type=job_type,
                        step_dir=step_dir,
                        run_bat=run_bat if run_bat.exists() else None,
                    )
                )
        else:
            safe_print(f"[WARN] Для {bc_name} не найдена папка jobs: {step_job_root}")

        if bool(cfg.get("process_real_load_key", True)):
            real_job_root = case_dir / str(cfg.get("real_job_root_name", "real_jobs"))
            if real_job_root.exists():
                for job_dir in sorted([p for p in real_job_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
                    kfiles = sorted(job_dir.glob("*.k"))
                    if not kfiles:
                        continue
                    run_bat = job_dir / cfg["run_bat_name"]
                    all_jobs.append(
                        Job(
                            bc_name=bc_name,
                            case_dir=case_dir,
                            name=job_dir.name,
                            source_k=kfiles[0],
                            job_dir=job_dir,
                            job_k=kfiles[0],
                            origin_dir=job_dir,
                            job_type="real",
                            step_dir=None,
                            run_bat=run_bat if run_bat.exists() else None,
                        )
                    )
            else:
                safe_print(f"[WARN] Для {bc_name} не найдена папка real jobs: {real_job_root}")

    if not all_jobs:
        raise RuntimeError("Не найдено готовых job-папок для запуска.")
    return all_jobs


def copy_payload_by_patterns(
    *,
    source_dir: Path,
    target_dir: Path,
    patterns: List[str],
    overwrite: bool,
) -> Dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    seen = set()

    for pattern in patterns:
        for src in sorted(source_dir.glob(pattern), key=lambda p: p.name.lower()):
            if not src.is_file():
                continue
            if src.name in seen:
                continue
            seen.add(src.name)

            dst = target_dir / src.name
            if dst.exists():
                if overwrite:
                    dst.unlink()
                else:
                    raise FileExistsError(f"Файл уже существует: {dst}")

            shutil.copy2(str(src), str(dst))
            copied[src.name] = str(dst)

    return copied


def export_step_reaction_file(cfg: dict, job: Job) -> Dict[str, str]:
    if not bool(cfg.get("export_reactions", True)):
        return {}
    if job.step_dir is None:
        return {}

    reactions_dir = job.case_dir / cfg["reactions_dir_name"]
    reactions_dir.mkdir(parents=True, exist_ok=True)

    source_name = str(cfg.get("reaction_source_file", "bndout"))
    src = job.job_dir / source_name
    if not src.exists() or not src.is_file():
        return {}

    out_name = str(cfg.get("reaction_filename_pattern", "{jobname}.txt")).format(
        bc_name=job.bc_name,
        jobname=job.name,
        source_name=source_name,
    )
    dst = reactions_dir / out_name

    if dst.exists():
        if bool(cfg.get("overwrite_reaction_files", True)):
            dst.unlink()
        else:
            raise FileExistsError(f"Файл реакции уже существует: {dst}")

    shutil.copy2(str(src), str(dst))
    exported = {source_name: str(dst)}

    if bool(cfg.get("write_reactions_manifest", True)):
        manifest_path = reactions_dir / f"{job.name}__reaction_manifest.json"
        manifest = {
            "bc_name": job.bc_name,
            "job_name": job.name,
            "job_type": job.job_type,
            "job_dir": str(job.job_dir),
            "reaction_source_file": source_name,
            "exported_file": str(dst),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    return exported


def export_dual_payload(cfg: dict, job: Job) -> Dict[str, str]:
    dual_dir = job.case_dir / str(cfg.get("dual_results_dir_name", "dual_fields")) / job.name
    copied = copy_payload_by_patterns(
        source_dir=job.job_dir,
        target_dir=dual_dir,
        patterns=list(cfg.get("dual_export_patterns", [])),
        overwrite=bool(cfg.get("overwrite_export_payload", True)),
    )

    if copied and bool(cfg.get("write_dual_manifest", True)):
        manifest = {
            "bc_name": job.bc_name,
            "job_name": job.name,
            "job_type": job.job_type,
            "job_dir": str(job.job_dir),
            "export_dir": str(dual_dir),
            "files": copied,
        }
        (dual_dir / "dual_export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    return copied


def export_fe_payload(cfg: dict, job: Job) -> Dict[str, str]:
    fe_dir = job.case_dir / str(cfg.get("fe_reference_dir_name", "fe_reference")) / job.name
    copied = copy_payload_by_patterns(
        source_dir=job.job_dir,
        target_dir=fe_dir,
        patterns=list(cfg.get("fe_export_patterns", [])),
        overwrite=bool(cfg.get("overwrite_export_payload", True)),
    )

    if copied and bool(cfg.get("write_fe_manifest", True)):
        manifest = {
            "bc_name": job.bc_name,
            "job_name": job.name,
            "job_type": job.job_type,
            "job_dir": str(job.job_dir),
            "export_dir": str(fe_dir),
            "files": copied,
        }
        (fe_dir / "fe_export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    return copied


def run_one_job(cfg: dict, job: Job) -> dict:
    stdout_path = job.job_dir / cfg["log_stdout_name"]
    stderr_path = job.job_dir / cfg["log_stderr_name"]

    if bool(cfg.get("launch_via_bat", True)):
        if job.run_bat is None or not job.run_bat.exists():
            raise FileNotFoundError(f"Не найден bat для job {job.name}: {job.run_bat}")
        command = ["cmd", "/c", str(job.run_bat.name)]
        cwd = job.job_dir
    else:
        command = build_command(cfg, job)
        cwd = job.job_dir if str(cfg.get("working_dir_mode", "job_dir")).lower() == "job_dir" else job.job_k.parent

    start_time = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="ignore") as f_out, \
         stderr_path.open("w", encoding="utf-8", errors="ignore") as f_err:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=f_out,
            stderr=f_err,
            shell=False,
        )
        try:
            return_code = proc.wait(timeout=cfg["timeout_sec"])
            status = "finished"
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = -999
            status = "timeout"

    elapsed = time.time() - start_time

    exported_reactions = {}
    exported_dual = {}
    exported_fe = {}

    if status == "finished" and return_code == 0:
        if job.job_type == "step":
            exported_reactions = export_step_reaction_file(cfg, job)
        elif job.job_type == "dual":
            exported_dual = export_dual_payload(cfg, job)
        elif job.job_type == "real":
            exported_fe = export_fe_payload(cfg, job)

    result = {
        "bc_name": job.bc_name,
        "job_name": job.name,
        "job_type": job.job_type,
        "job_dir": str(job.job_dir),
        "job_k": str(job.job_k),
        "command": command,
        "return_code": return_code,
        "status": status,
        "elapsed_sec": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "exported_reactions": exported_reactions,
        "exported_dual_files": exported_dual,
        "exported_fe_files": exported_fe,
    }

    (job.job_dir / "run_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return result


def run_jobs(cfg: dict, jobs: List[Job]) -> List[dict]:
    if bool(cfg.get("only_organize", False)):
        return []

    max_parallel = int(cfg.get("max_parallel", 1))
    if max_parallel < 1:
        raise ValueError("max_parallel должен быть >= 1")

    results: List[dict] = []

    if max_parallel == 1:
        for job in jobs:
            safe_print(f"[RUN] {job.bc_name} / {job.job_type} / {job.name}")
            result = run_one_job(cfg, job)
            safe_print(f"[DONE] {job.bc_name} / {job.job_type} / {job.name}: {result['status']} rc={result['return_code']}")
            if result["exported_reactions"]:
                safe_print(f"[REACTION] {job.bc_name} / {job.name}: {list(result['exported_reactions'].values())[0]}")
            if result["exported_dual_files"]:
                safe_print(f"[DUAL] {job.bc_name} / {job.name}: {len(result['exported_dual_files'])} files")
            if result["exported_fe_files"]:
                safe_print(f"[FE] {job.bc_name} / {job.name}: {len(result['exported_fe_files'])} files")
            results.append(result)
        return results

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {ex.submit(run_one_job, cfg, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            result = fut.result()
            safe_print(f"[DONE] {job.bc_name} / {job.job_type} / {job.name}: {result['status']} rc={result['return_code']}")
            if result["exported_reactions"]:
                safe_print(f"[REACTION] {job.bc_name} / {job.name}: {list(result['exported_reactions'].values())[0]}")
            if result["exported_dual_files"]:
                safe_print(f"[DUAL] {job.bc_name} / {job.name}: {len(result['exported_dual_files'])} files")
            if result["exported_fe_files"]:
                safe_print(f"[FE] {job.bc_name} / {job.name}: {len(result['exported_fe_files'])} files")
            results.append(result)

    return results


def write_summary(cfg: dict, base_dir: Path, results: List[dict]) -> None:
    if not results:
        return

    exports_root = (base_dir / cfg["exports_root"]).resolve()
    summary = {
        "job_count": len(results),
        "finished": sum(1 for r in results if r["status"] == "finished" and r["return_code"] == 0),
        "failed": sum(1 for r in results if not (r["status"] == "finished" and r["return_code"] == 0)),
        "results": results,
    }
    (exports_root / "lsrun_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Сортировка STEP/DUAL/REAL K-файлов по отдельным папкам, запуск LS-DYNA через BAT, экспорт reactions/dual/fe payload."
    )
    parser.add_argument(
        "--config",
        default="3_Ls_dyna_run_configs.json",
        help="Имя JSON-конфига рядом со скриптом."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir / args.config)

    if bool(cfg.get("only_run_existing_jobs", False)):
        jobs = discover_existing_jobs(cfg, base_dir)
    else:
        jobs = organize_jobs(cfg, base_dir)

    safe_print(f"[INFO] Jobs prepared/found: {len(jobs)}")
    results = run_jobs(cfg, jobs)
    write_summary(cfg, base_dir, results)
    safe_print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
