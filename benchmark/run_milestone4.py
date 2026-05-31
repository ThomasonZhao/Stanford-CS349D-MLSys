"""Run Milestone 4 HiCache benchmark experiments.

This runner launches one MiniEngine server at a time, waits for /health, runs
the cache benchmark client, captures both server and client logs, snapshots
/cache_stats, and shuts the server down before moving to the next experiment.

The required HiCache demonstration is:

1. GPU-only radix cache: show multiturn hit-rate collapse once the working set
   exceeds the GPU KV pool.
2. HiCache: rerun the exact same workload with a CPU tier and show the
   per-turn hit rate stays high.
3. Verify the workload completes and the cache stats show demotion/promotion.

Start with:

    python -m benchmark.run_milestone4 --suite smoke

Then tune the required workload if needed:

    python -m benchmark.run_milestone4 --suite required \
        --num-sessions 64 --turns-per-session 8 --concurrency 16 \
        --max-tokens 32 --cpu-cache-size-gb 32 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ClientRun:
    name: str
    cmd: list[str]


@dataclass
class Experiment:
    name: str
    description: str
    server_extra: list[str] = field(default_factory=list)
    clients: list[ClientRun] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Milestone 4 HiCache benchmarks")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=512)
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--cpu-cache-size-gb", type=float, default=16.0)
    p.add_argument(
        "--smoke-cpu-cache-size-gb",
        type=float,
        default=1.0,
        help="HiCache CPU tier size for --suite smoke.",
    )
    p.add_argument("--server-timeout-sec", type=int, default=900)
    p.add_argument("--logs-root", default="logs/milestone4")
    p.add_argument(
        "--suite",
        choices=["smoke", "required", "overlap", "accuracy", "all"],
        default="required",
        help="smoke validates plumbing; required runs GPU-only vs HiCache; "
        "overlap adds --hicache-overlap; accuracy runs MMLU correctness; "
        "all runs required+overlap+accuracy.",
    )
    p.add_argument(
        "--num-accuracy-samples",
        type=int,
        default=200,
        help="MMLU samples for HiCache correctness comparison.",
    )
    p.add_argument(
        "--accuracy-concurrency",
        type=int,
        default=8,
        help="Concurrency forwarded to bench_accuracy.",
    )
    p.add_argument(
        "--accuracy-max-tokens",
        type=int,
        default=32,
        help="Max output tokens forwarded to bench_accuracy.",
    )
    p.add_argument("--num-sessions", type=int, default=64)
    p.add_argument("--turns-per-session", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--multiturn-schedule",
        choices=["session", "round_robin"],
        default="round_robin",
        help="Forwarded to bench_cache. round_robin creates longer cache "
        "reuse distance and is the recommended HiCache benchmark.",
    )
    p.add_argument(
        "--question-len",
        type=int,
        default=0,
        help="Forwarded to bench_cache for shared workloads only; retained "
        "for symmetry with bench_cache.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the non-smoke confirmation prompt.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned server/client commands and exit without launching.",
    )
    return p.parse_args()


def base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def health_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def fetch_json(url: str, path: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{url}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wait_for_health(
    url: str, proc: subprocess.Popen, timeout_sec: int, server_log: Path
) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_sec:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {proc.returncode}; see {server_log}"
            )
        if health_ok(url, timeout=2.0):
            return
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy; see {server_log}")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        if proc.poll() is None:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=30)


def stream_command(cmd: list[str], log_path: Path) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def client_log_has_failure(log_path: Path) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    markers = (
        "Traceback",
        "RuntimeError",
        "KV pool exhausted",
        "CUDA out of memory",
        "out of memory",
        "HTTP 500",
        "No samples!",
    )
    return any(marker in text for marker in markers)


def start_server(
    args: argparse.Namespace, exp: Experiment, exp_dir: Path
) -> tuple[subprocess.Popen, Path]:
    url = base_url(args)
    if health_ok(url, timeout=0.5):
        raise RuntimeError(
            f"{url} is already healthy. Stop the existing server before running this script."
        )

    server_log = exp_dir / "server.log"
    server_cmd = server_cmd_for(args, exp)

    with server_log.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(server_cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            server_cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )

    print(f"Starting server for {exp.name}; log={server_log}", flush=True)
    wait_for_health(url, proc, args.server_timeout_sec, server_log)
    print(f"Server healthy for {exp.name}", flush=True)
    return proc, server_log


def bench_cache_multiturn_cmd(
    args: argparse.Namespace,
    *,
    num_sessions: int | None = None,
    turns_per_session: int | None = None,
    concurrency: int | None = None,
    max_tokens: int | None = None,
    schedule: str | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmark.bench_cache",
        "--base-url",
        base_url(args),
        "--workload",
        "multiturn",
        "--num-sessions",
        str(num_sessions if num_sessions is not None else args.num_sessions),
        "--turns-per-session",
        str(
            turns_per_session
            if turns_per_session is not None
            else args.turns_per_session
        ),
        "--concurrency",
        str(concurrency if concurrency is not None else args.concurrency),
        "--max-tokens",
        str(max_tokens if max_tokens is not None else args.max_tokens),
        "--multiturn-schedule",
        schedule if schedule is not None else args.multiturn_schedule,
    ]


def bench_accuracy_mmlu_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmark.bench_accuracy",
        "--base-url",
        base_url(args),
        "--dataset",
        "mmlu",
        "--num-samples",
        str(args.num_accuracy_samples),
        "--concurrency",
        str(args.accuracy_concurrency),
        "--max-tokens",
        str(args.accuracy_max_tokens),
    ]


def experiments(args: argparse.Namespace) -> list[Experiment]:
    gpu_only = ["--cpu-cache-size-gb", "0"]
    hicache = ["--cpu-cache-size-gb", str(args.cpu_cache_size_gb)]
    hicache_overlap = [*hicache, "--hicache-overlap"]
    smoke_hicache = ["--cpu-cache-size-gb", str(args.smoke_cpu_cache_size_gb)]

    required = [
        Experiment(
            name="gpu_only_multiturn_cliff",
            description=(
                "Milestone 4 Track 1 baseline: GPU-only radix cache on the "
                "multiturn workload. Tune sessions/turns until late-turn hit "
                "rate collapses."
            ),
            server_extra=gpu_only,
            clients=[
                ClientRun(
                    "bench_cache_multiturn",
                    bench_cache_multiturn_cmd(args),
                )
            ],
        ),
        Experiment(
            name="hicache_multiturn_recovery",
            description=(
                "Same multiturn workload with HiCache enabled. The CPU tier "
                "should preserve evicted prefixes and restore late-turn hits."
            ),
            server_extra=hicache,
            clients=[
                ClientRun(
                    "bench_cache_multiturn",
                    bench_cache_multiturn_cmd(args),
                )
            ],
        ),
    ]

    overlap = [
        Experiment(
            name="hicache_overlap_multiturn",
            description=(
                "Same HiCache workload with --hicache-overlap enabled for the "
                "optional overlap/bonus measurement."
            ),
            server_extra=hicache_overlap,
            clients=[
                ClientRun(
                    "bench_cache_multiturn",
                    bench_cache_multiturn_cmd(args),
                )
            ],
        )
    ]

    accuracy = [
        Experiment(
            name="gpu_only_mmlu_accuracy",
            description=(
                "Correctness baseline: MMLU accuracy with the Milestone 3 "
                "GPU-only radix cache."
            ),
            server_extra=gpu_only,
            clients=[ClientRun("bench_accuracy_mmlu", bench_accuracy_mmlu_cmd(args))],
        ),
        Experiment(
            name="hicache_mmlu_accuracy",
            description=(
                "Correctness check: MMLU accuracy with HiCache enabled and "
                "blocking CPU-tier copies."
            ),
            server_extra=hicache,
            clients=[ClientRun("bench_accuracy_mmlu", bench_accuracy_mmlu_cmd(args))],
        ),
        Experiment(
            name="hicache_overlap_mmlu_accuracy",
            description=(
                "Correctness check: MMLU accuracy with HiCache async overlap "
                "enabled."
            ),
            server_extra=hicache_overlap,
            clients=[ClientRun("bench_accuracy_mmlu", bench_accuracy_mmlu_cmd(args))],
        ),
    ]

    smoke = [
        Experiment(
            name="smoke_gpu_only_multiturn",
            description="Small GPU-only multiturn smoke test.",
            server_extra=gpu_only,
            clients=[
                ClientRun(
                    "bench_cache_multiturn_smoke",
                    bench_cache_multiturn_cmd(
                        args,
                        num_sessions=2,
                        turns_per_session=2,
                        concurrency=1,
                        max_tokens=4,
                        schedule="session",
                    ),
                )
            ],
        ),
        Experiment(
            name="smoke_hicache_multiturn",
            description="Small HiCache multiturn smoke test.",
            server_extra=smoke_hicache,
            clients=[
                ClientRun(
                    "bench_cache_multiturn_smoke",
                    bench_cache_multiturn_cmd(
                        args,
                        num_sessions=2,
                        turns_per_session=2,
                        concurrency=1,
                        max_tokens=4,
                        schedule="session",
                    ),
                )
            ],
        ),
    ]

    if args.suite == "smoke":
        return smoke
    if args.suite == "required":
        return required
    if args.suite == "overlap":
        return overlap
    if args.suite == "accuracy":
        return accuracy
    return required + overlap + accuracy


def server_cmd_for(
    args: argparse.Namespace, exp: Experiment, *, include_executable: bool = True
) -> list[str]:
    cmd = [
        sys.executable if include_executable else "python",
        "-m",
        "miniengine",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--mode",
        "paged",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--page-size",
        str(args.page_size),
        "--prefill-chunk-size",
        str(args.prefill_chunk_size),
        "--max-running",
        str(args.max_running),
        *exp.server_extra,
    ]
    return cmd


def write_readme(run_dir: Path, args: argparse.Namespace, exps: list[Experiment]) -> None:
    lines = [
        "# Milestone 4 HiCache Benchmark Run",
        "",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: `{args.model}`",
        f"Base URL: `{base_url(args)}`",
        f"Suite: `{args.suite}`",
        "",
        "Server settings:",
        "",
        f"- `mem_fraction_static`: `{args.mem_fraction_static}`",
        f"- `page_size`: `{args.page_size}`",
        f"- `prefill_chunk_size`: `{args.prefill_chunk_size}`",
        f"- `max_running`: `{args.max_running}`",
        f"- HiCache CPU tier: `{args.cpu_cache_size_gb}` GiB",
        f"- Smoke HiCache CPU tier: `{args.smoke_cpu_cache_size_gb}` GiB",
        "",
        "Client workload:",
        "",
        f"- `num_sessions`: `{args.num_sessions}`",
        f"- `turns_per_session`: `{args.turns_per_session}`",
        f"- `concurrency`: `{args.concurrency}`",
        f"- `max_tokens`: `{args.max_tokens}`",
        f"- `multiturn_schedule`: `{args.multiturn_schedule}`",
        f"- MMLU accuracy samples: `{args.num_accuracy_samples}`",
        f"- MMLU accuracy concurrency: `{args.accuracy_concurrency}`",
        f"- MMLU accuracy max tokens: `{args.accuracy_max_tokens}`",
        "",
        "Each experiment directory contains:",
        "",
        "- `server.log`: MiniEngine server stdout/stderr",
        "- `<client>.log`: benchmark client stdout/stderr",
        "- `cache_stats_before.json`: cache counters before client run",
        "- `cache_stats_after_<client>.json`: cache counters after each client",
        "- `metadata.json`: commands, return codes, and cache stats file paths",
        "",
        "Report mapping:",
        "",
        "- GPU-only cliff: `gpu_only_multiturn_cliff`",
        "- HiCache recovery: `hicache_multiturn_recovery`",
        "- Optional overlap: `hicache_overlap_multiturn`",
        "- MMLU correctness: `gpu_only_mmlu_accuracy`, "
        "`hicache_mmlu_accuracy`, `hicache_overlap_mmlu_accuracy`",
        "",
        "Experiments:",
        "",
    ]
    for exp in exps:
        lines.append(f"- `{exp.name}`: {exp.description}")
    run_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def confirm_long_run(args: argparse.Namespace, exps: list[Experiment]) -> None:
    if args.dry_run or args.yes or args.suite == "smoke":
        return
    print(
        f"This will run {len(exps)} server launches for suite={args.suite}. "
        "The required suite can take a long time on a single L4 GPU.",
        flush=True,
    )
    ans = input("Continue? Type 'yes' to proceed: ").strip().lower()
    if ans != "yes":
        raise SystemExit("aborted")


def main() -> None:
    args = parse_args()
    exps = experiments(args)
    confirm_long_run(args, exps)

    if args.dry_run:
        print(f"Base URL: {base_url(args)}")
        for idx, exp in enumerate(exps, start=1):
            print(f"\n=== [{idx}/{len(exps)}] {exp.name} ===")
            print(exp.description)
            print("server:")
            print(f"  $ {' '.join(server_cmd_for(args, exp, include_executable=False))}")
            for client in exp.clients:
                print(f"client {client.name}:")
                print(f"  $ {' '.join(client.cmd)}")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / args.logs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_readme(run_dir, args, exps)

    manifest: list[dict[str, Any]] = []
    failures: list[str] = []
    print(f"Logs: {run_dir}", flush=True)

    for idx, exp in enumerate(exps, start=1):
        exp_dir = run_dir / f"{idx:02d}_{exp.name}"
        exp_dir.mkdir(parents=True)
        print(f"\n=== [{idx}/{len(exps)}] {exp.name} ===", flush=True)
        print(exp.description, flush=True)

        metadata: dict[str, Any] = {
            "name": exp.name,
            "description": exp.description,
            "server_extra": exp.server_extra,
            "clients": [],
        }
        proc: subprocess.Popen | None = None
        try:
            proc, server_log = start_server(args, exp, exp_dir)
            metadata["server_log"] = str(server_log.relative_to(run_dir))

            before = fetch_json(base_url(args), "/cache_stats")
            before_path = exp_dir / "cache_stats_before.json"
            write_json(before_path, before)
            metadata["cache_stats_before"] = str(before_path.relative_to(run_dir))

            for client in exp.clients:
                client_log = exp_dir / f"{client.name}.log"
                code = stream_command(client.cmd, client_log)
                log_failed = client_log_has_failure(client_log)
                server_died = proc.poll() is not None
                server_returncode = proc.returncode if server_died else None

                after_path = exp_dir / f"cache_stats_after_{client.name}.json"
                after_error = None
                try:
                    after = fetch_json(base_url(args), "/cache_stats")
                    write_json(after_path, after)
                    after_rel = str(after_path.relative_to(run_dir))
                except Exception as exc:
                    after_error = repr(exc)
                    after_rel = None

                metadata["clients"].append(
                    {
                        "name": client.name,
                        "cmd": client.cmd,
                        "log": str(client_log.relative_to(run_dir)),
                        "returncode": code,
                        "log_failed": log_failed,
                        "server_returncode": server_returncode,
                        "cache_stats_after": after_rel,
                        "cache_stats_after_error": after_error,
                    }
                )
                if code != 0 or log_failed or server_died:
                    failures.append(
                        f"{exp.name}/{client.name}: failed "
                        f"(client_code={code}, log_failed={log_failed}, "
                        f"server_returncode={server_returncode})"
                    )
                if server_died:
                    break
        except Exception as exc:
            metadata["error"] = repr(exc)
            failures.append(f"{exp.name}: {exc}")
            print(f"ERROR: {exc}", flush=True)
        finally:
            stop_server(proc)
            write_json(exp_dir / "metadata.json", metadata)
            manifest.append(metadata)

    write_json(run_dir / "manifest.json", manifest)

    print(f"\nFinished. Logs: {run_dir}", flush=True)
    if failures:
        print("\nFailures / unexpected outcomes:", flush=True)
        for item in failures:
            print(f"- {item}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
