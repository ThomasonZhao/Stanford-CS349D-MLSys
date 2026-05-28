"""Run the Milestone 3 benchmark suite sequentially.

This script is intentionally operational rather than clever:

* launch exactly one MiniEngine server config at a time
* wait for /health before starting the client benchmark
* save server and client logs under a timestamped directory
* stop the server before moving to the next config

Run from the repo root, preferably in the same environment you use for the
server, for example:

    uv run python -m benchmark.run_milestone3 --suite required

The full/default suite is long. Use ``--suite smoke`` first to validate the
runner and server plumbing before collecting report numbers.
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


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ClientRun:
    name: str
    cmd: list[str]
    expect_failure: bool = False


@dataclass
class Experiment:
    name: str
    description: str
    server_extra: list[str] = field(default_factory=list)
    clients: list[ClientRun] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Milestone 3 benchmarks")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=512)
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--server-timeout-sec", type=int, default=900)
    p.add_argument("--logs-root", default="logs/milestone3")
    p.add_argument(
        "--suite",
        choices=["smoke", "required", "bonus", "all"],
        default="all",
        help="smoke is small; required covers report items 1-7; bonus covers retraction; all runs both.",
    )
    p.add_argument(
        "--num-accuracy-samples",
        type=int,
        default=200,
        help="MMLU samples for Part A accuracy comparison.",
    )
    p.add_argument(
        "--serving-num-requests",
        type=int,
        default=32,
        help="Requests per bench_serving run where a fixed count is useful.",
    )
    p.add_argument(
        "--oom-input-len",
        type=int,
        default=4096,
        help="Long prompt length used for prefill-OOM and retraction workloads.",
    )
    p.add_argument(
        "--oom-output-len",
        type=int,
        default=128,
        help="Output length for Part A prefill-OOM comparison.",
    )
    p.add_argument(
        "--bonus-output-len",
        type=int,
        default=512,
        help="Output length for Bonus decode-time OOM/retraction comparison.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the full-suite confirmation prompt.",
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
        " ALL FAILED",
        " ERR ",
        "Traceback",
        "RuntimeError",
        "KV pool exhausted",
        "CUDA out of memory",
        "out of memory",
        "HTTP 500",
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
    server_cmd = [
        sys.executable,
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
        "--max-running",
        str(args.max_running),
        *exp.server_extra,
    ]

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


def bench_serving_cmd(
    args: argparse.Namespace,
    *,
    input_len: int = 1024,
    output_len: int = 512,
    concurrencies: str = "1,4,16",
    randomness: float = 0.5,
    num_requests: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.bench_serving",
        "--base-url",
        base_url(args),
        "--model",
        args.model,
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--concurrencies",
        concurrencies,
        "--randomness",
        str(randomness),
        "--seed",
        "42",
    ]
    if num_requests is not None:
        cmd.extend(["--num-requests", str(num_requests)])
    return cmd


def bench_accuracy_cmd(args: argparse.Namespace) -> list[str]:
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
        "8",
        "--max-tokens",
        "32",
    ]


def bench_cache_shared_cmd(args: argparse.Namespace, prefix_len: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmark.bench_cache",
        "--base-url",
        base_url(args),
        "--workload",
        "shared",
        "--num-groups",
        "10",
        "--questions-per-group",
        "10",
        "--shared-prefix-len",
        str(prefix_len),
        "--concurrency",
        "4",
        "--max-tokens",
        "16",
    ]


def bench_cache_multiturn_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmark.bench_cache",
        "--base-url",
        base_url(args),
        "--workload",
        "multiturn",
        "--num-sessions",
        "16",
        "--turns-per-session",
        "6",
        "--concurrency",
        "8",
        "--max-tokens",
        "16",
    ]


def experiments(args: argparse.Namespace) -> list[Experiment]:
    chunk = ["--prefill-chunk-size", str(args.prefill_chunk_size)]
    no_chunk = ["--prefill-chunk-size", "0"]
    no_cache = ["--disable-radix-cache"]
    no_retraction = ["--disable-retraction"]

    required = [
        Experiment(
            name="part_a_oom_unchunked_expected_failure",
            description="Required item 1: long prefill without chunking, expected to OOM/fail.",
            server_extra=[*no_chunk, *no_cache],
            clients=[
                ClientRun(
                    "bench_serving_long_prefill",
                    bench_serving_cmd(
                        args,
                        input_len=args.oom_input_len,
                        output_len=args.oom_output_len,
                        concurrencies="16",
                        randomness=1.0,
                        num_requests=16,
                    ),
                    expect_failure=True,
                )
            ],
        ),
        Experiment(
            name="part_a_oom_chunked_success",
            description="Required item 1: same long prefill with chunking enabled.",
            server_extra=[*chunk, *no_cache],
            clients=[
                ClientRun(
                    "bench_serving_long_prefill",
                    bench_serving_cmd(
                        args,
                        input_len=args.oom_input_len,
                        output_len=args.oom_output_len,
                        concurrencies="16",
                        randomness=1.0,
                        num_requests=16,
                    ),
                )
            ],
        ),
        Experiment(
            name="part_a_accuracy_unchunked",
            description="Required item 2: MMLU accuracy without chunking.",
            server_extra=[*no_chunk, *no_cache],
            clients=[ClientRun("bench_accuracy_mmlu", bench_accuracy_cmd(args))],
        ),
        Experiment(
            name="part_a_accuracy_chunked",
            description="Required item 2: MMLU accuracy with chunking.",
            server_extra=[*chunk, *no_cache],
            clients=[ClientRun("bench_accuracy_mmlu", bench_accuracy_cmd(args))],
        ),
        Experiment(
            name="part_a_no_regression_unchunked",
            description="Required item 3: bench_serving default-ish workload without chunking.",
            server_extra=[*no_chunk, *no_cache],
            clients=[
                ClientRun(
                    "bench_serving_conc_1_4_16",
                    bench_serving_cmd(
                        args,
                        input_len=1024,
                        output_len=512,
                        concurrencies="1,4,16",
                        randomness=0.5,
                        num_requests=args.serving_num_requests,
                    ),
                )
            ],
        ),
        Experiment(
            name="part_a_no_regression_chunked",
            description="Required item 3: same workload with chunking.",
            server_extra=[*chunk, *no_cache],
            clients=[
                ClientRun(
                    "bench_serving_conc_1_4_16",
                    bench_serving_cmd(
                        args,
                        input_len=1024,
                        output_len=512,
                        concurrencies="1,4,16",
                        randomness=0.5,
                        num_requests=args.serving_num_requests,
                    ),
                )
            ],
        ),
        Experiment(
            name="part_b_shared_cache_off",
            description="Required item 4: shared-prefix workload with radix cache disabled.",
            server_extra=[*chunk, *no_cache],
            clients=[ClientRun("bench_cache_shared_2000", bench_cache_shared_cmd(args, 2000))],
        ),
        Experiment(
            name="part_b_shared_cache_on",
            description="Required item 4: shared-prefix workload with radix cache enabled.",
            server_extra=chunk,
            clients=[ClientRun("bench_cache_shared_2000", bench_cache_shared_cmd(args, 2000))],
        ),
        Experiment(
            name="part_b_shared_prefix_sweep_cache_on",
            description="Required item 5: radix-cache shared-prefix sweep.",
            server_extra=chunk,
            clients=[
                ClientRun(
                    f"bench_cache_shared_{prefix}",
                    bench_cache_shared_cmd(args, prefix),
                )
                for prefix in (200, 500, 2000, 4000)
            ],
        ),
        Experiment(
            name="part_b_multiturn_cache_off",
            description="Required item 6: multiturn workload with radix cache disabled.",
            server_extra=[*chunk, *no_cache],
            clients=[ClientRun("bench_cache_multiturn", bench_cache_multiturn_cmd(args))],
        ),
        Experiment(
            name="part_b_multiturn_cache_on",
            description="Required item 6: multiturn workload with radix cache enabled.",
            server_extra=chunk,
            clients=[ClientRun("bench_cache_multiturn", bench_cache_multiturn_cmd(args))],
        ),
        Experiment(
            name="part_b_no_regression_cache_off",
            description="Required item 7: default serving workload with cache disabled.",
            server_extra=[*chunk, *no_cache],
            clients=[
                ClientRun(
                    "bench_serving_conc_1_4_16",
                    bench_serving_cmd(
                        args,
                        input_len=1024,
                        output_len=512,
                        concurrencies="1,4,16",
                        randomness=0.5,
                        num_requests=args.serving_num_requests,
                    ),
                )
            ],
        ),
        Experiment(
            name="part_b_no_regression_cache_on",
            description="Required item 7: default serving workload with cache enabled.",
            server_extra=chunk,
            clients=[
                ClientRun(
                    "bench_serving_conc_1_4_16",
                    bench_serving_cmd(
                        args,
                        input_len=1024,
                        output_len=512,
                        concurrencies="1,4,16",
                        randomness=0.5,
                        num_requests=args.serving_num_requests,
                    ),
                )
            ],
        ),
    ]

    bonus = [
        Experiment(
            name="bonus_decode_oom_no_retraction_expected_failure",
            description="Bonus item 8: decode-time KV exhaustion with retraction disabled.",
            server_extra=[*chunk, *no_retraction],
            clients=[
                ClientRun(
                    "bench_serving_decode_oom",
                    bench_serving_cmd(
                        args,
                        input_len=args.oom_input_len,
                        output_len=args.bonus_output_len,
                        concurrencies="16",
                        randomness=1.0,
                        num_requests=16,
                    ),
                    expect_failure=True,
                )
            ],
        ),
        Experiment(
            name="bonus_decode_oom_retraction_success",
            description="Bonus item 9: same decode-heavy workload with retraction enabled.",
            server_extra=chunk,
            clients=[
                ClientRun(
                    "bench_serving_decode_oom",
                    bench_serving_cmd(
                        args,
                        input_len=args.oom_input_len,
                        output_len=args.bonus_output_len,
                        concurrencies="16",
                        randomness=1.0,
                        num_requests=16,
                    ),
                )
            ],
        ),
    ]

    smoke = [
        Experiment(
            name="smoke_chunked_cache_on",
            description="Small smoke test for server/client/log plumbing.",
            server_extra=chunk,
            clients=[
                ClientRun(
                    "bench_serving_smoke",
                    bench_serving_cmd(
                        args,
                        input_len=256,
                        output_len=16,
                        concurrencies="1,2",
                        randomness=1.0,
                        num_requests=4,
                    ),
                ),
                ClientRun(
                    "bench_cache_shared_smoke",
                    [
                        sys.executable,
                        "-m",
                        "benchmark.bench_cache",
                        "--base-url",
                        base_url(args),
                        "--workload",
                        "shared",
                        "--num-groups",
                        "2",
                        "--questions-per-group",
                        "4",
                        "--shared-prefix-len",
                        "256",
                        "--concurrency",
                        "1",
                        "--max-tokens",
                        "8",
                    ],
                ),
            ],
        )
    ]

    if args.suite == "smoke":
        return smoke
    if args.suite == "required":
        return required
    if args.suite == "bonus":
        return bonus
    return required + bonus


def write_readme(run_dir: Path, args: argparse.Namespace, exps: list[Experiment]) -> None:
    lines = [
        "# Milestone 3 Benchmark Run",
        "",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: `{args.model}`",
        f"Base URL: `{base_url(args)}`",
        f"Suite: `{args.suite}`",
        "",
        "Each experiment directory contains:",
        "",
        "- `server.log`: MiniEngine server stdout/stderr",
        "- `<client>.log`: benchmark client stdout/stderr",
        "- `metadata.json`: commands, return codes, and expectation flags",
        "",
        "Report mapping:",
        "",
        "- Part A item 1: `part_a_oom_*`",
        "- Part A item 2: `part_a_accuracy_*`",
        "- Part A item 3: `part_a_no_regression_*`",
        "- Part B item 4: `part_b_shared_cache_*`",
        "- Part B item 5: `part_b_shared_prefix_sweep_cache_on`",
        "- Part B item 6: `part_b_multiturn_cache_*`",
        "- Part B item 7: `part_b_no_regression_cache_*`",
        "- Bonus items 8-9: `bonus_decode_oom_*`",
        "",
        "Experiments:",
        "",
    ]
    for exp in exps:
        lines.append(f"- `{exp.name}`: {exp.description}")
    run_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def confirm_long_run(args: argparse.Namespace, exps: list[Experiment]) -> None:
    if args.yes or args.suite == "smoke":
        return
    print(
        f"This will run {len(exps)} server launches for suite={args.suite}. "
        "It may take hours on a single L4 GPU.",
        flush=True,
    )
    ans = input("Continue? Type 'yes' to proceed: ").strip().lower()
    if ans != "yes":
        raise SystemExit("aborted")


def main() -> None:
    args = parse_args()
    exps = experiments(args)
    confirm_long_run(args, exps)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / args.logs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_readme(run_dir, args, exps)

    manifest: list[dict] = []
    failures: list[str] = []
    print(f"Logs: {run_dir}", flush=True)

    for idx, exp in enumerate(exps, start=1):
        exp_dir = run_dir / f"{idx:02d}_{exp.name}"
        exp_dir.mkdir(parents=True)
        print(f"\n=== [{idx}/{len(exps)}] {exp.name} ===", flush=True)
        print(exp.description, flush=True)

        metadata = {
            "name": exp.name,
            "description": exp.description,
            "server_extra": exp.server_extra,
            "clients": [],
        }
        proc: subprocess.Popen | None = None
        try:
            proc, server_log = start_server(args, exp, exp_dir)
            metadata["server_log"] = str(server_log.relative_to(run_dir))

            for client in exp.clients:
                client_log = exp_dir / f"{client.name}.log"
                code = stream_command(client.cmd, client_log)
                log_failed = client_log_has_failure(client_log)
                server_died = proc.poll() is not None
                observed_failure = code != 0 or log_failed or server_died
                expected = client.expect_failure
                server_returncode = proc.returncode if server_died else None
                metadata["clients"].append(
                    {
                        "name": client.name,
                        "cmd": client.cmd,
                        "log": str(client_log.relative_to(run_dir)),
                        "returncode": code,
                        "log_failed": log_failed,
                        "server_returncode": server_returncode,
                        "expect_failure": expected,
                    }
                )
                if expected and not observed_failure:
                    failures.append(f"{exp.name}/{client.name}: expected failure but passed")
                elif not expected and observed_failure:
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
            exp_dir.joinpath("metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            manifest.append(metadata)

    run_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nFinished. Logs: {run_dir}", flush=True)
    if failures:
        print("\nFailures / unexpected outcomes:", flush=True)
        for item in failures:
            print(f"- {item}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
