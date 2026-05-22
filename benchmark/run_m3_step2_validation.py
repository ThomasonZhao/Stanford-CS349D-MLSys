"""Run Milestone 3 Step 2 chunked-prefill validation.

This is a thin orchestration wrapper around the existing server and benchmark
entry points.  It starts one server per configuration, captures the requested
logs under ``report_screenshots/M3_step2/``, and extracts a compact Markdown
summary table for the report.

The script intentionally does not tune parameters after failure.  Defaults are
the fixed workload chosen for the Step 2 validation plan:

* OOM demo: input_len=6144, concurrency=4, output_len=16.
* No-regression serving: input_len=4096, conc=1,4,16, output_len=512.
* Accuracy: MMLU, 200 samples.

Run on the L4 VM, from the repository root:

    python -m benchmark.run_m3_step2_validation
"""

from __future__ import annotations

import argparse
import binascii
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report_screenshots" / "M3_step2"
SCREENSHOT_DIR = REPORT_DIR
SERVING_REGRESSION_THRESHOLD = 0.10
MMLU_REGRESSION_THRESHOLD_PP = 1.0


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    seconds: float
    log_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", default="paged")
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--server-extra", default="", help="Extra args for miniengine")
    p.add_argument("--bench-extra", default="", help="Extra args for bench_serving")
    p.add_argument("--acc-extra", default="", help="Extra args for bench_accuracy")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--regression-input-len", type=int, default=4096)
    p.add_argument("--regression-output-len", type=int, default=512)
    p.add_argument("--regression-concurrencies", default="1,4,16")
    p.add_argument("--oom-input-len", type=int, default=6144)
    p.add_argument("--oom-output-len", type=int, default=16)
    p.add_argument("--oom-concurrency", type=int, default=4)
    p.add_argument("--mmlu-samples", type=int, default=200)
    p.add_argument("--mmlu-concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--server-ready-timeout-s", type=float, default=900)
    p.add_argument("--bench-timeout-s", type=float, default=3600)
    p.add_argument("--acc-timeout-s", type=float, default=7200)
    p.add_argument("--max-log-mb", type=float, default=9.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-gpu-check", action="store_true")
    return p.parse_args()


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def check_cuda(skip: bool) -> None:
    if skip:
        return
    code = (
        "import torch, sys; "
        "ok=torch.cuda.is_available(); "
        "print(f'cuda_available={ok} device_count={torch.cuda.device_count()}'); "
        "sys.exit(0 if ok else 2)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def server_cmd(args: argparse.Namespace, chunk: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "miniengine",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--mode",
        args.mode,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--page-size",
        str(args.page_size),
        "--max-running",
        str(args.max_running),
        "--prefill-chunk-size",
        str(chunk),
    ]
    cmd.extend(shlex.split(args.server_extra))
    return cmd


def serving_cmd(
    args: argparse.Namespace,
    *,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_requests: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.bench_serving",
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--model",
        args.model,
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--randomness",
        "1.0",
        "--concurrencies",
        str(concurrency),
        "--num-requests",
        str(num_requests),
        "--seed",
        str(args.seed),
    ]
    cmd.extend(shlex.split(args.bench_extra))
    return cmd


def accuracy_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.bench_accuracy",
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--dataset",
        "mmlu",
        "--num-samples",
        str(args.mmlu_samples),
        "--concurrency",
        str(args.mmlu_concurrency),
    ]
    cmd.extend(shlex.split(args.acc_extra))
    return cmd


def start_server(cmd: list[str], log_path: Path, dry_run: bool) -> subprocess.Popen:
    print(f"\n[server] {quote_cmd(cmd)}")
    print(f"[server-log] {log_path}")
    if dry_run:
        raise RuntimeError("dry-run does not start subprocesses")
    log_fh = log_path.open("w", encoding="utf-8")
    log_fh.write(f"$ {quote_cmd(cmd)}\n\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._m3_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=30)
    log_fh = getattr(proc, "_m3_log_fh", None)
    if log_fh is not None:
        log_fh.close()


def wait_for_server(args: argparse.Namespace, proc: subprocess.Popen) -> None:
    url = f"http://{args.host}:{args.port}/v1/models"
    deadline = time.time() + args.server_ready_timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_err = exc
        time.sleep(2)
    raise TimeoutError(f"server did not become ready at {url}: {last_err}")


def run_logged(
    cmd: list[str], log_path: Path, timeout_s: float, dry_run: bool
) -> CommandResult:
    print(f"[run] {quote_cmd(cmd)}")
    print(f"[log] {log_path}")
    if dry_run:
        return CommandResult(cmd=cmd, returncode=0, seconds=0.0, log_path=log_path)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {quote_cmd(cmd)}\n\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[TIMEOUT] command exceeded {timeout_s:.0f}s\n")
            returncode = 124
    return CommandResult(
        cmd=cmd,
        returncode=returncode,
        seconds=time.perf_counter() - t0,
        log_path=log_path,
    )


def cap_log(path: Path, max_bytes: int) -> None:
    if not path.exists() or path.stat().st_size <= max_bytes:
        return
    data = path.read_bytes()
    head = data[: max_bytes // 5]
    tail = data[-(max_bytes - len(head)) :]
    marker = (
        b"\n\n[... log truncated by run_m3_step2_validation.py; "
        b"kept head and tail ...]\n\n"
    )
    path.write_bytes(head + marker + tail)


def combine_logs(combined: Path, server_log: Path, client_log: Path) -> None:
    with combined.open("w", encoding="utf-8") as out:
        out.write("===== SERVER LOG =====\n\n")
        if server_log.exists():
            out.write(server_log.read_text(encoding="utf-8", errors="replace"))
        else:
            out.write("[MISSING SERVER LOG]\n")
        out.write("\n\n===== CLIENT LOG =====\n\n")
        if client_log.exists():
            out.write(client_log.read_text(encoding="utf-8", errors="replace"))
        else:
            out.write("[MISSING CLIENT LOG]\n")


def write_failure_log(log_path: Path, cmd: list[str], tag: str, exc: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {quote_cmd(cmd)}\n\n")
        fh.write(f"[{tag}] {type(exc).__name__}: {exc}\n")


def run_with_server(
    args: argparse.Namespace,
    *,
    chunk: int,
    server_log: Path,
    client_cmd: list[str],
    client_log: Path,
    timeout_s: float,
    wait_ready: bool = True,
) -> tuple[CommandResult, Path]:
    proc: subprocess.Popen | None = None
    try:
        proc = start_server(server_cmd(args, chunk), server_log, args.dry_run)
        if wait_ready:
            wait_for_server(args, proc)
        result = run_logged(client_cmd, client_log, timeout_s, args.dry_run)
        return result, server_log
    finally:
        stop_server(proc)


def extract_serving_rows(log_path: Path, chunk: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not log_path.exists():
        return rows
    pat = re.compile(
        r"^\s*(?P<conc>\d+)\s+"
        r"(?P<ttft_p50>\d+)\s+(?P<ttft_p99>\d+)\s+"
        r"(?P<compl_p50>\d+)\s+(?P<compl_p99>\d+)\s+"
        r"(?P<tpot_p50>\d+(?:\.\d+)?)\s+(?P<tpot_p99>\d+(?:\.\d+)?)\s+"
        r"(?P<gentok_s>\d+)\s+(?P<ok>\d+)\s*$"
    )
    failed_pat = re.compile(r"^\s*(?P<conc>\d+)\s+ALL FAILED\s*$")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pat.match(line)
        if match:
            row = match.groupdict()
            row["chunk"] = str(chunk)
            row["status"] = "OK"
            row["notes"] = ""
            rows.append(row)
            continue
        failed = failed_pat.match(line)
        if failed:
            row = failed.groupdict()
            row.update(
                {
                    "chunk": str(chunk),
                    "ttft_p50": "FAILED",
                    "ttft_p99": "FAILED",
                    "compl_p50": "FAILED",
                    "compl_p99": "FAILED",
                    "tpot_p50": "FAILED",
                    "tpot_p99": "FAILED",
                    "gentok_s": "FAILED",
                    "ok": "0",
                    "status": "ALL_FAILED",
                    "notes": "bench_serving reported ALL FAILED",
                }
            )
            rows.append(row)
    return rows


def extract_accuracy(log_path: Path, chunk: int) -> dict[str, str] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    samples = re.search(r"Samples\s*:\s*(\d+)", text)
    correct = re.search(r"Correct\s*:\s*(\d+)", text)
    accuracy = re.search(r"Accuracy\s*:\s*([0-9.]+%)", text)
    avg_latency = re.search(r"Avg latency\s*:\s*([0-9.]+s)", text)
    if not accuracy:
        return None
    return {
        "chunk": str(chunk),
        "samples": samples.group(1) if samples else "?",
        "correct": correct.group(1) if correct else "?",
        "accuracy": accuracy.group(1),
        "avg_latency": avg_latency.group(1) if avg_latency else "?",
        "status": "OK",
        "notes": "",
    }


def serving_failure_row(
    *, chunk: int, conc: int, status: str, notes: str
) -> dict[str, str]:
    return {
        "chunk": str(chunk),
        "conc": str(conc),
        "ttft_p50": "FAILED",
        "ttft_p99": "FAILED",
        "compl_p50": "FAILED",
        "compl_p99": "FAILED",
        "tpot_p50": "FAILED",
        "tpot_p99": "FAILED",
        "gentok_s": "FAILED",
        "ok": "0",
        "status": status,
        "notes": notes,
    }


def accuracy_failure_row(*, chunk: int, status: str, notes: str) -> dict[str, str]:
    return {
        "chunk": str(chunk),
        "samples": "?",
        "correct": "?",
        "accuracy": "FAILED",
        "avg_latency": "FAILED",
        "status": status,
        "notes": notes,
    }


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_percent(value: str) -> float | None:
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1]
    return parse_float(value)


def format_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta * 100:+.1f}%"


def format_pp(delta_pp: float | None) -> str:
    if delta_pp is None:
        return "n/a"
    return f"{delta_pp:+.1f} pp"


def pair_serving_rows(
    rows: list[dict[str, str]], chunk_size: int
) -> dict[int, tuple[dict[str, str] | None, dict[str, str] | None]]:
    paired: dict[int, tuple[dict[str, str] | None, dict[str, str] | None]] = {}
    for row in rows:
        try:
            conc = int(row["conc"])
            chunk = int(row["chunk"])
        except (KeyError, ValueError):
            continue
        base, chunked = paired.get(conc, (None, None))
        if chunk == 0:
            base = row
        elif chunk == chunk_size:
            chunked = row
        paired[conc] = (base, chunked)
    return paired


def serving_delta(
    base: dict[str, str] | None,
    chunked: dict[str, str] | None,
    key: str,
) -> float | None:
    if not base or not chunked:
        return None
    b = parse_float(base.get(key, ""))
    c = parse_float(chunked.get(key, ""))
    if b is None or c is None or b == 0:
        return None
    return (c - b) / b


def serving_verdict(base: dict[str, str] | None, chunked: dict[str, str] | None) -> str:
    if not base or not chunked:
        return "FAIL"
    if base.get("status") != "OK" or chunked.get("status") != "OK":
        return "FAIL"
    latency_keys = ["ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99"]
    for key in latency_keys:
        delta = serving_delta(base, chunked, key)
        if delta is None or delta > SERVING_REGRESSION_THRESHOLD:
            return "FAIL"
    throughput_delta = serving_delta(base, chunked, "gentok_s")
    if throughput_delta is None or throughput_delta < -SERVING_REGRESSION_THRESHOLD:
        return "FAIL"
    return "PASS"


def find_accuracy_row(
    rows: list[dict[str, str]], chunk: int
) -> dict[str, str] | None:
    for row in rows:
        try:
            if int(row["chunk"]) == chunk:
                return row
        except (KeyError, ValueError):
            continue
    return None


def accuracy_delta_pp(
    base: dict[str, str] | None, chunked: dict[str, str] | None
) -> float | None:
    if not base or not chunked:
        return None
    b = parse_percent(base.get("accuracy", ""))
    c = parse_percent(chunked.get("accuracy", ""))
    if b is None or c is None:
        return None
    return c - b


def accuracy_verdict(base: dict[str, str] | None, chunked: dict[str, str] | None) -> str:
    delta = accuracy_delta_pp(base, chunked)
    if (
        not base
        or not chunked
        or base.get("status") != "OK"
        or chunked.get("status") != "OK"
        or delta is None
    ):
        return "FAIL"
    return "PASS" if abs(delta) <= MMLU_REGRESSION_THRESHOLD_PP else "FAIL"


def analyze_oom_probe(
    *, label: str, chunk: int, result: CommandResult | None, server_log: Path, client_log: Path
) -> dict[str, str]:
    server_text = (
        server_log.read_text(encoding="utf-8", errors="replace")
        if server_log.exists()
        else ""
    )
    client_text = (
        client_log.read_text(encoding="utf-8", errors="replace")
        if client_log.exists()
        else ""
    )
    cuda_oom = bool(
        re.search(r"CUDA out of memory|OutOfMemoryError|out of memory", server_text, re.I)
    )
    client_error = bool(
        re.search(
            r"\bHTTP 5\d\d\b|\bERR\b|stream error|server disconnected|"
            r"\[SERVER_FAILED\]|\[TIMEOUT\]|exception",
            client_text,
            re.I,
        )
    )
    serving_rows = extract_serving_rows(client_log, chunk)
    ok = sum(int(row.get("ok", "0")) for row in serving_rows if row.get("ok", "0").isdigit())
    success = result is not None and result.returncode == 0 and ok > 0 and not cuda_oom
    if label == "unchunked":
        verdict = "PASS" if cuda_oom and client_error else "FAIL"
    else:
        verdict = "PASS" if success and not client_error else "FAIL"
    return {
        "label": label,
        "chunk": str(chunk),
        "returncode": "SERVER_FAILED" if result is None else str(result.returncode),
        "cuda_oom": "YES" if cuda_oom else "NO",
        "client_error": "YES" if client_error else "NO",
        "ok": str(ok),
        "verdict": verdict,
    }


def write_summary(
    args: argparse.Namespace,
    serving_rows: list[dict[str, str]],
    acc_rows: list[dict[str, str]],
    oom_rows: list[dict[str, str]],
) -> None:
    out = REPORT_DIR / "summary_tables.md"
    lines = [
        "# M3 Step 2 Summary Tables",
        "",
        "## Justification",
        "",
        f"Chosen `--prefill-chunk-size`: `{args.chunk_size}`.",
        f"Chosen regression `--input-len`: `{args.regression_input_len}`.",
        (
            "OOM-demo workload: "
            f"`--input-len {args.oom_input_len} "
            f"--output-len {args.oom_output_len} "
            f"--concurrencies {args.oom_concurrency} "
            f"--num-requests {args.oom_concurrency}`."
        ),
        (
            f"Trade-off reasoning: chunk size {args.chunk_size} bounds prefill "
            f"activation memory while making the {args.regression_input_len}-token "
            "regression prompt exercise the chunked path. Smaller chunks would add "
            "kernel-launch and attention-planning overhead per chunk; larger chunks "
            "would reduce overhead but weaken the OOM avoidance goal."
        ),
        (
            "L4-specific observation: use the OOM and regression verdicts below. "
            "This script fixes the workload before running and does not tune after "
            "seeing results."
        ),
        "",
        "## OOM Demo",
        "",
        "| Probe | Chunk | Return code | CUDA OOM in server log | Client error | OK responses | Verdict |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in oom_rows:
        lines.append(
            "| {label} | {chunk} | {returncode} | {cuda_oom} | "
            "{client_error} | {ok} | {verdict} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Serving Raw Rows",
            "",
            "| Chunk | Conc | Status | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | GenTok/s | OK | Notes |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(serving_rows, key=lambda r: (int(r["conc"]), int(r["chunk"]))):
        lines.append(
            "| {chunk} | {conc} | {status} | {ttft_p50} | {ttft_p99} | "
            "{tpot_p50} | {tpot_p99} | {gentok_s} | {ok} | {notes} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Serving No-Regression Check",
            "",
            "| Conc | TTFT p50 delta | TTFT p99 delta | TPOT p50 delta | TPOT p99 delta | GenTok/s delta | Verdict |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    paired = pair_serving_rows(serving_rows, args.chunk_size)
    for conc in sorted(paired):
        base, chunked = paired[conc]
        lines.append(
            "| {conc} | {ttft50} | {ttft99} | {tpot50} | {tpot99} | {tps} | {verdict} |".format(
                conc=conc,
                ttft50=format_delta(serving_delta(base, chunked, "ttft_p50")),
                ttft99=format_delta(serving_delta(base, chunked, "ttft_p99")),
                tpot50=format_delta(serving_delta(base, chunked, "tpot_p50")),
                tpot99=format_delta(serving_delta(base, chunked, "tpot_p99")),
                tps=format_delta(serving_delta(base, chunked, "gentok_s")),
                verdict=serving_verdict(base, chunked),
            )
        )
    lines.extend(
        [
            "",
            "Latency metrics pass when chunked is <= +10%; throughput passes when chunked is >= -10%.",
            "",
            "## MMLU Accuracy Raw Rows",
            "",
            "| Chunk | Status | Samples | Correct | Accuracy | Avg latency | Notes |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(acc_rows, key=lambda r: int(r["chunk"])):
        lines.append(
            "| {chunk} | {status} | {samples} | {correct} | {accuracy} | "
            "{avg_latency} | {notes} |".format(**row)
        )
    base_acc = find_accuracy_row(acc_rows, 0)
    chunked_acc = find_accuracy_row(acc_rows, args.chunk_size)
    lines.extend(
        [
            "",
            "## MMLU Accuracy No-Regression Check",
            "",
            "| Baseline accuracy | Chunked accuracy | Delta | Verdict |",
            "|---:|---:|---:|---|",
            "| {base} | {chunked} | {delta} | {verdict} |".format(
                base=base_acc.get("accuracy", "FAILED") if base_acc else "MISSING",
                chunked=chunked_acc.get("accuracy", "FAILED") if chunked_acc else "MISSING",
                delta=format_pp(accuracy_delta_pp(base_acc, chunked_acc)),
                verdict=accuracy_verdict(base_acc, chunked_acc),
            ),
            "",
            "MMLU passes when absolute accuracy delta is <= 1.0 percentage point.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] {out}")


FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    '"': ("01010", "01010", "01010", "00000", "00000", "00000", "00000"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    "$": ("00100", "01111", "10100", "01110", "00101", "11110", "00100"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "&": ("01100", "10010", "10100", "01000", "10101", "10010", "01101"),
    "'": ("00100", "00100", "01000", "00000", "00000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "*": ("00000", "00100", "10101", "01110", "10101", "00100", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    ",": ("00000", "00000", "00000", "00000", "00100", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00100", "00100"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ";": ("00000", "00100", "00100", "00000", "00100", "00100", "01000"),
    "<": ("00010", "00100", "01000", "10000", "01000", "00100", "00010"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    ">": ("01000", "00100", "00010", "00001", "00010", "00100", "01000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "@": ("01110", "10001", "00001", "01101", "10101", "10101", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"),
    "\\": ("10000", "01000", "00100", "00010", "00001", "00000", "00000"),
    "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"),
    "^": ("00100", "01010", "10001", "00000", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "`": ("01000", "00100", "00010", "00000", "00000", "00000", "00000"),
    "{": ("00010", "00100", "00100", "01000", "00100", "00100", "00010"),
    "|": ("00100", "00100", "00100", "00000", "00100", "00100", "00100"),
    "}": ("01000", "00100", "00100", "00010", "00100", "00100", "01000"),
    "~": ("00000", "00000", "01000", "10101", "00010", "00000", "00000"),
}


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw_rows = []
    stride = width * 3
    for y in range(height):
        raw_rows.append(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(b"".join(raw_rows), level=9)),
            png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    w: int,
    h: int,
    rgb: tuple[int, int, int],
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    for yy in range(y0, y1):
        row = yy * width * 3
        for xx in range(x0, x1):
            idx = row + xx * 3
            pixels[idx : idx + 3] = bytes(rgb)


def draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    rgb: tuple[int, int, int],
    scale: int,
) -> None:
    cursor = x
    for ch in text:
        glyph = FONT_5X7.get(ch.upper(), FONT_5X7.get("?", FONT_5X7[" "]))
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    draw_rect(
                        pixels,
                        width,
                        height,
                        cursor + gx * scale,
                        y + gy * scale,
                        scale,
                        scale,
                        rgb,
                    )
        cursor += 6 * scale


def write_text_screenshot(
    source: Path,
    target: Path,
    max_lines: int = 80,
    max_cols: int = 150,
) -> None:
    """Write a simple terminal-style PNG screenshot without image deps."""
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    visible = [line[:max_cols] for line in lines[-max_lines:]]
    cols = max([len(line) for line in visible] + [80])
    rows = max(len(visible), 20)
    scale = 2
    margin = 18
    line_h = 10 * scale
    width = cols * 6 * scale + margin * 2
    height = rows * line_h + margin * 2
    bg = (17, 24, 39)
    fg = (229, 231, 235)
    pixels = bytearray(bytes(bg) * (width * height))
    for i, line in enumerate(visible):
        draw_text(pixels, width, height, margin, margin + i * line_h, line, fg, scale)
    write_png(target, width, height, pixels)
    print(f"[screenshot-png] {target}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    check_cuda(args.skip_gpu_check or args.dry_run)

    concs = [int(x) for x in args.regression_concurrencies.split(",") if x.strip()]
    max_log_bytes = int(args.max_log_mb * 1024 * 1024)

    if args.dry_run:
        print("[dry-run] Commands that would run:")

    # OOM demo: same workload, unchunked then chunked.
    oom_rows: list[dict[str, str]] = []
    oom_client_0 = REPORT_DIR / "oom_demo_unchunked_client.log"
    oom_server_0 = REPORT_DIR / "oom_demo_unchunked_server.log"
    oom_cmd_0 = serving_cmd(
        args,
        input_len=args.oom_input_len,
        output_len=args.oom_output_len,
        concurrency=args.oom_concurrency,
        num_requests=args.oom_concurrency,
    )
    if args.dry_run:
        print(quote_cmd(server_cmd(args, 0)))
        print(quote_cmd(oom_cmd_0))
    else:
        oom_result_0: CommandResult | None = None
        try:
            oom_result_0, _ = run_with_server(
                args,
                chunk=0,
                server_log=oom_server_0,
                client_cmd=oom_cmd_0,
                client_log=oom_client_0,
                timeout_s=args.bench_timeout_s,
            )
        except Exception as exc:
            write_failure_log(oom_client_0, oom_cmd_0, "SERVER_FAILED", exc)
        cap_log(oom_server_0, max_log_bytes)
        cap_log(oom_client_0, max_log_bytes)
        combine_logs(REPORT_DIR / "oom_demo_unchunked.log", oom_server_0, oom_client_0)
        oom_rows.append(
            analyze_oom_probe(
                label="unchunked",
                chunk=0,
                result=oom_result_0,
                server_log=oom_server_0,
                client_log=oom_client_0,
            )
        )

    oom_client_512 = REPORT_DIR / "oom_demo_chunked_client.log"
    oom_server_512 = REPORT_DIR / "oom_demo_chunked_server.log"
    oom_cmd_512 = serving_cmd(
        args,
        input_len=args.oom_input_len,
        output_len=args.oom_output_len,
        concurrency=args.oom_concurrency,
        num_requests=args.oom_concurrency,
    )
    if args.dry_run:
        print(quote_cmd(server_cmd(args, args.chunk_size)))
        print(quote_cmd(oom_cmd_512))
    else:
        oom_result_512: CommandResult | None = None
        try:
            oom_result_512, _ = run_with_server(
                args,
                chunk=args.chunk_size,
                server_log=oom_server_512,
                client_cmd=oom_cmd_512,
                client_log=oom_client_512,
                timeout_s=args.bench_timeout_s,
            )
        except Exception as exc:
            write_failure_log(oom_client_512, oom_cmd_512, "SERVER_FAILED", exc)
        cap_log(oom_server_512, max_log_bytes)
        cap_log(oom_client_512, max_log_bytes)
        combine_logs(REPORT_DIR / "oom_demo_chunked.log", oom_server_512, oom_client_512)
        oom_rows.append(
            analyze_oom_probe(
                label="chunked",
                chunk=args.chunk_size,
                result=oom_result_512,
                server_log=oom_server_512,
                client_log=oom_client_512,
            )
        )

    serving_rows: list[dict[str, str]] = []
    for chunk in (0, args.chunk_size):
        for conc in concs:
            server_log = (
                REPORT_DIR
                / f"server_chunk_{chunk}_inputlen_{args.regression_input_len}_conc_{conc}.log"
            )
            bench_log = (
                REPORT_DIR
                / f"bench_chunk_{chunk}_inputlen_{args.regression_input_len}_conc_{conc}.log"
            )
            cmd = serving_cmd(
                args,
                input_len=args.regression_input_len,
                output_len=args.regression_output_len,
                concurrency=conc,
                num_requests=max(conc * 2, 8),
            )
            if args.dry_run:
                print(quote_cmd(server_cmd(args, chunk)))
                print(quote_cmd(cmd))
                continue
            result: CommandResult | None = None
            try:
                result, _ = run_with_server(
                    args,
                    chunk=chunk,
                    server_log=server_log,
                    client_cmd=cmd,
                    client_log=bench_log,
                    timeout_s=args.bench_timeout_s,
                )
            except Exception as exc:
                write_failure_log(bench_log, cmd, "SERVER_FAILED", exc)
            cap_log(server_log, max_log_bytes)
            cap_log(bench_log, max_log_bytes)
            rows = extract_serving_rows(bench_log, chunk)
            if not rows:
                if result is None:
                    rows.append(
                        serving_failure_row(
                            chunk=chunk,
                            conc=conc,
                            status="SERVER_FAILED",
                            notes="server failed before benchmark completed",
                        )
                    )
                elif result.returncode != 0:
                    rows.append(
                        serving_failure_row(
                            chunk=chunk,
                            conc=conc,
                            status="CLIENT_FAILED",
                            notes=f"bench_serving exited {result.returncode}",
                        )
                    )
                else:
                    rows.append(
                        serving_failure_row(
                            chunk=chunk,
                            conc=conc,
                            status="PARSE_FAILED",
                            notes="bench_serving exited 0 but no summary row was parsed",
                        )
                    )
            serving_rows.extend(rows)

    acc_rows: list[dict[str, str]] = []
    for chunk in (0, args.chunk_size):
        server_log = REPORT_DIR / f"server_acc_chunk_{chunk}.log"
        acc_log = REPORT_DIR / f"acc_chunk_{chunk}.log"
        cmd = accuracy_cmd(args)
        if args.dry_run:
            print(quote_cmd(server_cmd(args, chunk)))
            print(quote_cmd(cmd))
            continue
        result: CommandResult | None = None
        try:
            result, _ = run_with_server(
                args,
                chunk=chunk,
                server_log=server_log,
                client_cmd=cmd,
                client_log=acc_log,
                timeout_s=args.acc_timeout_s,
            )
        except Exception as exc:
            write_failure_log(acc_log, cmd, "SERVER_FAILED", exc)
        cap_log(server_log, max_log_bytes)
        cap_log(acc_log, max_log_bytes)
        row = extract_accuracy(acc_log, chunk)
        if row:
            acc_rows.append(row)
        elif result is None:
            acc_rows.append(
                accuracy_failure_row(
                    chunk=chunk,
                    status="SERVER_FAILED",
                    notes="server failed before accuracy run completed",
                )
            )
        elif result.returncode != 0:
            acc_rows.append(
                accuracy_failure_row(
                    chunk=chunk,
                    status="CLIENT_FAILED",
                    notes=f"bench_accuracy exited {result.returncode}",
                )
            )
        else:
            acc_rows.append(
                accuracy_failure_row(
                    chunk=chunk,
                    status="PARSE_FAILED",
                    notes="bench_accuracy exited 0 but no accuracy row was parsed",
                )
            )

    if not args.dry_run:
        write_summary(args, serving_rows, acc_rows, oom_rows)
        write_text_screenshot(
            REPORT_DIR / "summary_tables.md",
            SCREENSHOT_DIR / "M3_step2_summary_tables.png",
        )
        write_text_screenshot(
            REPORT_DIR / "oom_demo_unchunked.log",
            SCREENSHOT_DIR / "M3_step2_oom_unchunked.png",
        )
        write_text_screenshot(
            REPORT_DIR / "oom_demo_chunked.log",
            SCREENSHOT_DIR / "M3_step2_oom_chunked.png",
        )


if __name__ == "__main__":
    main()
