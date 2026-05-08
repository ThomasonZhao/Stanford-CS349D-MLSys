#!/usr/bin/env bash
# rerun_all.sh — runs all 9 benches needed for the Gradescope screenshots.
#
# Usage (on the L4 instance, inside the repo root):
#     cd /workspace/Stanford-CS349D-MLSys
#     source .venv/bin/activate
#     bash milestones/rerun_all.sh
#
# Output: 9 logs in report_screenshots/, named in the order they should
# be screenshot-cropped for the report.
#
# Wall-clock estimate: ~75 minutes total. Compile warmup (~7 min for
# step 4, ~7 min compile + ~6 min cuda-graph capture for step 5) is
# the slowest part.
#
# To screenshot after this finishes:
#     for f in report_screenshots/*.log; do
#         echo "=== $f ==="
#         cat "$f"
#         echo
#     done
# Then take per-file screenshots of the summary tables at the bottom.

set -e  # stop on first error
set -u  # error on unset vars

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
OUT_DIR="${REPO_ROOT}/report_screenshots"
MODEL="Qwen/Qwen3-8B"
BENCH_ARGS=(
    --base-url http://localhost:8000
    --input-len 512 --output-len 256 --randomness 0.5
    --concurrencies 1,2,4,8,16,32
)
ACC_ARGS=(
    --base-url http://localhost:8000
    --dataset mmlu --num-samples 200
)

mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"

# Ensure no stale server is running.
kill_server() {
    pkill -f "python -m miniengine" 2>/dev/null || true
    sleep 5
    pgrep -af "python -m miniengine" && {
        echo "Stale miniengine still running; force killing"
        pkill -9 -f "python -m miniengine" 2>/dev/null || true
        sleep 3
    } || true
}

# Start server in background; block until "Application startup complete".
start_server() {
    local log_path="$1"; shift
    local args=("$@")
    : > "$log_path"
    nohup python -m miniengine --model "$MODEL" --dtype bfloat16 "${args[@]}" \
        > "$log_path" 2>&1 < /dev/null &
    disown
    echo "  starting server (log: $log_path)..."
    until grep -q "Application startup complete" "$log_path" 2>/dev/null; do
        sleep 5
        if ! pgrep -af "python -m miniengine" > /dev/null; then
            echo "FATAL: server died during startup. Tail of log:"
            tail -30 "$log_path"
            exit 1
        fi
    done
    echo "  server up."
}

# ─── 1. M1 batched ──────────────────────────────────────────────────
echo
echo "=== [1/9] batched throughput ==="
kill_server
start_server "$OUT_DIR/01_server_batched.log" --mode batched --max-running 16
python benchmark/bench_serving.py "${BENCH_ARGS[@]}" \
    | tee "$OUT_DIR/01_bench_batched.log"

echo
echo "=== [5/9] batched MMLU ==="
python benchmark/bench_accuracy.py "${ACC_ARGS[@]}" \
    | tee "$OUT_DIR/05_acc_batched.log"

# ─── 2. paged ───────────────────────────────────────────────────────
echo
echo "=== [2/9] paged throughput ==="
kill_server
start_server "$OUT_DIR/02_server_paged.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 --max-running 16
python benchmark/bench_serving.py "${BENCH_ARGS[@]}" \
    | tee "$OUT_DIR/02_bench_paged.log"

echo
echo "=== [6/9] paged MMLU ==="
python benchmark/bench_accuracy.py "${ACC_ARGS[@]}" \
    | tee "$OUT_DIR/06_acc_paged.log"

# ─── 9. paged --page-size 32 ────────────────────────────────────────
echo
echo "=== [9/9] paged --page-size 32 throughput (page-size comparison) ==="
kill_server
start_server "$OUT_DIR/09_server_paged_ps32.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 32 --max-running 16
python benchmark/bench_serving.py "${BENCH_ARGS[@]}" \
    | tee "$OUT_DIR/09_bench_paged_ps32.log"

# ─── 3. paged + torch.compile ───────────────────────────────────────
# ~7 min compile warmup before the bench will start.
echo
echo "=== [3/9] paged + torch.compile throughput (Part C, max-autotune) ==="
echo "    expect ~7 min compile warmup before bench output appears..."
kill_server
start_server "$OUT_DIR/03_server_compile.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile --max-running 16
python benchmark/bench_serving.py "${BENCH_ARGS[@]}" \
    | tee "$OUT_DIR/03_bench_compile.log"

echo
echo "=== [7/9] paged + compile MMLU ==="
python benchmark/bench_accuracy.py "${ACC_ARGS[@]}" \
    | tee "$OUT_DIR/07_acc_compile.log"

# ─── 4. paged + compile + cuda-graph (extra credit) ────────────────
# ~7 min compile + ~6 min cuda-graph capture = ~13 min before bench.
echo
echo "=== [4/9] paged + compile + cuda-graph throughput (extra credit) ==="
echo "    expect ~13 min compile + capture before bench output appears..."
kill_server
start_server "$OUT_DIR/04_server_graph.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile --cuda-graph --max-running 16
python benchmark/bench_serving.py "${BENCH_ARGS[@]}" \
    | tee "$OUT_DIR/04_bench_graph.log"

echo
echo "=== [8/9] paged + compile + cuda-graph MMLU ==="
python benchmark/bench_accuracy.py "${ACC_ARGS[@]}" \
    | tee "$OUT_DIR/08_acc_graph.log"

# Cleanup
kill_server

echo
echo "════════════════════════════════════════════════════════════════"
echo "Done. All 9 logs in $OUT_DIR"
ls -la "$OUT_DIR"
echo
echo "To screenshot, run something like:"
echo "    for f in $OUT_DIR/0[1-9]_bench*.log $OUT_DIR/0[1-9]_acc*.log; do"
echo "        echo \"=== \$f ===\""
echo "        cat \"\$f\""
echo "        echo"
echo "        read -p 'Press enter for next' _"
echo "    done"
