#!/usr/bin/env bash
# Run every Stage 2 experiment (E2+E3, E4, Tent shuffle orders) for the given checkpoints,
# then analyze. Safe to run in the background:
#
#   nohup bash scripts/run_stage2.sh checkpoints/*/final.pt > results/run_stage2.log 2>&1 &
#   tail -f results/run_stage2.log
#
# Optional environment variables:
#   OUT=dir         where results go (default: results)
#   EVAL_ARGS=...   extra eval.py flags, e.g. for a quick test:
#                   EVAL_ARGS="--n-examples 200 --corruptions gaussian_noise,fog"
#   REPEATS=3       Tent continual runs per model; it varies from run to run on GPU (default 3)
#   REPEATS_ONLY=1  only the extra Tent continual runs (for models whose other runs are done)
set -uo pipefail
export PYTHONUNBUFFERED=1  # so the log shows each condition as it finishes
if [ $# -eq 0 ]; then
  echo "usage: bash scripts/run_stage2.sh <checkpoint.pt> [...]" >&2
  exit 1
fi
OUT="${OUT:-results}"
REPEATS="${REPEATS:-3}"
REPEATS_ONLY="${REPEATS_ONLY:-0}"
read -r -a EXTRA <<< "${EVAL_ARGS:-}"

run() {
  echo "=== $(date +%H:%M:%S) python eval.py $*"
  python eval.py "$@" --out-dir "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} || echo "!!! FAILED: $*"
}

for ckpt in "$@"; do
  if [ "$REPEATS_ONLY" != 1 ]; then
    # E2 + E3: clean + all 75 conditions, episodic
    run --model "$ckpt" --method none --protocol episodic --clean
    run --model "$ckpt" --method bn --protocol episodic --clean
    run --model "$ckpt" --method tent --protocol episodic --clean
    # E4: difficult streams
    for method in bn tent; do
      run --model "$ckpt" --method $method --protocol continual
      run --model "$ckpt" --method $method --protocol small_batch
    done
    # 3 shuffle orders for the severity-5 Tent runs
    for seed in 1 2 3; do
      run --model "$ckpt" --method tent --protocol episodic --severities 5 --shuffle-seed $seed
    done
  fi
  # Extra Tent continual runs, named ..._rep2, ..._rep3
  for ((r = 2; r <= REPEATS; r++)); do
    run --model "$ckpt" --method tent --protocol continual --tag rep$r
  done
done

echo "=== $(date +%H:%M:%S) python analyze.py"
python analyze.py --results "$OUT"
echo "=== $(date +%H:%M:%S) done"
