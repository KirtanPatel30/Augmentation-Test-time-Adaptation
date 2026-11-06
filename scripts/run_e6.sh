#!/usr/bin/env bash
# E6: Tent on the continual stream with a periodic reset (as in RDumb), for the models where
# plain continual Tent drifts. 3 reset intervals x 3 runs per model, then analyze.
#
#   OUT=results bash scripts/run_e6.sh runs/standard_seed2/final.pt runs/cutmix_seed1/final.pt
#
# Reset intervals are in batches of 200 and do not line up with the 50-batch conditions:
#   30 = 6,000 images, 100 = 20,000, 320 = 64,000 (RDumb's 1,000 steps x 64 images).
# Optional environment variables: OUT, EVAL_ARGS (as in run_stage2.sh), RESETS, REPEATS.
set -uo pipefail
export PYTHONUNBUFFERED=1
if [ $# -eq 0 ]; then
  echo "usage: bash scripts/run_e6.sh <checkpoint.pt> [...]" >&2
  exit 1
fi
OUT="${OUT:-results}"
RESETS="${RESETS:-30 100 320}"
REPEATS="${REPEATS:-3}"
read -r -a EXTRA <<< "${EVAL_ARGS:-}"

run() {
  echo "=== $(date +%H:%M:%S) python eval.py $*"
  python eval.py "$@" --out-dir "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} || echo "!!! FAILED: $*"
}

for ckpt in "$@"; do
  # analyze.py needs the no-adaptation run of each model as the harm reference
  run --model "$ckpt" --method none --protocol continual
  for n in $RESETS; do
    for ((r = 1; r <= REPEATS; r++)); do
      tag=$([ "$r" -gt 1 ] && echo "--tag rep$r")
      run --model "$ckpt" --method tent --protocol continual --reset-every "$n" $tag
    done
  done
done

echo "=== $(date +%H:%M:%S) python analyze.py"
python analyze.py --results "$OUT"
echo "=== $(date +%H:%M:%S) done"
