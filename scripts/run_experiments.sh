#!/usr/bin/env bash
# All training runs reported in the paper. Each run resumes automatically from its last checkpoint
# and is skipped when it is already finished, so the script can be interrupted and restarted.
#
#   bash scripts/run_experiments.sh main         # Table 1: supervised, MixMatch, IA-MixMatch, FreeMatch
#   bash scripts/run_experiments.sh losses       # Table 1: class-balanced and asymmetric loss (+ CB inside SSL)
#   bash scripts/run_experiments.sh ablation     # Table 3: ablation with 5% labels
#   bash scripts/run_experiments.sh sensitivity  # Figure 4: temperature T and controlled imbalance
#   bash scripts/run_experiments.sh all
#
# Paths can be overridden with IMG_DIR, SPLIT_DIR and RUNS. One NVIDIA L4 needs about 0.3 s per iteration.
set -euo pipefail
cd "$(dirname "$0")/.."
IMG_DIR=${IMG_DIR:-data/cxr14_256}
SPLIT_DIR=${SPLIT_DIR:-splits}
RUNS=${RUNS:-runs}
COMMON="--img_dir $IMG_DIR --split_dir $SPLIT_DIR --out_root $RUNS --eval_every 500"
declare -A ITERS=([2]=4000 [5]=4000 [10]=6000 [15]=8000 [20]=8000)   # about 16 passes over the labeled set

run() { echo ">>> python train.py $*"; python train.py "$@" $COMMON; }

main() {
  for seed in 0 1 2; do
    for r in 2 5 10 15 20; do
      for m in supervised mixmatch ia_mixmatch; do
        run --method $m --ratio $r --seed $seed --iters ${ITERS[$r]} --tag p2
      done
    done
    for r in 5 20; do
      run --method freematch --ratio $r --seed $seed --iters ${ITERS[$r]} --tag p2
    done
  done
}

losses() {
  for seed in 0 1 2; do
    for r in 5 20; do
      for loss in cb asl; do
        run --method supervised --loss $loss --ratio $r --seed $seed --iters ${ITERS[$r]} --tag p2
      done
      for m in mixmatch ia_mixmatch; do
        run --method $m --loss cb --ratio $r --seed $seed --iters ${ITERS[$r]} --tag p2
      done
    done
  done
}

ablation() {
  local variants=(
    "--method mixmatch --lw_temp"                   # label-wise temperature (Sec. 2.2)
    "--method mixmatch --pcs_naive"                 # distribution alignment only (eta = 1)
    "--method mixmatch --pcs --pcs_mode integral"   # additive shift with an integral controller
    "--method mixmatch --pcs"                       # MPS only
    "--method mixmatch --acw"                       # ACW only
    "--method mixmatch --acw_hard"                  # ACW with a hard threshold
    "--method ia_mixmatch --eta 1"                  # MPS with alignment (eta = 1) + ACW
  )
  for seed in 0 1 2; do
    for v in "${variants[@]}"; do
      # shellcheck disable=SC2086
      run $v --ratio 5 --seed $seed --iters 4000 --tag p3
    done
  done
}

sensitivity() {
  for seed in 0 1 2; do
    for T in 0.3 0.7; do
      for m in ia_mixmatch mixmatch; do
        run --method $m --T $T --ratio 5 --seed $seed --iters 4000 --tag p5
      done
    done
  done
  for seed in 0 1; do
    for k in 2 4 8; do
      for m in ia_mixmatch mixmatch; do
        run --method $m --imb $k --ratio 10 --seed $seed --iters 6000 --tag p5
      done
    done
  done
}

case "${1:-}" in
  main) main ;;
  losses) losses ;;
  ablation) ablation ;;
  sensitivity) sensitivity ;;
  all) main; losses; ablation; sensitivity ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
