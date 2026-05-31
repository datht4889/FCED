#!/bin/bash
# =============================================================================
# Sensitivity sweep for CPL+PRAGAS (response to R1 W3 / R2 W3)
# =============================================================================
# Sweeps the hyperparameters that the saveSynData-branch train.py supports,
# plus two minimal CLI additions (--ml_margin, --lambda_A) on top of that base.
#
# Log layout (one file per task x value, grouped by hyperparameter):
#   log/sensitivity-paper/rho/<Task>_rho_<value>.log
#   log/sensitivity-paper/beta_1/<Task>_b1_<value>.log
#   log/sensitivity-paper/beta_2/<Task>_b2_<value>.log
#   log/sensitivity-paper/ml_margin/<Task>_m<value>.log
#   log/sensitivity-paper/lambda_A/<Task>_la_<value>.log
#
# CLI <-> paper mapping (Table 3, CPL framework):
#   --rho            = SAM perturbation radius rho     (default 0.05)
#   --mixup_loss_1   = beta_1                          (default 0.25)
#   --mixup_loss_2   = beta_2                          (default 0.25)
#   --ml_margin      = margin m (NegativeCosSimLoss temperature)  (default 1.0)
#   --lambda_A       = lambda_A (composite contrastive loss weight) (default 1.0)
#
# NOT swept here (mechanism absent in saveSynData branch):
#   - lambda_rho: requires dynamic-rho scaling (GuidedSAM), not in this train.py
#   - lambda_D:   requires explicit distillation loss term, not in this train.py
# =============================================================================

set -euo pipefail

# Locate the CPL train.py (FCED/FCRE/CPL/, sibling under this script's dir).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPL_DIR="$SCRIPT_DIR/FCRE/CPL"
cd "$CPL_DIR"

# Common flags (match the saveSynData branch defaults).
COMMON_FLAGS="\
  --num_k 5 \
  --num_gen 5 \
  --gen 1 \
  --mixup \
  --SAM"

# run_one <task> <param_subdir> <filename_label> [extra train.py flags...]
#   Final log path: $SCRIPT_DIR/log/sensitivity-paper/<param_subdir>/<task>_<filename_label>.log
run_one() {
  local task="$1"; local param_dir="$2"; local fname="$3"; shift 3
  local logdir="$SCRIPT_DIR/log/sensitivity-paper/$param_dir"
  mkdir -p "$logdir"
  local logfile="$logdir/${task}_${fname}.log"
  echo "[sensitivity] task=$task param=$param_dir cell=$fname -> $logfile"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TOKENIZER_PARALELISM=True \
    python train.py --task_name "$task" $COMMON_FLAGS "$@" 2>&1 | tee "$logfile"
}

# ----------------------------------------------------------------------------
# (i) SAM perturbation radius rho (--rho)  default 0.05  -> sweep {0.01, 0.05, 0.1, 0.2, 0.5}
# Output: log/sensitivity-paper/rho/<Task>_rho_<value>.log
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for r in 0.01 0.05 0.1 0.2 0.5; do
    run_one "$task" "rho" "rho_${r}" \
      --rho "$r" \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 0.25 \
      --ml_margin 1.0 \
      --lambda_A 1.0
  done
done

# ----------------------------------------------------------------------------
# (ii) beta_1 sweep (--mixup_loss_1)  default 0.25  -> sweep {0.1, 0.25, 0.5, 1.0}
# Output: log/sensitivity-paper/beta_1/<Task>_b1_<value>.log
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for b1 in 0.1 0.25 0.5 1.0; do
    run_one "$task" "beta_1" "b1_${b1}" \
      --rho 0.05 \
      --mixup_loss_1 "$b1" \
      --mixup_loss_2 0.25 \
      --ml_margin 1.0 \
      --lambda_A 1.0
  done
done

# ----------------------------------------------------------------------------
# (iii) beta_2 sweep (--mixup_loss_2)  default 0.25  -> sweep {0.1, 0.25, 0.5, 1.0}
# Output: log/sensitivity-paper/beta_2/<Task>_b2_<value>.log
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for b2 in 0.1 0.25 0.5 1.0; do
    run_one "$task" "beta_2" "b2_${b2}" \
      --rho 0.05 \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 "$b2" \
      --ml_margin 1.0 \
      --lambda_A 1.0
  done
done

# ----------------------------------------------------------------------------
# (iv) margin m sweep (--ml_margin)  default 1.0  -> sweep {0.25, 0.5, 1.0, 2.0, 4.0}
# Wired to NegativeCosSimLoss(temperature=ml_margin) at the L_ML site.
# Output: log/sensitivity-paper/ml_margin/<Task>_m<value>.log
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for m in 0.25 0.5 1.0 2.0 4.0; do
    run_one "$task" "ml_margin" "m${m}" \
      --rho 0.05 \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 0.25 \
      --ml_margin "$m" \
      --lambda_A 1.0
  done
done

# ----------------------------------------------------------------------------
# (v) lambda_A sweep (--lambda_A)  default 1.0  -> sweep {0.25, 0.5, 1.0, 2.0, 4.0}
# Wired to: sum_loss += lambda_A * 0.5 * loss  (composite contrastive loss).
# Output: log/sensitivity-paper/lambda_A/<Task>_la_<value>.log
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for w in 0.25 0.5 1.0 2.0 4.0; do
    run_one "$task" "lambda_A" "la_${w}" \
      --rho 0.05 \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 0.25 \
      --ml_margin 1.0 \
      --lambda_A "$w"
  done
done
