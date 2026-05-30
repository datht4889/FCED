#!/bin/bash
# =============================================================================
# Sensitivity sweep for CPL+PRAGAS (response to R1 W3 / R2 W3)
# =============================================================================
# Sweeps lambda_rho, beta_1, beta_2 on CPL+PRAGAS framework over FewRel + TACRED.
# Six seeds per cell (matches main-results protocol).
#
# CLI <-> paper Table 3 mapping:
#   --rho_weight     = lambda_rho   (default 6)
#   --mixup_loss_1   = beta_1       (default 0.25)
#   --mixup_loss_2   = beta_2       (default 0.25)
#
# NOTE: margin m (Table 3 default 1.0) and lambda_A / lambda_D are not currently
# exposed as CLI args in FCRE/CPL/train.py. To sweep them, either (a) add CLI
# flags to train.py, or (b) edit the relevant constants in mixup.py / add_loss.py
# before each run. Marked as TODO at the bottom of this script.
# =============================================================================

set -euo pipefail

# Locate the CPL train.py relative to this script's parent (PRAGAS root).
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CPL_DIR="$ROOT_DIR/FCRE/CPL"
cd "$CPL_DIR"

# Common flags (match fewrel_5shot_pragas.sh / tacred_5shot_pragas.sh setup).
COMMON_FLAGS="\
  --model bert \
  --output-size 768 \
  --max-length 256 \
  --num_k 5 \
  --gen 1 \
  --num_gen 5 \
  --decay 0.01 \
  --mixup \
  --SAM \
  --sam_optimizer ASAM \
  --rho 0.1 \
  --dynamic-rho \
  --distill \
  --distill_type RKD \
  --distill_loss_weight 0 \
  --distill_top_k 10 \
  --batch-size 16"

run_one() {
  local task="$1"; local label="$2"; shift 2
  local logdir="$ROOT_DIR/log/sensitivity-paper/$label"
  mkdir -p "$logdir"
  local logfile="$logdir/${task}_${label}.log"
  echo "[sensitivity] task=$task label=$label -> $logfile"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TOKENIZER_PARALELISM=True \
    python train.py --task_name "$task" $COMMON_FLAGS "$@" 2>&1 | tee "$logfile"
}

# ----------------------------------------------------------------------------
# (i) lambda_rho sweep (--rho_weight)  default 6  -> sweep {0.5, 1, 2, 3, 4, 6, 8, 10}
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for rw in 0.5 1 2 3 4 6 8 10; do
    run_one "$task" "rho_weight/rw${rw}" \
      --rho_weight "$rw" \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 0.25
  done
done

# ----------------------------------------------------------------------------
# (ii) beta_1 sweep (--mixup_loss_1)  default 0.25  -> sweep {0.1, 0.25, 0.5, 1.0}
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for b1 in 0.1 0.25 0.5 1.0; do
    run_one "$task" "beta_1/b1_${b1}" \
      --rho_weight 6 \
      --mixup_loss_1 "$b1" \
      --mixup_loss_2 0.25
  done
done

# ----------------------------------------------------------------------------
# (iii) beta_2 sweep (--mixup_loss_2)  default 0.25  -> sweep {0.1, 0.25, 0.5, 1.0}
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for b2 in 0.1 0.25 0.5 1.0; do
    run_one "$task" "beta_2/b2_${b2}" \
      --rho_weight 6 \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 "$b2"
  done
done

# ----------------------------------------------------------------------------
# (iv) margin m sweep (--ml_margin)  default 1.0  -> sweep {0.25, 0.5, 1.0, 2.0, 4.0}
# Wired in train.py to NegativeCosSimLoss(temperature=ml_margin) at the L_ML site.
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for m in 0.25 0.5 1.0 2.0 4.0; do
    run_one "$task" "ml_margin/m_${m}" \
      --rho_weight 6 \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 0.25 \
      --ml_margin "$m"
  done
done

# ----------------------------------------------------------------------------
# (v) lambda_A / lambda_D sweep  default 1.0  -> sweep {0.25, 0.5, 1.0, 2.0, 4.0}
# Wired in train.py:
#   lambda_A multiplies the 0.5 weight on the composite contrastive loss `loss`
#   lambda_D multiplies distill_loss_weight on the distill loss term
# ----------------------------------------------------------------------------
for task in FewRel Tacred; do
  for w in 0.25 0.5 1.0 2.0 4.0; do
    run_one "$task" "lambda_AD/w_${w}" \
      --rho_weight 6 \
      --mixup_loss_1 0.25 \
      --mixup_loss_2 0.25 \
      --lambda_A "$w" \
      --lambda_D "$w"
  done
done
