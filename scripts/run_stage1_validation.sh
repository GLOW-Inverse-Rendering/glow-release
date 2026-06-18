#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/run_stage1_validation.sh <real|synthetic> <scene> <stage1_exp_dir>" >&2
  exit 2
fi

DATASET_KIND="$1"
SCENE="$2"
EXP_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

case "$DATASET_KIND" in
  real)
    CONF="real"
    ;;
  synthetic)
    CONF="syn"
    ;;
  *)
    echo "Expected dataset kind to be 'real' or 'synthetic', got: $DATASET_KIND" >&2
    exit 2
    ;;
esac

DATASET_DIR="datasets/$DATASET_KIND/$SCENE"
if [ ! -d "$DATASET_DIR" ]; then
  echo "Missing dataset directory: $DATASET_DIR" >&2
  exit 1
fi
if ! compgen -G "$EXP_DIR/checkpoints/ckpt_*.pth" >/dev/null; then
  echo "Missing numeric NeuS checkpoint in: $EXP_DIR/checkpoints" >&2
  exit 1
fi

export MI_DEFAULT_VARIANT=cuda_ad_rgb
export OPENCV_IO_ENABLE_OPENEXR=1

CMD=(
  python -u exp_runner.py
  "case=$SCENE"
  "conf=$CONF"
  mode=validate_mesh
  "hydra.run.dir=$EXP_DIR"
  is_continue=true
  conf/model=principled
)

if [ "${PRINT_ONLY:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
