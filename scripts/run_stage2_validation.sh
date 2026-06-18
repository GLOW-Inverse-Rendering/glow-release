#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/run_stage2_validation.sh <real|synthetic> <scene> <stage2_exp_dir>" >&2
  exit 2
fi

DATASET_KIND="$1"
SCENE="$2"
STAGE2_EXP_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

case "$DATASET_KIND" in
  real)
    CONF="real"
    BSDF="principledmy"
    DEFAULT_SKIP_ITERS=1200000
    ;;
  synthetic)
    CONF="syn"
    BSDF="principled"
    DEFAULT_SKIP_ITERS=700000
    ;;
  *)
    echo "Expected dataset kind to be 'real' or 'synthetic', got: $DATASET_KIND" >&2
    exit 2
    ;;
esac

BASE_SCENE="${SCENE%_colocated}"
BASE_SCENE="${BASE_SCENE%_natural}"
DATASET_DIR="datasets/$DATASET_KIND/$SCENE"
SCENE_PATH="datasets/mitsuba3_scenes/$BASE_SCENE/scene_principled.xml"

if [ ! -d "$DATASET_DIR" ]; then
  echo "Missing dataset directory: $DATASET_DIR" >&2
  exit 1
fi
if [ ! -f "$SCENE_PATH" ]; then
  echo "Missing Mitsuba scene file: $SCENE_PATH" >&2
  exit 1
fi
if ! compgen -G "$STAGE2_EXP_DIR/checkpoints/ckpt_*.pth" >/dev/null; then
  echo "Missing numeric NeuS checkpoint in: $STAGE2_EXP_DIR/checkpoints" >&2
  exit 1
fi
if ! compgen -G "$STAGE2_EXP_DIR/mitsuba/checkpoints/[0-9]*.ckpt" >/dev/null; then
  echo "Missing numeric Mitsuba checkpoint in: $STAGE2_EXP_DIR/mitsuba/checkpoints" >&2
  exit 1
fi

latest_numeric_pth_iter() {
  local dir="$1"
  local best=0
  local file name iter
  shopt -s nullglob
  for file in "$dir"/ckpt_*.pth; do
    name="$(basename "$file")"
    iter="${name#ckpt_}"
    iter="${iter%.pth}"
    if [[ "$iter" =~ ^[0-9]+$ ]] && [ "$iter" -gt "$best" ]; then
      best="$iter"
    fi
  done
  shopt -u nullglob
  if [ "$best" -eq 0 ]; then
    return 1
  fi
  printf '%s\n' "$best"
}

CKPT_ITER="$(latest_numeric_pth_iter "$STAGE2_EXP_DIR/checkpoints")"
END_ITER="${END_ITER:-$((CKPT_ITER + 10000))}"
BSDF_SAMPLE_SKIP_ITERS="${BSDF_SAMPLE_SKIP_ITERS:-$DEFAULT_SKIP_ITERS}"

export MI_DEFAULT_VARIANT=cuda_ad_rgb
export OPENCV_IO_ENABLE_OPENEXR=1

CMD=(
  python -u exp_runner.py
  "case=$SCENE"
  "conf=$CONF"
  mitsuba_renderer=default
  "mitsuba_renderer.scene_path=$SCENE_PATH"
  mitsuba_renderer.mesh_path=dummy
  "mitsuba_renderer.out_dir=$STAGE2_EXP_DIR/mitsuba"
  "hydra.run.dir=$STAGE2_EXP_DIR"
  is_continue=true
  "conf.train.end_iter=$END_ITER"
  mitsuba_renderer/config/geometry_type=vol_bsdf_adjoint
  mitsuba_renderer.config.geometry_type.options.grad_scale=0.1
  mitsuba_renderer.config.geometry_type.options.reflectance_grad_scale=1.0
  conf.train.val_freq=10000
  conf.train.color_weight=0.0
  mitsuba_renderer.config.loss.use_radiosity=true
  mitsuba_renderer.config.loss.rhs_radiosity=true
  "mitsuba_renderer.config.render.bsdf_sample_skip_iters=$BSDF_SAMPLE_SKIP_ITERS"
  conf.train.batch_size=512
  conf/model=principled
  mitsuba_renderer.config.train.orient_loss_period_first_half=2
  mitsuba_renderer.config.train.protective_loss_type=orient
  mitsuba_renderer.config.train.subsample_radiosity=32
  conf.train.migrate_ckpt=true
  mitsuba_renderer.config.loss.use_visibility_loss=false
  conf.train.learning_rate=5e-5
  "mitsuba_renderer.config.render.bsdf=$BSDF"
  mitsuba_renderer.config.loss.loss_type=l1_orient
  conf.train.prune_outside_cam=false
  conf.train.use_shadow_mask=false
  mode=validate_image_4
  conf.dataset.use_pcd=false
  mitsuba_renderer.config.render.freeze_flashlight_intensity=false
)

if [ "${PRINT_ONLY:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
