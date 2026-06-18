# GLOW Release

[![Project](https://img.shields.io/badge/Project-Page-blue)](https://glow-inverse-rendering.github.io/)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-green)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wu_GLOW_Global_Illumination-Aware_Inverse_Rendering_of_Indoor_Scenes_Captured_with_CVPRF_2026_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2511.22857-b31b1b.svg)](https://arxiv.org/abs/2511.22857)
[![MAW Project](https://img.shields.io/badge/MAW-Project-blueviolet)](https://measuredalbedo.github.io/)
[![MAW Code](https://img.shields.io/badge/MAW-Code-black)](https://github.com/MeasuredAlbedo/code)

This is a research code release intended to reproduce the current GLOW results.
The code has been cleaned enough to provide the released pipeline and scripts,
but it is not yet a minimal or fully refactored codebase.

Dataset-processing scripts are not included in this release. We may release them
later, and please contact us if they would be useful for your work.

## Roadmap

This release keeps the Mitsuba/NeRAD-derived rendering path used for the current
results. We are moving toward a cleaner pure-PyTorch implementation with
multi-GPU training support and larger whole-room scene examples.

## MAW 2.0 Measurements

For measured-albedo evaluation, use the MAW 2.0 / GLOW measurement release:

[Download MAW 2.0 / GLOW measurements](https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/glow_maw2_measurements_release.zip)

The archive provides measured-albedo metadata, masks, and measurement files
used by the MAW evaluation protocol. See the
[MAW project page](https://measuredalbedo.github.io/) and
[MAW code repository](https://github.com/MeasuredAlbedo/code) for the evaluator
and protocol details.

## Layout

- `exp_runner.py`: main training/evaluation entrypoint.
- `material_train.py`: material-stage and material-validation entrypoint.
- `models/`: model, renderer, dataset, and material code.
- `config_hydra/`: curated Hydra configs for the release path.
- `integrations/inverse-neural-radiosity/`: runtime components required by the
  current Mitsuba bridge.
- `scripts/`: small release helpers for setting up downloaded assets.

## Prepare Data

Run these commands from the `glow-release` directory.

Download and install the GLOW dataset:

```bash
python3 scripts/setup_datasets.py download --output-dir /path/to/data
python3 scripts/setup_datasets.py install --source /path/to/data/glow_dataset_release
```

This downloads the [dataset archive](https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/glow_dataset_release.zip)
and installs:

```text
datasets/real/
datasets/synthetic/
datasets/mitsuba3_scenes/
```

## Setup

Run the setup, evaluation, and training commands from the `glow-release`
directory.

### Use The Prebuilt Image

For most users, use the published Docker image:

```text
public.ecr.aws/z8e4h4q6/glow-project/glow-env:latest
```

```bash
docker pull public.ecr.aws/z8e4h4q6/glow-project/glow-env:latest
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/glow-release \
  -w /workspace/glow-release \
  public.ecr.aws/z8e4h4q6/glow-project/glow-env:latest \
  bash
```

After the container starts, run the evaluation or training commands below from
`/workspace/glow-release`.

### Build The Image Locally

If you want to rebuild the Docker image locally, download the patched
Mitsuba/Dr.Jit wheel bundle before running `docker build`. The Dockerfile
installs these wheels directly. The corresponding patched Mitsuba source is
published at
[GLOW-Inverse-Rendering/mitsuba3-myprincipled-fork](https://github.com/GLOW-Inverse-Rendering/mitsuba3-myprincipled-fork).

```bash
curl -L -o glow-mitsuba3-patched-wheels.zip \
  https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/glow-mitsuba3-patched-wheels.zip
unzip -q glow-mitsuba3-patched-wheels.zip
mv mitsuba3_patched_wheels mitsuba3_output

DOCKER_BUILDKIT=1 docker build -t glow-env .
```

Then start the locally built image:

```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/glow-release \
  -w /workspace/glow-release \
  glow-env \
  bash
```

## Evaluate Released Checkpoints

After installing the dataset and entering the Docker environment, download the
released checkpoints:

```bash
python3 scripts/setup_checkpoints.py download
python3 scripts/setup_checkpoints.py list
```

This downloads the [checkpoint archive](https://dzwmyzdewsbxi.cloudfront.net/projects/glow-project/glow-checkpoint-release.zip).

Install the checkpoint stage you want to evaluate into an experiment directory,
then run the matching validation script. The `<exp_dir>` is where
`setup_checkpoints.py install` writes checkpoint files and where the validation
script reads them.

```bash
python3 scripts/setup_checkpoints.py install \
  --scene coffee_table_colocated \
  --stage stage3 \
  --exp-dir exp/coffee_table_colocated/nomask_material

bash scripts/run_stage3_validation.sh \
  real \
  coffee_table_colocated \
  exp/coffee_table_colocated/nomask_material
```

Validation scripts always run from an existing experiment directory:

```bash
bash scripts/run_stage1_validation.sh <real|synthetic> <scene> <stage1_exp_dir>
bash scripts/run_stage2_validation.sh <real|synthetic> <scene> <stage2_exp_dir>
bash scripts/run_stage3_validation.sh <real|synthetic> <scene> <stage3_exp_dir>
```

## Train from Scratch

Training runs as a staged pipeline with explicit experiment-directory handoff.
Stage 1 writes geometry initialization checkpoints. Stage 2 takes the Stage 1
experiment directory, copies the latest numeric checkpoint into a new Stage 2
directory, and runs the physically based rendering stage. Stage 3 takes the
Stage 2 experiment directory, copies the latest numeric checkpoint pair into a
new Stage 3 directory, and runs material optimization.

```bash
bash scripts/run_stage1.sh <real|synthetic> <scene> <stage1_exp_dir>
bash scripts/run_stage2.sh <real|synthetic> <scene> <stage1_exp_dir> <stage2_exp_dir>
bash scripts/run_stage3.sh <real|synthetic> <scene> <stage2_exp_dir> <stage3_exp_dir>
```

Use fresh destination experiment directories for Stage 2 and Stage 3 so old
checkpoint files are not mixed with the handoff state.

You can validate after any stage:

```text
bash scripts/run_stage1_validation.sh <real|synthetic> <scene> <stage1_exp_dir>
bash scripts/run_stage2_validation.sh <real|synthetic> <scene> <stage2_exp_dir>
bash scripts/run_stage3_validation.sh <real|synthetic> <scene> <stage3_exp_dir>
```

## Reference

Released scene groups:

```text
synthetic: bedroom, shelf, kitchen_counter
real:      coffee_table, window_sill, shoe_rack, table
```

Most scene groups include both `natural` and `colocated` train/validation
cases. The real `table` scene includes `colocated` train/validation cases.

Checkpoint stages:

```text
stage1  # geometry initialization
stage2  # physically based rendering
stage3  # material optimization
```

The checkpoint archive is organized by scene and stage:

```text
<scene>/
  01_wildlight/            # geometry initialization checkpoint
  02_refinement2_neus/     # refinement checkpoint
  02_refinement2_mitsuba/  # refinement Mitsuba checkpoint
  03_material_neus/        # material optimization checkpoint
  03_material_mitsuba/     # material Mitsuba checkpoint
```
