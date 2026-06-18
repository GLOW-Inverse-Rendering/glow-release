# glow-release Manifest

Updated: 2026-06-16

This repository is the code release for the current GLOW/WildLight runtime.
Datasets, Mitsuba scene assets, and experiment outputs are kept on disk in this
workspace for testing and review, but are ignored by git and should be released
separately.

Current release plan:

- Ship this git repository as the code release.
- Ship the trained checkpoint separately.
- Ship the MAW dataset separately.
- Defer the data-processing pipeline release until later, or provide it upon
  request.

## Included In Git

- WildLight/GLOW Python entrypoints:
  - `exp_runner.py`
  - `material_train.py`
  - `reextract_mesh_wrapper.py`
- Runtime modules:
  - `models/`
  - `wildlightutils/`
  - `config_hydra/`
- Folded Mitsuba/NeRAD runtime code needed by the current bridge:
  - `integrations/inverse-neural-radiosity/`
- Environment/reference docs:
  - `README.md`
  - `requirements.txt`
  - `LICENSE`

## Ignored From Git

- Local datasets:
  - `datasets/`
  - `integrations/inverse-neural-radiosity/data/`
- Pipeline test outputs:
  - `exp/`
  - `official_test_runs/`
- Generated/runtime clutter:
  - `__pycache__/`
  - `*.pyc`
  - `*.modified.xml`
  - `*.modified_roughness.xml`
  - local debug/export output directories

## Notes

- `models/giphysicalshader.py` intentionally retains visualization dump logic
  for inspecting material/radiosity behavior.
- The folded integration still contains inherited compatibility code for the
  current Mitsuba bridge. A future cleaned PyTorch implementation is planned
  separately.
- The final README should be rewritten after the release code and companion
  artifacts are fixed.
