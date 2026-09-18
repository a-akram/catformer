## Running training without installing the package

You don't need `pip3 install .` to develop against this repo — that's a
non-editable install, so edits to `src/cat_finder/...` won't take effect
until you reinstall. Instead, run everything with `PYTHONPATH=src` from the
repo root, exactly like the existing example commands:

```bash
cd catfinder   # repo root — config paths below are relative to here
PYTHONPATH=src python3 training/train_cat.py --config config/config.yaml --run pretraining
PYTHONPATH=src python3 training/evaluation.py --trainings_dir results/pretraining/
PYTHONPATH=src python3 test_backbones.py
PYTHONPATH=src python3 test_gravnet_regression.py
```

If you'd rather not type `PYTHONPATH=src` every time, `export PYTHONPATH=src`
once per shell session, or use `pip install -e .` (editable install) instead
— either avoids the "have to reinstall after every edit" problem of a plain
`pip3 install .`.

### Choosing a backbone

`CDCNet`'s `backbone` argument defaults to `"gravnet"`, so **unpatched**
`train_cat.py` (which doesn't pass `backbone=` at all) runs GravNet
unchanged — no config or code edit needed for a GravNet run.

To run HEPT (or any other registered backbone), apply
`train_cat_patch.diff` to `training/train_cat.py` and add a `backbone:` key
plus that backbone's `*_kwargs:` block to `config/config.yaml`'s `model:`
section (see `config_additions.yaml`):

```yaml
model:
  backbone: hept   # or "gravnet"; omit entirely to default to gravnet
  hept_kwargs:
    h_dim: 64
    n_layers: 4
    ...
```

### wandb

`train_cat.py` calls `wandb.init(project="CAT_Finder", config=config)`
unconditionally — with no wandb account configured, this will prompt for an
API key or hang in a non-interactive shell. Before running:

```bash
wandb login              # to log real training curves, or
export WANDB_MODE=offline # to skip needing an account entirely
```

`WANDB_MODE=offline` still writes local run logs (under `./wandb/`) without
any network calls — good enough for a smoke test or a quick benchmark run
where you don't need the dashboard.

### Dataset paths and caching — what actually needs changing

The example dataset (`paper_dataset/cdchits.csv`) works with `config.yaml`'s
defaults out of the box — you do **not** need to edit `input_dir` or
`val_input_dir` just to get a first run going. `CDCDataset.raw_file_names`
picks up *any* `.csv` file in `sampledir`; `evt_type`/`val_type` only label
the cached filename (see below), they don't filter which file gets loaded.

Two things worth knowing before you rely on this for real comparisons:

1. **With the default config, train and val are the same 10 events**,
   because `input_dir` and `val_input_dir` both point at `./paper_dataset/`.
   Fine for confirming the pipeline runs end-to-end; not meaningful for
   real GravNet-vs-HEPT numbers. Point `val_input_dir` at a genuinely
   separate directory once you're past the smoke-test stage.

2. **The processed `.pt` cache does not invalidate on config changes.**
   `CDCDataset` caches processed graphs to
   `{torch_dir}/{evt_type}_{samples_per_file}_{n_batches}.pt` — that
   filename depends only on `evt_type`/`samples_per_file`/`n_batches`, **not**
   on `input_features`, `truth`, `scaling`, or `clipping`. If you change
   `input_features` in `config.yaml` (e.g. while iterating on backbone
   input dims) but keep the same `evt_type`/`samples_per_file`, you'll
   silently get the **old** cached tensors — usually surfacing as a
   confusing shape mismatch rather than an obvious "stale cache" error.
   Delete the relevant `.pt` file(s) under `torch_dir` after any change to
   `input_features`, `truth`, `scaling`, or `clipping`:
   ```bash
   rm ./paper_dataset/train_*.pt ./paper_dataset/val_*.pt
   ```
