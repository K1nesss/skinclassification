# Skin Disease Classification

Course design project for multi-source skin image classification and explainability.

## Environment

Use the existing conda environment:

```powershell
& "$env:USERPROFILE\anaconda3\envs\pytorch\python.exe" -m pip install -r requirements.txt
```

## Data

Expected raw files:

```text
data/raw/Dermnet/train.zip
data/raw/Dermnet/test.zip
data/raw/SkinDisNet_2.zip
data/raw/Mendeley Skin Disease Classification Dataset/*.zip
data/raw/SCIN/...
```

The current project uses a rebuilt 10-class dataset. `data/raw/` is only the source pool.
The build script extracts the selected raw labels, filters low-quality images, writes clean
images to `data/new/`, rejected images to `data/pass/`, and split train/val/test images to
`data/processed/`.

```powershell
& "$env:USERPROFILE\anaconda3\envs\pytorch\python.exe" scripts/build_new_dataset.py --config config.yaml --clean
```

Main output files:

```text
data/new/manifest.csv
data/pass/rejected_manifest.csv
data/pass/quality_report.csv
data/pass/quality_summary.csv
data/pass/quality_report.md
data/pass/quality_figures/
data/interim/split_samples.csv
```

## Training

Smoke run:

```powershell
& "$env:USERPROFILE\anaconda3\envs\pytorch\python.exe" train.py --config config.yaml --model resnet18 --epochs 1 --limit-train-batches 2 --limit-val-batches 2
```

The default training setup uses 384px inputs and a hard class-aware sampler. The
sampler keeps inverse-frequency class balancing, then gives extra sampling weight
to the currently most confused classes:

```text
eczema
dermatitis
psoriasis_lichen_planus
```

Full model names:

```text
resnet18
densenet121
efficientnet_b0
mobilenet_v3_small
convnext_tiny
convnext_base
swin_s
swin_b
```

Formal runs should be done on the 4090 server for `convnext_base` and `swin_s` or `swin_b`.

## Evaluation

```powershell
& "$env:USERPROFILE\anaconda3\envs\pytorch\python.exe" evaluate.py --config config.yaml --checkpoint outputs/checkpoints/densenet121_best.pt --split test
```

## Demo

```powershell
& "$env:USERPROFILE\anaconda3\envs\pytorch\python.exe" run_app.py --config config.yaml
```

The demo is for course research and display only. It is not a medical diagnosis tool.

## Server Full Run

On the server, put large data and outputs under:

```text
/mnt/disk002/skinclassification/
```

Then from the code directory:

```bash
conda activate pytorch
bash run_all_experiments.sh
```

The script creates `data` and `outputs` symlinks to `/mnt/disk002/skinclassification/`, builds the new 10-class cleaned dataset, generates dataset figures, trains the full model set, and evaluates each checkpoint on `test`.

Default server model set:

```text
resnet18 densenet121 efficientnet_b0 mobilenet_v3_small convnext_tiny convnext_base swin_s swin_b
```

Override examples:

```bash
MODELS="resnet18 densenet121 efficientnet_b0 convnext_base swin_s" bash run_all_experiments.sh
EPOCHS=5 bash run_all_experiments.sh
BATCH_SIZE_SWIN_B=8 bash run_all_experiments.sh
```

Run only the strongest baseline with the 384px hard-class sampler:

```bash
SKIP_PREPARE=1 MODELS="convnext_base" RUN_BALANCE_ABLATION=0 RUN_FULL_ANALYSIS=0 bash run_all_experiments.sh
```

If `data` or `outputs` already exists in the server code directory and you want to move it to the large disk before linking:

```bash
MOVE_EXISTING=1 bash scripts/setup_server_storage.sh
```
