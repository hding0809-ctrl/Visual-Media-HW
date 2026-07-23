# NAFNet GoPro Deblurring Reproduction Package

This package reproduces the NAFNet baseline on the GoPro deblurring task, and provides a failure-case
analysis and a D4 test-time augmentation (TTA) inference experiment. It contains the code, test data,
pretrained checkpoint, experiment outputs, visualizations, and bilingual (English/Chinese) reports.

- **Course:** Visual Media
- **Paper:** *Simple Baselines for Image Restoration* / NAFNet, ECCV 2022
- **Task:** GoPro single-image deblurring
- **Source code:** https://github.com/hding0809-ctrl/Visual-Media-HW

## 1. Directory Structure

```text
.
├── README.md              # Chinese readme
├── README_EN.md           # This English readme
├── DELIVERY_MANIFEST.txt
├── environment.yml
├── requirements_extra.txt
├── configs/
├── scripts/
├── src/
├── downloads/
├── data/
├── results/
├── logs/
├── report/
└── third_party/
```

Main directories:

- `src/`: metric computation, failure-case mining, D4-TTA inference, and visualization code.
- `scripts/`: entry-point scripts for a full rerun.
- `third_party/NAFNet/`: a copy of the official NAFNet code, the GoPro test LMDB, and the pretrained weights.
- `downloads/`: a backup of the raw GoPro dataset archive, used to rebuild the LMDB.
- `results/`: experiment CSVs, charts, failure-case panels, and TTA outputs.
- `logs/`: baseline, TTA, experiment, and AI-usage records.
- `report/`: the bilingual final reports and the figures used in them.

## 2. Environment Requirements

Recommended environment:

- Python 3.9
- PyTorch 1.11.0
- CUDA 11.3
- torchvision 0.12.0
- torchaudio 0.11.0

If the machine already has a compatible PyTorch/CUDA setup, you may reuse it and simply install the
extra analysis dependencies.

## 3. Creating the Environment

From the package root:

```bash
conda env create -f environment.yml
conda activate nafnet-vm
pip install -r requirements_extra.txt
```

Then install the official NAFNet dependencies:

```bash
cd third_party/NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
cd ../..
```

You can also run the conservative environment-hint script:

```bash
bash scripts/setup_env.sh
```

To have the script create the conda environment for you:

```bash
bash scripts/setup_env.sh --create
```

## 4. Data and Checkpoint Check

The package already includes the GoPro test LMDB and the official NAFNet GoPro checkpoint. Confirm the
following paths exist:

```text
third_party/NAFNet/datasets/GoPro/test/input.lmdb
third_party/NAFNet/datasets/GoPro/test/target.lmdb
third_party/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth
```

To rebuild the test LMDB from the raw GoPro archive:

```bash
bash scripts/prepare_gopro_test_lmdb.sh
```

## 5. Full Rerun Commands

All commands are run from the package root. A single GPU is used by default:

```bash
bash scripts/prepare_dirs.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_failure_analysis.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_tta_eval.sh
bash scripts/make_figures.sh
bash scripts/collect_report_assets.sh
```

Without a GPU, the baseline and TTA inference may not run as expected. A single CUDA-capable GPU is
recommended to complete the rerun.

## 6. Result Locations

Key logs:

```text
logs/baseline_eval.log
logs/tta_eval.log
logs/experiment_log.md
logs/ai_usage_log.md
```

Key CSVs:

```text
results/csv/baseline_per_image_metrics.csv
results/csv/failure_group_summary.csv
results/csv/tta_per_image_metrics.csv
results/csv/before_after_summary.csv
```

Key charts and visualizations:

```text
results/figures/
results/failure_cases/
report/figures/
```

Final reports:

```text
report/nafnet_gopro_report_en.pdf
report/nafnet_gopro_report_en.docx
report/nafnet_gopro_report_en.md
report/nafnet_gopro_report_zh.pdf
report/nafnet_gopro_report_zh.docx
report/nafnet_gopro_report_zh.md
```

## 7. Expected Reproduction Results

The official `NAFNet-GoPro-width64` reference results on the GoPro test set are approximately:

- PSNR: 33.7103
- SSIM: 0.9668

This project's baseline rerun should match these values closely. If the PSNR deviates by more than
about 0.1–0.2 dB, first check the data paths, checkpoint, NAFNet test configuration, and the
image-metric conventions.

The D4 test-time augmentation experiment further raises the overall result to about PSNR 33.9301
(+0.22 dB) and SSIM 0.9680, at roughly 1.21 s per image (about 8× the forward passes).

## 8. FAQ

CUDA is unavailable:

- Confirm the driver, CUDA runtime, and PyTorch CUDA build are compatible.
- Make sure the correct conda environment is activated before running the commands.

Missing data paths:

- Check `third_party/NAFNet/datasets/GoPro/test/input.lmdb` and `target.lmdb`.
- To rebuild, run `bash scripts/prepare_gopro_test_lmdb.sh`.

Missing checkpoint:

- Check `third_party/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth`.

Out of GPU memory:

- Make sure only one task is running.
- The default commands already use a single GPU; adjust `CUDA_VISIBLE_DEVICES` to target a specific GPU.

LMDB rebuild fails:

- Confirm `downloads/GOPRO_Large.zip` exists and is not corrupted.
- Confirm there is enough disk space for the extracted temporary data.

## 9. Archive Checksum

The package ships with a SHA256 checksum file. After downloading, run the following in the directory
that holds the archive:

```bash
shasum -a 256 -c nafnet-gopro-complete-deliverable-20260611.tar.gz.sha256
```

On Linux you can also use:

```bash
sha256sum -c nafnet-gopro-complete-deliverable-20260611.tar.gz.sha256
```

After the checksum passes, extract:

```bash
tar -xzf nafnet-gopro-complete-deliverable-20260611.tar.gz
cd nafnet-gopro-complete-deliverable-20260611
```
