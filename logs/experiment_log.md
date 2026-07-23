# Experiment Log

| Date | Experiment | Command | Input | Output | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-31 | Project runtime | `python --version`; `python -c "import torch; print(torch.__version__)"` | Isolated project environment | Compatible Python/PyTorch runtime | Python 3.10.20, PyTorch 2.2.0+cu121, CUDA runtime available | The delivered README also provides the official NAFNet reference stack for clean reproduction. |
| 2026-05-31 | Data and checkpoint preparation | `bash scripts/prepare_gopro_test_lmdb.sh` | `downloads/GOPRO_Large.zip` + official NAFNet checkpoint | GoPro test LMDB + `NAFNet-GoPro-width64.pth` | 1111 GoPro test pairs prepared | Checkpoint SHA256: `329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f`. |
| 2026-05-31 | Baseline reproduction | `CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh` | GoPro test LMDB + official checkpoint | `logs/baseline_eval.log` | PSNR 33.7103 / SSIM 0.9668 | Matches the official `NAFNet-GoPro-width64` reference. |
| 2026-05-31 | Failure analysis | `CUDA_VISIBLE_DEVICES=0 bash scripts/run_failure_analysis.sh` | baseline outputs + GoPro GT | `results/csv/`, `results/failure_cases/` | 1111 per-image rows; mean PSNR 33.7103 / SSIM 0.9668 | Exported heavy blur, high saturation/dark, and high texture panels. |
| 2026-05-31 | D4-TTA evaluation | `CUDA_VISIBLE_DEVICES=0 bash scripts/run_tta_eval.sh` | GoPro test input + checkpoint | `results/tta/`, `results/csv/before_after_summary.csv` | PSNR 33.9301 / SSIM 0.9680; delta +0.2198 dB / +0.001221 SSIM | 1111 TTA images, mean D4-TTA model time 1.2105 s/image. |
| 2026-05-31 | Report figures | `bash scripts/make_figures.sh`; `bash scripts/collect_report_assets.sh` | CSV outputs and failure panels | `results/figures/`, `report/figures/` | 6 summary figures; 26 report figure assets | Ready for customer review and final report reading. |
