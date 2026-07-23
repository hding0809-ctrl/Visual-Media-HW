# NAFNet GoPro Deblurring Reproduction Package

本交付包用于复现 NAFNet 在 GoPro 去模糊任务上的 baseline 结果，并提供失效案例分析与 D4 test-time augmentation 推理增强实验。包内已包含代码、测试数据、预训练权重、实验输出、可视化结果和中英文报告。

## 1. 目录结构

```text
.
├── README.md
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

主要目录说明：

- `src/`：指标计算、失效案例挖掘、D4-TTA 推理和可视化代码。
- `scripts/`：完整复跑入口脚本。
- `third_party/NAFNet/`：官方 NAFNet 代码副本、GoPro 测试 LMDB 和预训练权重。
- `downloads/`：原始 GoPro 数据压缩包备份，可用于重建 LMDB。
- `results/`：实验 CSV、图表、失效案例面板和 TTA 输出。
- `logs/`：baseline、TTA、实验记录和 AI 使用记录。
- `report/`：中英文最终报告及报告用图。

## 2. 环境要求

推荐环境：

- Python 3.9
- PyTorch 1.11.0
- CUDA 11.3
- torchvision 0.12.0
- torchaudio 0.11.0

如果运行机器已有兼容的 PyTorch/CUDA 环境，也可以直接复用该环境并安装额外分析依赖。

## 3. 创建环境

在交付包根目录执行：

```bash
conda env create -f environment.yml
conda activate nafnet-vm
pip install -r requirements_extra.txt
```

然后安装 NAFNet 官方依赖：

```bash
cd third_party/NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
cd ../..
```

也可以运行保守版环境提示脚本：

```bash
bash scripts/setup_env.sh
```

如需由脚本创建 conda 环境：

```bash
bash scripts/setup_env.sh --create
```

## 4. 数据和权重检查

交付包已包含 GoPro 测试 LMDB 和官方 NAFNet GoPro checkpoint。请确认以下路径存在：

```text
third_party/NAFNet/datasets/GoPro/test/input.lmdb
third_party/NAFNet/datasets/GoPro/test/target.lmdb
third_party/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth
```

如需从原始 GoPro 数据压缩包重建测试 LMDB，可执行：

```bash
bash scripts/prepare_gopro_test_lmdb.sh
```

## 5. 完整复跑命令

所有命令均从交付包根目录执行。默认使用一张 GPU：

```bash
bash scripts/prepare_dirs.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_failure_analysis.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_tta_eval.sh
bash scripts/make_figures.sh
bash scripts/collect_report_assets.sh
```

如果机器没有 GPU，baseline 和 TTA 推理可能无法按预期运行。建议使用单张支持 CUDA 的 GPU 完成复跑。

## 6. 结果位置

关键日志：

```text
logs/baseline_eval.log
logs/tta_eval.log
logs/experiment_log.md
logs/ai_usage_log.md
```

关键 CSV：

```text
results/csv/baseline_per_image_metrics.csv
results/csv/failure_group_summary.csv
results/csv/tta_per_image_metrics.csv
results/csv/before_after_summary.csv
```

关键图表和可视化：

```text
results/figures/
results/failure_cases/
report/figures/
```

最终报告：

```text
report/nafnet_gopro_report_zh.pdf
report/nafnet_gopro_report_zh.docx
report/nafnet_gopro_report_zh.md
report/nafnet_gopro_report_en.pdf
report/nafnet_gopro_report_en.docx
report/nafnet_gopro_report_en.md
```

## 7. 期望复现结果

官方 `NAFNet-GoPro-width64` 的 GoPro 测试集参考结果约为：

- PSNR: 33.7103
- SSIM: 0.9668

本项目 baseline 复跑结果应接近该数值。若 PSNR 偏差超过约 0.1-0.2 dB，请优先检查数据路径、checkpoint、NAFNet 测试配置和图像指标约定。

## 8. 常见问题

CUDA 不可用：

- 确认驱动、CUDA runtime 和 PyTorch CUDA 版本兼容。
- 确认运行命令前已激活正确 conda 环境。

数据路径缺失：

- 检查 `third_party/NAFNet/datasets/GoPro/test/input.lmdb` 和 `target.lmdb`。
- 如需重建，执行 `bash scripts/prepare_gopro_test_lmdb.sh`。

checkpoint 缺失：

- 检查 `third_party/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth`。

显存不足：

- 确认只运行一个任务。
- 默认命令已使用单卡；可调整 `CUDA_VISIBLE_DEVICES` 指向目标 GPU。

LMDB 重建失败：

- 确认 `downloads/GOPRO_Large.zip` 存在且未损坏。
- 确认磁盘有足够空间保存解压后的临时数据。

## 9. 压缩包校验

交付包随附 SHA256 校验文件。下载后可在压缩包所在目录执行：

```bash
shasum -a 256 -c nafnet-gopro-complete-deliverable-20260611.tar.gz.sha256
```

Linux 环境也可以使用：

```bash
sha256sum -c nafnet-gopro-complete-deliverable-20260611.tar.gz.sha256
```

校验通过后再解压：

```bash
tar -xzf nafnet-gopro-complete-deliverable-20260611.tar.gz
cd nafnet-gopro-complete-deliverable-20260611
```

