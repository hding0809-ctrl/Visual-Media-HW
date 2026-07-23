# Checkpoints

Do not commit model checkpoints to this repository.

The expected official checkpoint path is:

```text
third_party/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth
```

You may override it with:

```bash
NAFNET_CHECKPOINT=/path/to/NAFNet-GoPro-width64.pth bash scripts/run_tta_eval.sh
```

