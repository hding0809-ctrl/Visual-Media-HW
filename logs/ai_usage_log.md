# Generative AI Usage Log

| Date | AI Tool | Purpose | Outputs Used | Human Review and Modification |
| --- | --- | --- | --- | --- |
| 2026-05-31 | Codex | Project planning, repository scaffold drafting, remote execution assistance, debugging, and result summarization | Scripts, analysis modules, README/report outline, experiment logs, metric summaries, figure generation workflow | Student should inspect paths, verify outputs, and write final conclusions using actual experiment results. |
| 2026-05-31 | Codex | Implementation adjustment during execution | Added PyTorch 2.2 launch compatibility, BasicSR-compatible SSIM, TTA logging, and independent `nafnet-vm` environment references | Results were checked against `logs/baseline_eval.log` and generated CSV files before being recorded. |
| 2026-06-06 | Codex image generation | Visual polishing for the final report | AI-generated method overview illustration: `report/figures/nafnet_workflow_ai_illustration.png` | The generated image is labeled as an illustration only; all quantitative results and failure panels still come from actual experiment outputs. |

Notes:

- AI may be used for planning, code drafts, debugging suggestions, and language polishing.
- Experimental design, result judgment, and final report claims should be manually reviewed.
- Do not directly copy AI-generated text into the final report without checking it.
- All reported numbers must come from actual experiment outputs.
