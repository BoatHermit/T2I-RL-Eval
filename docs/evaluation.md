# Evaluation Workflow

This repository stores the benchmark manifests, generation scripts, and scoring scripts needed for Topic 3 evaluation.

## Paths

- TIFA manifest: `data/evaluation/tifa/samples.jsonl`
- GenAI-Bench manifest: `data/evaluation/genai_bench/samples.jsonl`
- Default LoRA directory: `artifacts/lora/grpo_siliconflow_quick_final`
- Generated images: `outputs/evaluation/images/<benchmark>/<variant>/<sample_id>.png`
- Result JSONL: `outputs/evaluation/results/<benchmark>/<variant>.jsonl`
- Reports: `outputs/evaluation/reports/`

## Commands

Generate baseline images:

```bash
python scripts/generate_benchmark_images.py \
  --benchmark tifa \
  --variant before \
  --manifest_path data/evaluation/tifa/samples.jsonl \
  --output_dir outputs/evaluation
```

Generate LoRA images:

```bash
python scripts/generate_benchmark_images.py \
  --benchmark tifa \
  --variant after \
  --manifest_path data/evaluation/tifa/samples.jsonl \
  --lora_path artifacts/lora/grpo_siliconflow_quick_final \
  --output_dir outputs/evaluation
```

Run TIFA judge:

```bash
python scripts/run_tifa.py \
  --manifest_path data/evaluation/tifa/samples.jsonl \
  --images_root outputs/evaluation/images \
  --variant before \
  --output_path outputs/evaluation/results/tifa_before.jsonl
```

Run GenAI-Bench judge:

```bash
python scripts/run_genai_bench.py \
  --manifest_path data/evaluation/genai_bench/samples.jsonl \
  --images_root outputs/evaluation/images \
  --variant after \
  --output_path outputs/evaluation/results/genai_after.jsonl
```

Write the summary:

```bash
python scripts/summarize_evaluation.py \
  --tifa_results_before outputs/evaluation/results/tifa_before.jsonl \
  --tifa_results_after outputs/evaluation/results/tifa_after.jsonl \
  --genai_results_before outputs/evaluation/results/genai_before.jsonl \
  --genai_results_after outputs/evaluation/results/genai_after.jsonl \
  --output_dir outputs/evaluation/reports
```

## Colab Order

1. Install dependencies.
2. Mount Drive if needed.
3. Generate `before` images.
4. Generate `after` images.
5. Run TIFA scoring for both variants.
6. Run GenAI-Bench scoring for both variants.
7. Produce summary files.

## Notes

- The evaluation scripts are resumable through `--resume`.
- The scoring scripts use API judges and keep per-sample error metadata.
- Only the inference-required LoRA files are stored in this repository.

