# Evaluation Workflow

This repository stores the benchmark manifests, generation scripts, and scoring scripts needed for Topic 3 evaluation.

The repository now includes converted manifests built from official benchmark metadata:

- TIFA v1.0: 4,073 prompts with 25,829 question-answer items
- GenAI-Bench-1600: 1,600 prompts with official skill tags

## Paths

- TIFA manifest: `data/evaluation/tifa/samples.jsonl`
- TIFA raw source files: `data/evaluation/tifa/raw/tifa_v1.0_text_inputs.json`, `data/evaluation/tifa/raw/tifa_v1.0_question_answers.json`
- GenAI-Bench manifest: `data/evaluation/genai_bench/samples.jsonl`
- GenAI-Bench raw source files: `data/evaluation/genai_bench/raw/genai_image.json`, `data/evaluation/genai_bench/raw/genai_skills.json`
- Default LoRA directory: `artifacts/lora/grpo_siliconflow_quick_final`
- Generated images: `outputs/evaluation/images/<benchmark>/<variant>/<sample_id>.png`
- Result JSONL: `outputs/evaluation/results/<benchmark>/<variant>.jsonl`
- Reports: `outputs/evaluation/reports/`

## Dataset Notes

- `samples.jsonl` is the runtime manifest that the scripts consume directly.
- `raw/` keeps upstream official metadata for provenance and future re-conversion.
- Full before/after evaluation over all official prompts is expensive on Colab. Use `--limit` for smoke runs, then scale up selectively.

## Commands

Generate baseline images:

```bash
python scripts/generate_benchmark_images.py \
  --benchmark tifa \
  --variant before \
  --manifest_path data/evaluation/tifa/samples.jsonl \
  --limit 20 \
  --output_dir outputs/evaluation
```

Generate LoRA images:

```bash
python scripts/generate_benchmark_images.py \
  --benchmark tifa \
  --variant after \
  --manifest_path data/evaluation/tifa/samples.jsonl \
  --lora_path artifacts/lora/grpo_siliconflow_quick_final \
  --limit 20 \
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
