# Colab Evaluation Workflow Design

**Date:** 2026-04-09

**Goal:** Build a Colab-first evaluation workflow for Topic 3 that compares baseline `Janus-Pro-1B` against `Janus-Pro-1B + LoRA` using TIFA and GenAI-Bench, with reproducible scripts and report-ready outputs.

## Scope

This design covers:

- Repository-local benchmark manifests for TIFA and GenAI-Bench
- Colab-based image generation for `before` and `after` variants
- API-judge evaluation runners for TIFA and GenAI-Bench
- Summary export for benchmark scores, qualitative comparisons, and error taxonomy
- Repository-local storage of inference-only LoRA artifacts

This design does not cover:

- RL training changes
- Additional benchmark integration beyond the workflow hooks needed for future extension
- Local offline judge execution

## Constraints

- The generation backend is fixed to `Janus-Pro-1B`
- The `after` variant uses an optional LoRA adapter or checkpoint-compatible adapter path
- Colab is used for model inference and judge execution
- Benchmark prompts and metadata must be stored inside the repository
- API judges are allowed for scoring
- The current repository only stores inference-required LoRA files, not training state

## High-Level Architecture

The workflow is split into four layers:

1. **Benchmark manifests**
   Static benchmark inputs stored in the repository. These define prompt ids, prompt text, categories, and benchmark-specific metadata.

2. **Generation layer**
   A generation script produces images for `before` and `after` variants using the same manifest rows and consistent output contracts.

3. **Benchmark runners**
   Independent scripts score generated images for TIFA and GenAI-Bench using API judges and write normalized result records.

4. **Summary/reporting layer**
   A summarization script compares `before` and `after`, aggregates benchmark metrics, counts error types, and emits report-ready artifacts.

This split is intentional for Colab reliability. Image generation is the most expensive stage, so it must be resumable and reusable across multiple scoring passes.

## Repository Layout

The evaluation repository should use the following layout:

```text
D:/Code/T2I-RL-Eval/
├── artifacts/
│   └── lora/
│       └── grpo_siliconflow_quick_final/
│           ├── adapter_config.json
│           └── adapter_model.safetensors
├── data/
│   └── evaluation/
│       ├── tifa/
│       │   └── samples.jsonl
│       └── genai_bench/
│           └── samples.jsonl
├── docs/
│   └── evaluation.md
├── notebooks/
│   └── colab_evaluation.ipynb
├── outputs/
│   └── evaluation/
│       ├── images/
│       ├── results/
│       └── reports/
├── scripts/
│   ├── generate_benchmark_images.py
│   ├── run_tifa.py
│   ├── run_genai_bench.py
│   └── summarize_evaluation.py
└── src/
    └── evaluation/
        ├── __init__.py
        ├── benchmarks.py
        ├── io.py
        ├── reporting.py
        ├── schemas.py
        ├── tifa_runner.py
        └── genai_bench_runner.py
```

## Benchmark Data Contract

The repository stores benchmark inputs as JSONL to support streaming, filtering, and resume-friendly processing.

### TIFA manifest

Each record contains:

- `sample_id`
- `prompt`
- `category`
- `source`
- `questions`

Each `questions` item contains:

- `question`
- `expected_answer`
- `question_type`

Example:

```json
{
  "sample_id": "tifa-0001",
  "prompt": "a red apple on a wooden table",
  "category": "attribute_binding",
  "source": "tifa",
  "questions": [
    {
      "question": "What color is the apple?",
      "expected_answer": "red",
      "question_type": "attribute"
    },
    {
      "question": "What is the apple on?",
      "expected_answer": "table",
      "question_type": "relation"
    }
  ]
}
```

### GenAI-Bench manifest

Each record contains:

- `sample_id`
- `prompt`
- `category`
- `skills`
- `source`

Example:

```json
{
  "sample_id": "genai-0001",
  "prompt": "two cats on a blue sofa",
  "category": "reasoning",
  "skills": ["counting", "attribute_binding"],
  "source": "genai-bench"
}
```

## Output Contract

### Generated image records

Each generated sample writes both an image file and a normalized metadata record.

Image path layout:

- `outputs/evaluation/images/tifa/before/<sample_id>.png`
- `outputs/evaluation/images/tifa/after/<sample_id>.png`
- `outputs/evaluation/images/genai_bench/before/<sample_id>.png`
- `outputs/evaluation/images/genai_bench/after/<sample_id>.png`

Generated metadata fields:

- `benchmark`
- `sample_id`
- `prompt`
- `variant`
- `seed`
- `model_name`
- `checkpoint_or_lora`
- `image_path`
- `generation_config`

### Scored result records

TIFA result fields:

- `benchmark`
- `sample_id`
- `prompt`
- `variant`
- `score`
- `question_accuracy`
- `question_results`
- `error_types`
- `judge_metadata`
- `image_path`

GenAI-Bench result fields:

- `benchmark`
- `sample_id`
- `prompt`
- `variant`
- `score`
- `subscores`
- `error_types`
- `judge_metadata`
- `image_path`

### Summary/report artifacts

The summary layer writes:

- `outputs/evaluation/reports/summary.json`
- `outputs/evaluation/reports/summary.csv`
- `outputs/evaluation/reports/summary.md`

The summary must include:

- Overall `before` vs `after` benchmark scores
- Per-category deltas where available
- Error taxonomy counts
- Top improved samples
- Top failed samples
- A qualitative sample shortlist for report figures

## Script Responsibilities

### `scripts/generate_benchmark_images.py`

Responsibilities:

- Load TIFA and GenAI-Bench manifest rows
- Generate `before` images with base `Janus-Pro-1B`
- Generate `after` images with `Janus-Pro-1B + LoRA`
- Save images and generation metadata
- Skip completed records when `--resume` is enabled

Key CLI parameters:

- `--benchmark`
- `--manifest_path`
- `--variant`
- `--base_model`
- `--lora_path`
- `--output_dir`
- `--limit`
- `--seed`
- `--resume`

### `scripts/run_tifa.py`

Responsibilities:

- Read TIFA manifest and generated images
- Call the configured API judge for question answering or answer checking
- Score each TIFA sample
- Map incorrect question types to normalized error taxonomy labels
- Save result records to JSONL

Key CLI parameters:

- `--manifest_path`
- `--images_root`
- `--variant`
- `--judge_model`
- `--api_provider`
- `--output_path`
- `--resume`

### `scripts/run_genai_bench.py`

Responsibilities:

- Read GenAI-Bench manifest and generated images
- Call the configured API judge using a rubric-style prompt
- Score each sample with normalized subscores
- Save result records to JSONL

Key CLI parameters:

- `--manifest_path`
- `--images_root`
- `--variant`
- `--judge_model`
- `--api_provider`
- `--output_path`
- `--resume`

Default rubric dimensions:

- `alignment`
- `instruction_fidelity`
- `compositionality`
- `visual_quality`

### `scripts/summarize_evaluation.py`

Responsibilities:

- Load `before` and `after` result files for both benchmarks
- Compute aggregate metrics and deltas
- Count error taxonomy labels
- Produce JSON, CSV, and Markdown reports

Key CLI parameters:

- `--tifa_results_before`
- `--tifa_results_after`
- `--genai_results_before`
- `--genai_results_after`
- `--output_dir`

## Colab Notebook Flow

The notebook `notebooks/colab_evaluation.ipynb` should have six sections:

1. Environment setup
   Install dependencies, clone or sync the repository, and mount Drive if needed.

2. Configuration
   Set API keys, benchmark limits, model path, and LoRA path.

3. Before/after image generation
   Run the generation script for both variants.

4. TIFA scoring
   Run the TIFA scoring script for both variants.

5. GenAI-Bench scoring
   Run the GenAI-Bench scoring script for both variants.

6. Summary export
   Run the summarization script and package report-ready outputs.

## LoRA Asset Handling

The inference-only LoRA assets are copied from:

- `D:/Code/#OpenSoursce/T2I-RL-Project/outputs/grpo_siliconflow_quick/final_checkpoint`

The evaluation repository stores only:

- `adapter_config.json`
- `adapter_model.safetensors`

The repository does not store:

- `training_state.pt`

The default LoRA target directory is:

- `artifacts/lora/grpo_siliconflow_quick_final`

This keeps evaluation inputs independent from training-state artifacts and reduces sync cost for Colab.

## Error Handling And Resume Strategy

### Resume behavior

- Generation skips samples whose target image already exists when `--resume` is set
- Benchmark runners skip samples already present in the result JSONL when `--resume` is set
- Summary always recomputes from existing result files

### Failure handling

- Missing manifest files, missing LoRA assets, or missing images cause immediate fail-fast errors with explicit paths
- API judge failures are recorded per sample with error status and preserved in output files
- Failed judge samples are excluded from score aggregation and counted separately

## Testing Strategy

The implementation should test contracts, not heavy model execution.

### Unit tests

- Manifest parsing and validation
- JSON and JSONL read/write helpers
- Generated record serialization
- Result record serialization
- Error taxonomy mapping
- `before` vs `after` summary aggregation
- Resume-skip logic

### Integration tests

- End-to-end smoke test with a mock generator
- End-to-end smoke test with a mock judge
- Summary generation from small fixture result files

The tests should not require real GPU inference or live API calls.

## Risks And Mitigations

### API instability

Risk:
Different judge outputs may vary in structure or wording.

Mitigation:
Use structured prompts with explicit JSON contracts and robust parsing fallback.

### Colab interruption

Risk:
Long-running generation or scoring jobs may be interrupted.

Mitigation:
Keep generation, scoring, and reporting as separate resumable stages.

### Benchmark drift

Risk:
Benchmark manifests may diverge from external upstream benchmark data.

Mitigation:
Document the repository-stored benchmark subset and preserve source provenance in each manifest row.

## Success Criteria

The workflow is considered complete when it can:

- Generate `before` and `after` images from repository-local benchmark manifests
- Score both variants on TIFA and GenAI-Bench using API judges
- Export benchmark comparisons and error taxonomy summaries
- Run end-to-end from Colab with a documented sequence
- Reuse inference-only LoRA assets stored in the repository
