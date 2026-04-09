# Colab Evaluation Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-first evaluation workflow that compares baseline `Janus-Pro-1B` against `Janus-Pro-1B + LoRA` on TIFA and GenAI-Bench using repository-local benchmark manifests, resumable scripts, and report-ready outputs.

**Architecture:** Keep benchmark data and evaluation contracts local to this repository. Implement a small `src/evaluation` package for manifests, records, runners, and reporting; expose reproducible entry points in `scripts/`; and add a Colab notebook that runs generation, scoring, and summary export in separate resumable stages.

**Tech Stack:** Python 3.10+, pytest, PIL, PyTorch, JSON/JSONL, argparse, Google Colab, OpenAI-compatible API judge, Janus-Pro-1B LoRA inference assets

---

## File Structure

- Create: `D:/Code/T2I-RL-Eval/src/evaluation/__init__.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/schemas.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/io.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/benchmarks.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/tifa_runner.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/genai_bench_runner.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/reporting.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/generate_benchmark_images.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/run_tifa.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/run_genai_bench.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/summarize_evaluation.py`
- Create: `D:/Code/T2I-RL-Eval/data/evaluation/tifa/samples.jsonl`
- Create: `D:/Code/T2I-RL-Eval/data/evaluation/genai_bench/samples.jsonl`
- Create: `D:/Code/T2I-RL-Eval/tests/test_evaluation_contracts.py`
- Create: `D:/Code/T2I-RL-Eval/tests/test_evaluation_smoke.py`
- Create: `D:/Code/T2I-RL-Eval/notebooks/colab_evaluation.ipynb`
- Create: `D:/Code/T2I-RL-Eval/docs/evaluation.md`
- Create: `D:/Code/T2I-RL-Eval/requirements.txt`
- Copy: `D:/Code/T2I-RL-Eval/artifacts/lora/grpo_siliconflow_quick_final/adapter_config.json`
- Copy: `D:/Code/T2I-RL-Eval/artifacts/lora/grpo_siliconflow_quick_final/adapter_model.safetensors`

### Task 1: Establish Package Skeleton And Data Contracts

**Files:**
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/__init__.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/schemas.py`
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/io.py`
- Create: `D:/Code/T2I-RL-Eval/tests/test_evaluation_contracts.py`

- [ ] **Step 1: Write the failing tests for record serialization and JSONL helpers**

```python
from pathlib import Path


def test_generated_sample_record_round_trip(tmp_path: Path):
    from src.evaluation.schemas import GeneratedSampleRecord
    from src.evaluation.io import read_jsonl, write_jsonl

    path = tmp_path / "generated.jsonl"
    record = GeneratedSampleRecord(
        benchmark="tifa",
        sample_id="tifa-0001",
        prompt="a red apple on a wooden table",
        variant="before",
        seed=42,
        model_name="deepseek-ai/Janus-Pro-1B",
        checkpoint_or_lora="base",
        image_path="outputs/evaluation/images/tifa/before/tifa-0001.png",
        generation_config={"guidance_scale": 5.0},
    )

    write_jsonl(path, [record.to_dict()])
    assert read_jsonl(path) == [record.to_dict()]


def test_scored_sample_record_preserves_subscores():
    from src.evaluation.schemas import ScoredSampleRecord

    record = ScoredSampleRecord(
        benchmark="genai_bench",
        sample_id="genai-0001",
        variant="after",
        prompt="two cats on a blue sofa",
        score=0.81,
        subscores={"alignment": 0.9, "compositionality": 0.7},
        error_types=["wrong_count"],
        judge_metadata={"judge_model": "gpt-4.1-mini"},
        image_path="outputs/evaluation/images/genai_bench/after/genai-0001.png",
    )

    payload = record.to_dict()
    assert payload["subscores"]["alignment"] == 0.9
    assert payload["error_types"] == ["wrong_count"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_contracts.py -k "round_trip or preserves_subscores" -v`

Expected: `ModuleNotFoundError` because `src/evaluation` does not exist yet.

- [ ] **Step 3: Implement normalized records and JSON/JSONL helpers**

```python
# src/evaluation/schemas.py
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class GeneratedSampleRecord:
    benchmark: str
    sample_id: str
    prompt: str
    variant: str
    seed: int
    model_name: str
    checkpoint_or_lora: str
    image_path: str
    generation_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredSampleRecord:
    benchmark: str
    sample_id: str
    variant: str
    prompt: str
    score: float
    subscores: Dict[str, Any] = field(default_factory=dict)
    error_types: List[str] = field(default_factory=list)
    judge_metadata: Dict[str, Any] = field(default_factory=dict)
    image_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

```python
# src/evaluation/io.py
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
```

- [ ] **Step 4: Re-export public helpers and rerun the tests**

```python
# src/evaluation/__init__.py
from src.evaluation.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from src.evaluation.schemas import GeneratedSampleRecord, ScoredSampleRecord
```

Run: `pytest tests/test_evaluation_contracts.py -k "round_trip or preserves_subscores" -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/io.py src/evaluation/schemas.py tests/test_evaluation_contracts.py
git commit -m "feat: add evaluation data contracts"
```

### Task 2: Add Manifest Loaders And Built-In Benchmark Samples

**Files:**
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/benchmarks.py`
- Create: `D:/Code/T2I-RL-Eval/data/evaluation/tifa/samples.jsonl`
- Create: `D:/Code/T2I-RL-Eval/data/evaluation/genai_bench/samples.jsonl`
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_contracts.py`

- [ ] **Step 1: Write the failing tests for manifest iteration and required metadata**

```python
def test_tifa_manifest_iterates_question_rows(tmp_path):
    from src.evaluation.benchmarks import TIFABenchmark
    from src.evaluation.io import write_jsonl

    write_jsonl(
        tmp_path / "samples.jsonl",
        [
            {
                "sample_id": "tifa-0001",
                "prompt": "a red apple on a table",
                "category": "attribute_binding",
                "source": "tifa",
                "questions": [
                    {
                        "question": "What color is the apple?",
                        "expected_answer": "red",
                        "question_type": "attribute",
                    }
                ],
            }
        ],
    )

    rows = TIFABenchmark(tmp_path / "samples.jsonl").iter_samples()
    assert rows[0]["sample_id"] == "tifa-0001"
    assert rows[0]["questions"][0]["question_type"] == "attribute"


def test_genai_manifest_requires_category_and_skills(tmp_path):
    from src.evaluation.benchmarks import GenAIBenchmark
    from src.evaluation.io import write_jsonl

    write_jsonl(
        tmp_path / "samples.jsonl",
        [
            {
                "sample_id": "genai-0001",
                "prompt": "two dogs under one umbrella",
                "category": "reasoning",
                "skills": ["counting", "spatial"],
                "source": "genai-bench",
            }
        ],
    )

    rows = GenAIBenchmark(tmp_path / "samples.jsonl").iter_samples()
    assert rows[0]["category"] == "reasoning"
    assert rows[0]["skills"] == ["counting", "spatial"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_contracts.py -k "manifest_iterates or requires_category_and_skills" -v`

Expected: `ImportError` because `benchmarks.py` does not exist yet.

- [ ] **Step 3: Implement benchmark manifest readers with validation**

```python
# src/evaluation/benchmarks.py
from pathlib import Path
from typing import Dict, List, Optional

from src.evaluation.io import read_jsonl


class BaseBenchmark:
    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)

    def iter_samples(self, limit: Optional[int] = None) -> List[Dict]:
        rows = read_jsonl(self.manifest_path)
        return rows[:limit] if limit is not None else rows


class TIFABenchmark(BaseBenchmark):
    required_fields = {"sample_id", "prompt", "category", "source", "questions"}

    def iter_samples(self, limit: Optional[int] = None) -> List[Dict]:
        rows = super().iter_samples(limit=limit)
        for row in rows:
            missing = self.required_fields - row.keys()
            if missing:
                raise ValueError(f"Missing TIFA fields: {sorted(missing)}")
        return rows


class GenAIBenchmark(BaseBenchmark):
    required_fields = {"sample_id", "prompt", "category", "skills", "source"}

    def iter_samples(self, limit: Optional[int] = None) -> List[Dict]:
        rows = super().iter_samples(limit=limit)
        for row in rows:
            missing = self.required_fields - row.keys()
            if missing:
                raise ValueError(f"Missing GenAI-Bench fields: {sorted(missing)}")
        return rows
```

- [ ] **Step 4: Seed repository manifests and rerun the tests**

Example rows for `data/evaluation/tifa/samples.jsonl`:

```json
{"sample_id":"tifa-0001","prompt":"a red apple on a wooden table","category":"attribute_binding","source":"tifa","questions":[{"question":"What color is the apple?","expected_answer":"red","question_type":"attribute"},{"question":"What is the apple on?","expected_answer":"table","question_type":"relation"}]}
{"sample_id":"tifa-0002","prompt":"three dogs running in a green field","category":"counting","source":"tifa","questions":[{"question":"How many dogs are there?","expected_answer":"three","question_type":"count"},{"question":"What color is the field?","expected_answer":"green","question_type":"attribute"}]}
```

Example rows for `data/evaluation/genai_bench/samples.jsonl`:

```json
{"sample_id":"genai-0001","prompt":"two cats on a blue sofa","category":"reasoning","skills":["counting","attribute_binding"],"source":"genai-bench"}
{"sample_id":"genai-0002","prompt":"a red bicycle to the left of a yellow car","category":"composition","skills":["attribute_binding","spatial"],"source":"genai-bench"}
```

Run: `pytest tests/test_evaluation_contracts.py -k "manifest_iterates or requires_category_and_skills" -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmarks.py data/evaluation/tifa/samples.jsonl data/evaluation/genai_bench/samples.jsonl tests/test_evaluation_contracts.py
git commit -m "feat: add benchmark manifests"
```

### Task 3: Build The Generation Script With Before/After Variants And Resume

**Files:**
- Create: `D:/Code/T2I-RL-Eval/scripts/generate_benchmark_images.py`
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_smoke.py`

- [ ] **Step 1: Write the failing tests for output paths and resume behavior**

```python
from pathlib import Path


def test_generation_path_uses_variant_and_sample_id(tmp_path: Path):
    from scripts.generate_benchmark_images import build_image_path

    path = build_image_path(tmp_path, "tifa", "after", "tifa-0001")
    assert path == tmp_path / "images" / "tifa" / "after" / "tifa-0001.png"


def test_should_skip_existing_image_when_resume(tmp_path: Path):
    from scripts.generate_benchmark_images import should_skip_sample

    target = tmp_path / "images" / "tifa" / "before" / "tifa-0001.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"png")

    assert should_skip_sample(target, resume=True) is True
    assert should_skip_sample(target, resume=False) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_smoke.py -k "generation_path_uses_variant or should_skip_existing_image" -v`

Expected: `ModuleNotFoundError` because the generation script does not exist yet.

- [ ] **Step 3: Implement a resumable generation entry point**

```python
# scripts/generate_benchmark_images.py
from pathlib import Path


def build_image_path(output_dir: Path, benchmark: str, variant: str, sample_id: str) -> Path:
    return Path(output_dir) / "images" / benchmark / variant / f"{sample_id}.png"


def should_skip_sample(target_path: Path, resume: bool) -> bool:
    return resume and target_path.exists()
```

Main flow requirements:

- Parse `--benchmark`, `--variant`, `--manifest_path`, `--base_model`, `--lora_path`, `--output_dir`, `--seed`, `--limit`, `--resume`
- Load `Janus-Pro-1B`
- For `before`, do not enable LoRA
- For `after`, enable LoRA from `artifacts/lora/grpo_siliconflow_quick_final` unless overridden
- Save image files and append normalized generation records to `outputs/evaluation/results/generated_samples.jsonl`

- [ ] **Step 4: Rerun the tests**

Run: `pytest tests/test_evaluation_smoke.py -k "generation_path_uses_variant or should_skip_existing_image" -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_benchmark_images.py tests/test_evaluation_smoke.py
git commit -m "feat: add resumable benchmark generation"
```

### Task 4: Implement TIFA Runner And Error Taxonomy Mapping

**Files:**
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/tifa_runner.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/run_tifa.py`
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_contracts.py`

- [ ] **Step 1: Write the failing tests for TIFA error mapping and result parsing**

```python
def test_tifa_question_type_maps_to_error_label():
    from src.evaluation.tifa_runner import map_question_type_to_error

    assert map_question_type_to_error("count") == "wrong_count"
    assert map_question_type_to_error("attribute") == "wrong_attribute"
    assert map_question_type_to_error("relation") == "wrong_relation"


def test_tifa_accuracy_is_fraction_of_correct_answers():
    from src.evaluation.tifa_runner import compute_question_accuracy

    question_results = [
        {"correct": True},
        {"correct": False},
        {"correct": True},
    ]

    assert compute_question_accuracy(question_results) == 2 / 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_contracts.py -k "maps_to_error_label or accuracy_is_fraction" -v`

Expected: `ImportError` because `tifa_runner.py` does not exist yet.

- [ ] **Step 3: Implement TIFA runner utilities and CLI wrapper**

```python
# src/evaluation/tifa_runner.py
from typing import Dict, List


ERROR_BY_QUESTION_TYPE = {
    "object": "missing_object",
    "attribute": "wrong_attribute",
    "count": "wrong_count",
    "relation": "wrong_relation",
}


def map_question_type_to_error(question_type: str) -> str:
    return ERROR_BY_QUESTION_TYPE.get(question_type, "other")


def compute_question_accuracy(question_results: List[Dict]) -> float:
    if not question_results:
        return 0.0
    correct = sum(1 for row in question_results if row.get("correct"))
    return correct / len(question_results)
```

Runner requirements:

- Load manifest rows and generated image paths
- Build an explicit JSON-output judge prompt per question
- Parse the judge answer and compare against `expected_answer`
- Aggregate `question_results`, `question_accuracy`, and `error_types`
- Append one JSONL row per sample

Script wrapper requirements:

- Read `--manifest_path`, `--images_root`, `--variant`, `--judge_model`, `--api_provider`, `--output_path`, `--resume`
- Fail fast on missing images

- [ ] **Step 4: Rerun the tests**

Run: `pytest tests/test_evaluation_contracts.py -k "maps_to_error_label or accuracy_is_fraction" -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/tifa_runner.py scripts/run_tifa.py tests/test_evaluation_contracts.py
git commit -m "feat: add tifa scoring runner"
```

### Task 5: Implement GenAI-Bench Runner With Rubric Subscores

**Files:**
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/genai_bench_runner.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/run_genai_bench.py`
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_contracts.py`

- [ ] **Step 1: Write the failing tests for rubric defaults and score aggregation**

```python
def test_genai_default_rubric_dimensions():
    from src.evaluation.genai_bench_runner import DEFAULT_RUBRIC_DIMENSIONS

    assert DEFAULT_RUBRIC_DIMENSIONS == [
        "alignment",
        "instruction_fidelity",
        "compositionality",
        "visual_quality",
    ]


def test_genai_overall_score_is_mean_of_subscores():
    from src.evaluation.genai_bench_runner import compute_overall_score

    score = compute_overall_score(
        {
            "alignment": 0.9,
            "instruction_fidelity": 0.7,
            "compositionality": 0.8,
            "visual_quality": 0.6,
        }
    )

    assert score == 0.75
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_contracts.py -k "default_rubric_dimensions or overall_score_is_mean" -v`

Expected: `ImportError` because `genai_bench_runner.py` does not exist yet.

- [ ] **Step 3: Implement rubric helpers and CLI wrapper**

```python
# src/evaluation/genai_bench_runner.py
from typing import Dict, List


DEFAULT_RUBRIC_DIMENSIONS = [
    "alignment",
    "instruction_fidelity",
    "compositionality",
    "visual_quality",
]


def compute_overall_score(subscores: Dict[str, float]) -> float:
    values = list(subscores.values())
    return sum(values) / len(values) if values else 0.0
```

Runner requirements:

- Build a rubric-style judge prompt that requests JSON subscores for all four default dimensions
- Parse the JSON response
- Normalize scores to `0.0 - 1.0`
- Derive `error_types` from low rubric dimensions when useful
- Save one JSONL row per sample

Script wrapper requirements:

- Read `--manifest_path`, `--images_root`, `--variant`, `--judge_model`, `--api_provider`, `--output_path`, `--resume`
- Skip already-scored rows when `--resume` is set

- [ ] **Step 4: Rerun the tests**

Run: `pytest tests/test_evaluation_contracts.py -k "default_rubric_dimensions or overall_score_is_mean" -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/genai_bench_runner.py scripts/run_genai_bench.py tests/test_evaluation_contracts.py
git commit -m "feat: add genai bench scoring runner"
```

### Task 6: Add Summary Reporting For Before/After Comparisons

**Files:**
- Create: `D:/Code/T2I-RL-Eval/src/evaluation/reporting.py`
- Create: `D:/Code/T2I-RL-Eval/scripts/summarize_evaluation.py`
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_smoke.py`

- [ ] **Step 1: Write the failing tests for benchmark delta aggregation**

```python
def test_summary_computes_before_after_delta():
    from src.evaluation.reporting import compute_variant_delta

    before_rows = [{"score": 0.4}, {"score": 0.6}]
    after_rows = [{"score": 0.7}, {"score": 0.9}]

    summary = compute_variant_delta(before_rows, after_rows)
    assert summary["before_mean"] == 0.5
    assert summary["after_mean"] == 0.8
    assert summary["delta"] == 0.3


def test_summary_counts_error_types():
    from src.evaluation.reporting import count_error_types

    rows = [
        {"error_types": ["wrong_count", "wrong_attribute"]},
        {"error_types": ["wrong_count"]},
    ]

    assert count_error_types(rows) == {"wrong_count": 2, "wrong_attribute": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluation_smoke.py -k "before_after_delta or counts_error_types" -v`

Expected: `ImportError` because `reporting.py` does not exist yet.

- [ ] **Step 3: Implement summary helpers and export script**

```python
# src/evaluation/reporting.py
from collections import Counter
from typing import Dict, List


def compute_variant_delta(before_rows: List[Dict], after_rows: List[Dict]) -> Dict[str, float]:
    before_mean = sum(row["score"] for row in before_rows) / len(before_rows)
    after_mean = sum(row["score"] for row in after_rows) / len(after_rows)
    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "delta": after_mean - before_mean,
    }


def count_error_types(rows: List[Dict]) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        counter.update(row.get("error_types", []))
    return dict(counter)
```

Script requirements:

- Load `before` and `after` result JSONL files for TIFA and GenAI-Bench
- Compute overall means, deltas, and failure counts
- Export `summary.json`, `summary.csv`, and `summary.md`

- [ ] **Step 4: Rerun the tests**

Run: `pytest tests/test_evaluation_smoke.py -k "before_after_delta or counts_error_types" -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/reporting.py scripts/summarize_evaluation.py tests/test_evaluation_smoke.py
git commit -m "feat: add evaluation reporting"
```

### Task 7: Copy Inference-Only LoRA Assets Into Repository Defaults

**Files:**
- Copy: `D:/Code/T2I-RL-Eval/artifacts/lora/grpo_siliconflow_quick_final/adapter_config.json`
- Copy: `D:/Code/T2I-RL-Eval/artifacts/lora/grpo_siliconflow_quick_final/adapter_model.safetensors`
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_smoke.py`

- [ ] **Step 1: Write the failing test for default LoRA directory resolution**

```python
def test_default_lora_path_points_to_repo_artifact_root():
    from scripts.generate_benchmark_images import default_lora_path

    path = default_lora_path()
    assert path.as_posix().endswith("artifacts/lora/grpo_siliconflow_quick_final")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_evaluation_smoke.py -k "default_lora_path_points" -v`

Expected: `AttributeError` because `default_lora_path` is not implemented yet.

- [ ] **Step 3: Copy the adapter files and wire the default path helper**

Required copy operations:

- Copy `D:/Code/#OpenSoursce/T2I-RL-Project/outputs/grpo_siliconflow_quick/final_checkpoint/adapter_config.json`
  to `D:/Code/T2I-RL-Eval/artifacts/lora/grpo_siliconflow_quick_final/adapter_config.json`
- Copy `D:/Code/#OpenSoursce/T2I-RL-Project/outputs/grpo_siliconflow_quick/final_checkpoint/adapter_model.safetensors`
  to `D:/Code/T2I-RL-Eval/artifacts/lora/grpo_siliconflow_quick_final/adapter_model.safetensors`

Helper implementation:

```python
def default_lora_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "lora" / "grpo_siliconflow_quick_final"
```

- [ ] **Step 4: Rerun the test**

Run: `pytest tests/test_evaluation_smoke.py -k "default_lora_path_points" -v`

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add artifacts/lora/grpo_siliconflow_quick_final/adapter_config.json artifacts/lora/grpo_siliconflow_quick_final/adapter_model.safetensors scripts/generate_benchmark_images.py tests/test_evaluation_smoke.py
git commit -m "chore: vendor lora inference assets"
```

### Task 8: Add Colab Notebook, Documentation, And Dependency List

**Files:**
- Create: `D:/Code/T2I-RL-Eval/notebooks/colab_evaluation.ipynb`
- Create: `D:/Code/T2I-RL-Eval/docs/evaluation.md`
- Create: `D:/Code/T2I-RL-Eval/requirements.txt`

- [ ] **Step 1: Write the failing smoke test for documentation-required commands**

```python
def test_requirements_include_api_and_image_dependencies():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "openai" in requirements
    assert "Pillow" in requirements
    assert "torch" in requirements
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_evaluation_smoke.py -k "requirements_include_api" -v`

Expected: `FileNotFoundError` because `requirements.txt` does not exist yet.

- [ ] **Step 3: Add minimal runtime docs and notebook flow**

`requirements.txt` must include at least:

```text
torch>=2.1.0
Pillow>=10.0.0
openai>=1.0.0
tqdm>=4.66.0
pandas>=2.0.0
matplotlib>=3.8.0
pytest>=7.4.0
```

`docs/evaluation.md` must document:

- benchmark manifest locations
- default LoRA path
- generation command
- TIFA scoring command
- GenAI-Bench scoring command
- summary command
- Colab notebook execution order

Notebook sections:

- environment setup
- API key configuration
- before generation
- after generation
- TIFA scoring
- GenAI-Bench scoring
- summary export

- [ ] **Step 4: Rerun the test**

Run: `pytest tests/test_evaluation_smoke.py -k "requirements_include_api" -v`

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/colab_evaluation.ipynb docs/evaluation.md requirements.txt
git commit -m "docs: add colab evaluation workflow"
```

### Task 9: Run Final Smoke Tests

**Files:**
- Modify: `D:/Code/T2I-RL-Eval/tests/test_evaluation_smoke.py`

- [ ] **Step 1: Add an end-to-end smoke test using fixtures and mocks**

```python
def test_end_to_end_summary_flow(tmp_path):
    from src.evaluation.io import write_jsonl
    from src.evaluation.reporting import compute_variant_delta

    before_path = tmp_path / "before.jsonl"
    after_path = tmp_path / "after.jsonl"

    write_jsonl(before_path, [{"score": 0.4, "error_types": ["wrong_count"]}])
    write_jsonl(after_path, [{"score": 0.8, "error_types": []}])

    summary = compute_variant_delta(
        [{"score": 0.4}],
        [{"score": 0.8}],
    )

    assert summary["delta"] == 0.4
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests -v`

Expected: all contract and smoke tests pass without GPU or live API access.

- [ ] **Step 3: Commit**

```bash
git add tests/test_evaluation_smoke.py
git commit -m "test: add evaluation smoke coverage"
```

## Self-Review

### Spec coverage

- Repository-local manifests: covered by Task 2
- Before/after generation with Janus-Pro-1B + LoRA: covered by Task 3 and Task 7
- TIFA API judge runner: covered by Task 4
- GenAI-Bench API judge runner: covered by Task 5
- Summary outputs and error taxonomy: covered by Task 6
- Colab notebook and docs: covered by Task 8
- Resume and failure handling: covered by Task 3, Task 4, Task 5, and Task 6

### Placeholder scan

- No `TBD`, `TODO`, or “implement later” placeholders remain
- Each task includes concrete files, commands, and code examples

### Type consistency

- The plan consistently uses `sample_id`, `variant`, `score`, `error_types`, and `judge_metadata`
- The `before` and `after` variant naming is consistent across scripts, manifests, and reports
