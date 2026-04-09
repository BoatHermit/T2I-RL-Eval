"""Contract tests for evaluation data structures and manifests."""

from pathlib import Path

import pytest

from src.evaluation.benchmarks import GenAIBenchmark, TIFABenchmark
from src.evaluation.io import read_json, read_jsonl, write_json, write_jsonl
from src.evaluation.schemas import GeneratedSampleRecord, ScoredSampleRecord


def test_generated_sample_record_round_trip(tmp_path: Path):
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


def test_json_helpers_round_trip(tmp_path: Path):
    payload = {"benchmark": "tifa", "count": 2}
    path = tmp_path / "payload.json"
    write_json(path, payload)
    assert read_json(path) == payload


def test_tifa_manifest_loads_seed_samples():
    benchmark = TIFABenchmark()
    samples = benchmark.iter_samples()

    assert len(samples) >= 4000
    assert samples[0]["source"] == "tifa_v1.0"
    assert sum(len(sample["questions"]) for sample in samples) >= 25000
    assert samples[0]["questions"][0]["question_type"] in {"object", "attribute", "count", "relation", "other"}


def test_genai_manifest_loads_seed_samples():
    benchmark = GenAIBenchmark()
    samples = benchmark.iter_samples()

    assert len(samples) == 1600
    assert samples[0]["source"] == "GenAI-Bench-1600"
    assert isinstance(samples[0]["skills"], list)
    assert "reference_models" in samples[0]


def test_manifest_validation_rejects_missing_fields(tmp_path: Path):
    bad_manifest = tmp_path / "bad.jsonl"
    write_jsonl(
        bad_manifest,
        [
            {
                "sample_id": "broken-1",
                "prompt": "missing fields",
                "source": "tifa",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing required fields"):
        TIFABenchmark(bad_manifest).iter_samples()
