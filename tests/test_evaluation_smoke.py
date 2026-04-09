from pathlib import Path

from src.evaluation.benchmarks import GenAIBenchmark, TIFABenchmark
from src.evaluation.io import read_jsonl, write_jsonl
from src.evaluation.reporting import compute_variant_delta, count_error_types, write_summary_outputs
from scripts.generate_benchmark_images import build_image_path, default_lora_path, should_skip_sample


def test_generation_path_uses_variant_and_sample_id(tmp_path: Path):
    path = build_image_path(tmp_path, "tifa", "after", "tifa-0001")
    assert path == tmp_path / "images" / "tifa" / "after" / "tifa-0001.png"


def test_should_skip_existing_image_when_resume(tmp_path: Path):
    target = tmp_path / "images" / "tifa" / "before" / "tifa-0001.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"png")

    assert should_skip_sample(target, resume=True) is True
    assert should_skip_sample(target, resume=False) is False


def test_default_lora_path_points_to_repo_artifact_root():
    assert default_lora_path().as_posix().endswith("artifacts/lora/grpo_siliconflow_quick_final")


def test_manifest_loaders_accept_expected_fields(tmp_path: Path):
    tifa_path = tmp_path / "tifa.jsonl"
    genai_path = tmp_path / "genai.jsonl"
    write_jsonl(
        tifa_path,
        [
            {
                "sample_id": "tifa-0001",
                "prompt": "a red apple on a wooden table",
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
    write_jsonl(
        genai_path,
        [
            {
                "sample_id": "genai-0001",
                "prompt": "two cats on a blue sofa",
                "category": "reasoning",
                "skills": ["counting", "attribute_binding"],
                "source": "genai-bench",
            }
        ],
    )

    assert TIFABenchmark(tifa_path).iter_samples()[0]["sample_id"] == "tifa-0001"
    assert GenAIBenchmark(genai_path).iter_samples()[0]["skills"] == ["counting", "attribute_binding"]


def test_summary_computes_before_after_delta():
    before_rows = [{"score": 0.4}, {"score": 0.6}]
    after_rows = [{"score": 0.7}, {"score": 0.9}]

    summary = compute_variant_delta(before_rows, after_rows)
    assert summary["before_mean"] == 0.5
    assert summary["after_mean"] == 0.8
    assert summary["delta"] == 0.3


def test_summary_counts_error_types():
    rows = [
        {"error_types": ["wrong_count", "wrong_attribute"]},
        {"error_types": ["wrong_count"]},
    ]

    assert count_error_types(rows) == {"wrong_count": 2, "wrong_attribute": 1}


def test_write_summary_outputs(tmp_path: Path):
    summary = {
        "tifa": {
            "before_mean": 0.5,
            "after_mean": 0.8,
            "delta": 0.3,
            "before_count": 2,
            "after_count": 2,
            "before_errors": {},
            "after_errors": {},
        },
        "genai_bench": {
            "before_mean": 0.4,
            "after_mean": 0.6,
            "delta": 0.2,
            "before_count": 1,
            "after_count": 1,
            "before_errors": {},
            "after_errors": {},
        },
    }

    write_summary_outputs(tmp_path, summary)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_requirements_include_api_and_image_dependencies():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "openai" in requirements
    assert "Pillow" in requirements
    assert "torch" in requirements

