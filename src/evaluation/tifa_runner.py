import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from PIL import Image

from src.evaluation.io import append_jsonl, read_jsonl

ERROR_BY_QUESTION_TYPE = {
    "object": "missing_object",
    "attribute": "wrong_attribute",
    "count": "wrong_count",
    "relation": "wrong_relation",
}


def map_question_type_to_error(question_type: str) -> str:
    return ERROR_BY_QUESTION_TYPE.get(question_type.lower(), "other")


def compute_question_accuracy(question_results: List[Dict[str, Any]]) -> float:
    if not question_results:
        return 0.0
    return sum(1 for row in question_results if row.get("correct")) / len(question_results)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def answers_match(predicted: str, expected: str) -> bool:
    predicted_norm = normalize_text(predicted)
    expected_norm = normalize_text(expected)
    if not predicted_norm or not expected_norm:
        return False
    if predicted_norm == expected_norm:
        return True
    return expected_norm in predicted_norm or predicted_norm in expected_norm


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def build_question_prompt(prompt: str, question: str, expected_answer: str, question_type: str) -> str:
    return (
        "You are judging whether an image matches a text prompt.\n"
        f"Prompt: {prompt}\n"
        f"Question type: {question_type}\n"
        f"Question: {question}\n"
        f"Expected answer: {expected_answer}\n"
        "Answer with JSON only in the form "
        '{"answer": "...", "confidence": 0.0, "reason": "..."}'
    )


def build_openai_client(api_key: Optional[str], base_url: Optional[str] = None):
    from openai import OpenAI

    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def judge_question(
    client: Any,
    model: str,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 150,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(image)}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def score_tifa_sample(
    sample: Dict[str, Any],
    image: Image.Image,
    judge_client: Any,
    judge_model: str,
) -> Dict[str, Any]:
    question_results: List[Dict[str, Any]] = []
    error_types: List[str] = []
    for question in sample["questions"]:
        prompt = build_question_prompt(
            sample["prompt"],
            question["question"],
            question["expected_answer"],
            question["question_type"],
        )
        response_text = judge_question(judge_client, judge_model, image, prompt)
        response_json = extract_json_object(response_text)
        predicted = str(response_json.get("answer", response_text)).strip()
        correct = answers_match(predicted, question["expected_answer"])
        question_results.append(
            {
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "predicted_answer": predicted,
                "question_type": question["question_type"],
                "correct": correct,
                "judge_raw": response_text,
                "judge_json": response_json,
            }
        )
        if not correct:
            error_types.append(map_question_type_to_error(question["question_type"]))

    question_accuracy = compute_question_accuracy(question_results)
    return {
        "benchmark": "tifa",
        "sample_id": sample["sample_id"],
        "prompt": sample["prompt"],
        "category": sample.get("category", ""),
        "score": question_accuracy,
        "question_accuracy": question_accuracy,
        "question_results": question_results,
        "error_types": sorted(set(error_types)),
        "judge_metadata": {
            "judge_model": judge_model,
            "question_count": len(question_results),
        },
    }


def evaluate_manifest(
    manifest_path: Path,
    images_root: Path,
    variant: str,
    judge_client: Any,
    judge_model: str,
    output_path: Path,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    output_path = Path(output_path)
    if output_path.exists() and not resume:
        output_path.unlink()
    existing_ids = set()
    if resume and output_path.exists():
        existing_ids = {row["sample_id"] for row in read_jsonl(output_path)}

    results: List[Dict[str, Any]] = []
    for sample in read_jsonl(manifest_path):
        if sample["sample_id"] in existing_ids:
            continue
        image_path = Path(images_root) / "tifa" / variant / f"{sample['sample_id']}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing generated image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        result = score_tifa_sample(sample, image, judge_client, judge_model)
        result["variant"] = variant
        result["image_path"] = str(image_path)
        append_jsonl(output_path, result)
        results.append(result)
    return results
