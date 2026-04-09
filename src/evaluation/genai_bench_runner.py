import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.evaluation.io import append_jsonl, read_jsonl

DEFAULT_RUBRIC_DIMENSIONS = [
    "alignment",
    "instruction_fidelity",
    "compositionality",
    "visual_quality",
]


def compute_overall_score(subscores: Dict[str, float]) -> float:
    values = list(subscores.values())
    return round(sum(values) / len(values), 6) if values else 0.0


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


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


def build_rubric_prompt(prompt: str, category: str, skills: List[str]) -> str:
    rubric = "\n".join(f'- "{name}": score from 0 to 1' for name in DEFAULT_RUBRIC_DIMENSIONS)
    skills_text = ", ".join(skills) if skills else "none"
    return (
        "You are judging a text-to-image sample.\n"
        f"Prompt: {prompt}\n"
        f"Category: {category}\n"
        f"Skills: {skills_text}\n"
        "Return JSON only with numeric subscores and a short reason.\n"
        f"Rubric:\n{rubric}\n"
        'Format: {"alignment": 0.0, "instruction_fidelity": 0.0, "compositionality": 0.0, "visual_quality": 0.0, "reason": "..."}'
    )


def build_openai_client(api_key: Optional[str], base_url: Optional[str] = None):
    from openai import OpenAI

    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def judge_image(client: Any, model: str, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
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


def _coerce_subscores(payload: Dict[str, Any]) -> Dict[str, float]:
    subscores = {}
    for name in DEFAULT_RUBRIC_DIMENSIONS:
        value = payload.get(name, 0.0)
        try:
            subscores[name] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            subscores[name] = 0.0
    return subscores


def derive_error_types(subscores: Dict[str, float]) -> List[str]:
    error_types: List[str] = []
    if subscores.get("alignment", 1.0) < 0.5:
        error_types.append("low_alignment")
    if subscores.get("instruction_fidelity", 1.0) < 0.5:
        error_types.append("missed_instruction")
    if subscores.get("compositionality", 1.0) < 0.5:
        error_types.append("weak_composition")
    if subscores.get("visual_quality", 1.0) < 0.5:
        error_types.append("low_visual_quality")
    return error_types


def score_genai_sample(
    sample: Dict[str, Any],
    image: Image.Image,
    judge_client: Any,
    judge_model: str,
) -> Dict[str, Any]:
    prompt = build_rubric_prompt(sample["prompt"], sample.get("category", ""), sample.get("skills", []))
    response_text = judge_image(judge_client, judge_model, image, prompt)
    payload = extract_json_object(response_text)
    subscores = _coerce_subscores(payload)
    return {
        "benchmark": "genai_bench",
        "sample_id": sample["sample_id"],
        "prompt": sample["prompt"],
        "category": sample.get("category", ""),
        "skills": sample.get("skills", []),
        "score": compute_overall_score(subscores),
        "subscores": subscores,
        "error_types": derive_error_types(subscores),
        "judge_metadata": {
            "judge_model": judge_model,
            "judge_raw": response_text,
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
        image_path = Path(images_root) / "genai_bench" / variant / f"{sample['sample_id']}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing generated image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        result = score_genai_sample(sample, image, judge_client, judge_model)
        result["variant"] = variant
        result["image_path"] = str(image_path)
        append_jsonl(output_path, result)
        results.append(result)
    return results
