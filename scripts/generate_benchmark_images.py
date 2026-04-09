#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmarks import GenAIBenchmark, TIFABenchmark
from src.evaluation.io import append_jsonl
from src.evaluation.schemas import GeneratedSampleRecord


@dataclass
class GenerationConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    height: int = 384
    width: int = 384
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    temperature: float = 1.0


def default_lora_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "lora" / "grpo_siliconflow_quick_final"


def build_image_path(output_dir: Path, benchmark: str, variant: str, sample_id: str) -> Path:
    return Path(output_dir) / "images" / benchmark / variant / f"{sample_id}.png"


def should_skip_sample(target_path: Path, resume: bool) -> bool:
    return resume and target_path.exists()


class JanusProRunner:
    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/Janus-Pro-1B",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.vl_chat_processor = None
        self.tokenizer = None
        self.image_token_num_per_image = 576
        self.img_size = 384
        self.patch_size = 16

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            from janus.models import VLChatProcessor

            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_name_or_path)
        except Exception:
            self.vl_chat_processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        self.tokenizer = self.vl_chat_processor.tokenizer
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **load_kwargs)
        if quantization_config is None:
            self.model = self.model.to(self.dtype).to(self.device).eval()
        else:
            self.model = self.model.eval()
            if hasattr(self.model, "gen_vision_model"):
                self.model.gen_vision_model = self.model.gen_vision_model.to(torch.float16)
        if hasattr(self.model, "gen_vision_model"):
            self._patch_janus_upsample_dtype()

    def _patch_janus_upsample_dtype(self) -> None:
        def _patched_forward(module_self, x):
            target_dtype = x.dtype
            if target_dtype != torch.float32:
                x = F.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest").to(target_dtype)
            else:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            if module_self.with_conv:
                x = module_self.conv(x)
            return x

        for module in self.model.gen_vision_model.modules():
            if module.__class__.__name__ == "Upsample" and hasattr(module, "with_conv"):
                module.forward = _patched_forward.__get__(module, module.__class__)

    def enable_lora(self, lora_path: str, scale: float = 1.0) -> None:
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("Model not loaded")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        if hasattr(self.model, "set_adapter"):
            try:
                self.model.set_adapter(scale)
            except Exception:
                pass

    def disable_lora(self) -> None:
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()

    def generate(self, prompt: Union[str, List[str]], config: GenerationConfig) -> List[Image.Image]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if isinstance(prompt, str):
            prompt = [prompt]
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        images: List[Image.Image] = []
        for single_prompt in prompt:
            images.extend(
                self._generate_single(
                    prompt=single_prompt,
                    temperature=config.temperature,
                    cfg_weight=config.guidance_scale,
                    parallel_size=config.num_images_per_prompt,
                )
            )
        return images

    @torch.inference_mode()
    def _generate_single(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
    ) -> List[Image.Image]:
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        formatted_prompt = sft_format + self.vl_chat_processor.image_start_tag
        input_ids = torch.LongTensor(self.tokenizer.encode(formatted_prompt))
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(self.device)
        for idx in range(parallel_size * 2):
            tokens[idx, :] = input_ids
            if idx % 2 != 0:
                tokens[idx, 1:-1] = self.vl_chat_processor.pad_id
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        compute_dtype = torch.float16 if self.device.startswith("cuda") else self.dtype
        inputs_embeds = inputs_embeds.to(compute_dtype)
        generated_tokens = torch.zeros((parallel_size, self.image_token_num_per_image), dtype=torch.int).to(self.device)
        past_key_values = None
        autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype) if self.device.startswith("cuda") else nullcontext()
        with autocast_ctx:
            for step in range(self.image_token_num_per_image):
                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values if step != 0 else None,
                )
                past_key_values = outputs.past_key_values
                hidden_states = outputs.last_hidden_state
                logits = self.model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, step] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1).to(compute_dtype)
        return self._decode_tokens_to_images(generated_tokens, parallel_size)

    def _decode_tokens_to_images(self, generated_tokens: torch.Tensor, parallel_size: int) -> List[Image.Image]:
        compute_dtype = torch.float16 if self.device.startswith("cuda") else self.dtype
        autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype) if self.device.startswith("cuda") else nullcontext()
        with autocast_ctx:
            dec = self.model.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, self.img_size // self.patch_size, self.img_size // self.patch_size],
            )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return [Image.fromarray(dec[idx]) for idx in range(parallel_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark images for TIFA and GenAI-Bench")
    parser.add_argument("--benchmark", choices=["tifa", "genai_bench"], required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    parser.add_argument("--base_model", type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    return parser.parse_args()


def load_manifest(benchmark: str, manifest_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    if benchmark == "tifa":
        return TIFABenchmark(manifest_path).iter_samples(limit=limit)
    return GenAIBenchmark(manifest_path).iter_samples(limit=limit)


def save_generated_sample(
    output_dir: Path,
    benchmark: str,
    variant: str,
    sample: Dict[str, Any],
    image: Image.Image,
    generation_config: GenerationConfig,
    model_name: str,
    checkpoint_or_lora: str,
) -> Path:
    image_path = build_image_path(output_dir, benchmark, variant, sample["sample_id"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    record = GeneratedSampleRecord(
        benchmark=benchmark,
        sample_id=sample["sample_id"],
        prompt=sample["prompt"],
        variant=variant,
        seed=generation_config.seed if generation_config.seed is not None else -1,
        model_name=model_name,
        checkpoint_or_lora=checkpoint_or_lora,
        image_path=str(image_path),
        generation_config=asdict(generation_config),
    )
    append_jsonl(output_dir / "results" / "generated_samples.jsonl", record.to_dict())
    return image_path


def run() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    samples = load_manifest(args.benchmark, manifest_path, args.limit)
    results_path = output_dir / "results" / "generated_samples.jsonl"
    if results_path.exists() and not args.resume:
        results_path.unlink()
    generator = JanusProRunner(
        model_name_or_path=args.base_model,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    generator.load_model()
    if args.variant == "after":
        lora_path = Path(args.lora_path) if args.lora_path else default_lora_path()
        generator.enable_lora(str(lora_path))
    else:
        generator.disable_lora()

    generation_config = GenerationConfig(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    for sample in samples:
        image_path = build_image_path(output_dir, args.benchmark, args.variant, sample["sample_id"])
        if should_skip_sample(image_path, args.resume):
            continue
        images = generator.generate(sample["prompt"], generation_config)
        save_generated_sample(
            output_dir=output_dir,
            benchmark=args.benchmark,
            variant=args.variant,
            sample=sample,
            image=images[0],
            generation_config=generation_config,
            model_name=args.base_model,
            checkpoint_or_lora=str(args.lora_path or default_lora_path()) if args.variant == "after" else "base",
        )


if __name__ == "__main__":
    run()
