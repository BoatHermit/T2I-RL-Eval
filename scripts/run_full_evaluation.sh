#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Server evaluation configuration
# Override any variable below via environment variables before running:
#   API_KEY=... API_BASE_URL=... JUDGE_MODEL=... bash scripts/run_full_evaluation.sh
# Notebook-style aliases are supported:
#   API_KEY, API_BASE_URL, JUDGE_MODEL, JUDGE_MAX_WORKERS, JUDGE_LOG_EVERY
# -----------------------------------------------------------------------------

: "${PYTHON_BIN:=python}"
: "${API_KEY:=}"
: "${API_BASE_URL:=https://api.openai.com/v1}"
: "${JUDGE_MODEL:=gpt-4.1-mini}"
: "${BASE_MODEL:=deepseek-ai/Janus-Pro-1B}"
: "${LORA_DIR:=artifacts/lora/grpo_siliconflow_quick_final}"
: "${OUTPUT_DIR:=outputs/evaluation}"

: "${TIFA_MANIFEST:=data/evaluation/tifa/samples.jsonl}"
: "${GENAI_MANIFEST:=data/evaluation/genai_bench/samples.jsonl}"

: "${GEN_DTYPE:=float16}"
: "${PROMPT_BATCH_SIZE:=}"
: "${PROMPT_BATCH_SIZE_BEFORE:=32}"
: "${PROMPT_BATCH_SIZE_AFTER:=16}"

: "${JUDGE_MAX_WORKERS:=}"
: "${TIFA_MAX_WORKERS:=20}"
: "${GENAI_MAX_WORKERS:=20}"
: "${JUDGE_LOG_EVERY:=1}"

: "${LIMIT:=}"
: "${RESUME:=1}"

: "${RUN_GENERATE_BEFORE:=1}"
: "${RUN_GENERATE_AFTER:=1}"
: "${RUN_TIFA:=1}"
: "${RUN_GENAI:=1}"
: "${RUN_SUMMARY:=1}"

if [[ -n "${PROMPT_BATCH_SIZE}" ]]; then
  PROMPT_BATCH_SIZE_BEFORE="${PROMPT_BATCH_SIZE}"
  PROMPT_BATCH_SIZE_AFTER="${PROMPT_BATCH_SIZE}"
fi

if [[ -n "${JUDGE_MAX_WORKERS}" ]]; then
  TIFA_MAX_WORKERS="${JUDGE_MAX_WORKERS}"
  GENAI_MAX_WORKERS="${JUDGE_MAX_WORKERS}"
fi

if [[ -n "${API_KEY}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${API_KEY}"
fi

if [[ -n "${API_BASE_URL}" && -z "${OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="${API_BASE_URL}"
fi

export JUDGE_MODEL

if [[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_COMPAT_API_KEY:-}" && -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo "[run_full] warning: no judge API key found in OPENAI_API_KEY / OPENAI_COMPAT_API_KEY / SILICONFLOW_API_KEY"
  echo "[run_full] generation can still run, but judge stages will fail until a key is provided"
fi

echo "[run_full] project_root=${PROJECT_ROOT}"
echo "[run_full] base_model=${BASE_MODEL}"
echo "[run_full] lora_dir=${LORA_DIR}"
echo "[run_full] output_dir=${OUTPUT_DIR}"
echo "[run_full] gen_dtype=${GEN_DTYPE}"
echo "[run_full] prompt_batch_size_before=${PROMPT_BATCH_SIZE_BEFORE}"
echo "[run_full] prompt_batch_size_after=${PROMPT_BATCH_SIZE_AFTER}"
echo "[run_full] judge_model=${JUDGE_MODEL}"
echo "[run_full] openai_base_url=${OPENAI_BASE_URL:-${OPENAI_API_BASE:-unset}}"
echo "[run_full] tifa_max_workers=${TIFA_MAX_WORKERS}"
echo "[run_full] genai_max_workers=${GENAI_MAX_WORKERS}"
echo "[run_full] judge_log_every=${JUDGE_LOG_EVERY}"
echo "[run_full] limit=${LIMIT:-full}"
echo "[run_full] resume=${RESUME}"
echo "[run_full] stages generate_before=${RUN_GENERATE_BEFORE} generate_after=${RUN_GENERATE_AFTER} tifa=${RUN_TIFA} genai=${RUN_GENAI} summary=${RUN_SUMMARY}"

COMMON_RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  COMMON_RESUME_ARGS+=(--resume)
fi

COMMON_LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  COMMON_LIMIT_ARGS+=(--limit "${LIMIT}")
fi

run_generate() {
  local benchmark="$1"
  local variant="$2"
  local manifest="$3"
  local prompt_batch_size="$4"

  local extra_args=(
    --benchmark "${benchmark}"
    --variant "${variant}"
    --manifest_path "${manifest}"
    --base_model "${BASE_MODEL}"
    --output_dir "${OUTPUT_DIR}"
    --dtype "${GEN_DTYPE}"
    --prompt_batch_size "${prompt_batch_size}"
  )

  if [[ "${variant}" == "after" ]]; then
    extra_args+=(--lora_path "${LORA_DIR}")
  fi

  echo "[run_full] generate benchmark=${benchmark} variant=${variant} prompt_batch_size=${prompt_batch_size}"
  "${PYTHON_BIN}" scripts/generate_benchmark_images.py \
    "${extra_args[@]}" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${COMMON_RESUME_ARGS[@]}"
}

run_tifa_stage() {
  local variant="$1"
  local output_path="${OUTPUT_DIR}/results/tifa_${variant}.jsonl"
  echo "[run_full] tifa variant=${variant} workers=${TIFA_MAX_WORKERS}"
  "${PYTHON_BIN}" scripts/run_tifa.py \
    --manifest_path "${TIFA_MANIFEST}" \
    --images_root "${OUTPUT_DIR}/images" \
    --variant "${variant}" \
    --output_path "${output_path}" \
    --max_workers "${TIFA_MAX_WORKERS}" \
    --log_every "${JUDGE_LOG_EVERY}" \
    "${COMMON_RESUME_ARGS[@]}"
}

run_genai_stage() {
  local variant="$1"
  local output_path="${OUTPUT_DIR}/results/genai_${variant}.jsonl"
  echo "[run_full] genai variant=${variant} workers=${GENAI_MAX_WORKERS}"
  "${PYTHON_BIN}" scripts/run_genai_bench.py \
    --manifest_path "${GENAI_MANIFEST}" \
    --images_root "${OUTPUT_DIR}/images" \
    --variant "${variant}" \
    --output_path "${output_path}" \
    --max_workers "${GENAI_MAX_WORKERS}" \
    --log_every "${JUDGE_LOG_EVERY}" \
    "${COMMON_RESUME_ARGS[@]}"
}

run_summary_stage() {
  echo "[run_full] summarize"
  "${PYTHON_BIN}" scripts/summarize_evaluation.py \
    --tifa_results_before "${OUTPUT_DIR}/results/tifa_before.jsonl" \
    --tifa_results_after "${OUTPUT_DIR}/results/tifa_after.jsonl" \
    --genai_results_before "${OUTPUT_DIR}/results/genai_before.jsonl" \
    --genai_results_after "${OUTPUT_DIR}/results/genai_after.jsonl" \
    --output_dir "${OUTPUT_DIR}/reports"
}

if [[ "${RUN_GENERATE_BEFORE}" == "1" ]]; then
  run_generate "tifa" "before" "${TIFA_MANIFEST}" "${PROMPT_BATCH_SIZE_BEFORE}"
  run_generate "genai_bench" "before" "${GENAI_MANIFEST}" "${PROMPT_BATCH_SIZE_BEFORE}"
fi

if [[ "${RUN_GENERATE_AFTER}" == "1" ]]; then
  run_generate "tifa" "after" "${TIFA_MANIFEST}" "${PROMPT_BATCH_SIZE_AFTER}"
  run_generate "genai_bench" "after" "${GENAI_MANIFEST}" "${PROMPT_BATCH_SIZE_AFTER}"
fi

if [[ "${RUN_TIFA}" == "1" ]]; then
  run_tifa_stage "before"
  run_tifa_stage "after"
fi

if [[ "${RUN_GENAI}" == "1" ]]; then
  run_genai_stage "before"
  run_genai_stage "after"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  run_summary_stage
fi

echo "[run_full] done"
