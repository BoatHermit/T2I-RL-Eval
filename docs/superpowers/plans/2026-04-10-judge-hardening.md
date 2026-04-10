# Judge Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TIFA and GenAI-Bench judge calls resilient to empty, truncated, and provider-specific responses.

**Architecture:** Introduce a shared helper module for OpenAI-compatible judge requests and response extraction, then switch both evaluation runners to use it. Cover the failure modes with focused regression tests so behavior stays stable across providers.

**Tech Stack:** Python, OpenAI Python SDK, pytest

---

### Task 1: Add shared judge helpers

**Files:**
- Create: `D:\Code\T2I-RL-Eval\src\evaluation\judge_utils.py`

- [ ] **Step 1: Add request and response helper functions**

- [ ] **Step 2: Include Qwen-specific non-thinking request options**

- [ ] **Step 3: Add retry-aware JSON response helper**

### Task 2: Migrate TIFA runner

**Files:**
- Modify: `D:\Code\T2I-RL-Eval\src\evaluation\tifa_runner.py`

- [ ] **Step 1: Replace direct chat-completions call with shared helper**

- [ ] **Step 2: Store richer metadata for failed and successful judge responses**

### Task 3: Migrate GenAI-Bench runner

**Files:**
- Modify: `D:\Code\T2I-RL-Eval\src\evaluation\genai_bench_runner.py`

- [ ] **Step 1: Replace direct chat-completions call with shared helper**

- [ ] **Step 2: Keep existing scoring semantics while improving metadata**

### Task 4: Add regression tests

**Files:**
- Create: `D:\Code\T2I-RL-Eval\tests\test_judge_utils.py`

- [ ] **Step 1: Cover request option selection and content extraction**

- [ ] **Step 2: Cover retries for empty and malformed JSON responses**

### Task 5: Verify

**Files:**
- Test: `D:\Code\T2I-RL-Eval\tests\test_judge_utils.py`
- Test: `D:\Code\T2I-RL-Eval\tests\test_evaluation_contracts.py`
- Test: `D:\Code\T2I-RL-Eval\tests\test_evaluation_smoke.py`

- [ ] **Step 1: Run focused pytest targets**

- [ ] **Step 2: Fix any regressions and rerun**
