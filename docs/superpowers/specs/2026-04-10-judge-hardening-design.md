# Judge Hardening Design

## Goal

Make TIFA and GenAI-Bench evaluation robust against OpenAI-compatible judge responses that are empty, truncated, or wrapped in provider-specific formats.

## Current Problems

- The runners only read `response.choices[0].message.content`.
- Qwen-compatible endpoints may emit thinking content unless explicitly disabled.
- `max_tokens` is low enough to truncate JSON answers.
- Empty content and malformed JSON are treated as final outputs instead of retryable failures.

## Design

### Shared request behavior

- Add a shared helper in the evaluation package to send chat-completions requests.
- Default to larger output budgets for both benchmarks.
- Pass `extra_body.chat_template_kwargs.enable_thinking=False` for Qwen models.
- Preserve compatibility with non-Qwen OpenAI-compatible endpoints by only adding this flag conditionally.

### Shared response extraction

- Extract text from `message.content` whether it arrives as a plain string or as structured content parts.
- Capture `finish_reason` and a serialized `choice` payload for diagnostics.
- Parse JSON objects from fenced JSON, plain JSON, or mixed text.

### Retry policy

- Retry a small fixed number of times when the response text is empty or when JSON parsing fails.
- Keep the last raw response for debugging even when parsing fails.

### Benchmark integration

- TIFA keeps its existing scoring semantics but stores richer judge metadata.
- GenAI-Bench reuses the same helper and parsing flow.

## Testing

- Add unit tests for:
  - conditional Qwen request options
  - structured content extraction
  - retry on empty response
  - retry on malformed/truncated JSON followed by valid JSON
