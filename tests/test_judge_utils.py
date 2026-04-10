from types import SimpleNamespace

from src.evaluation.judge_utils import (
    build_chat_completion_kwargs,
    extract_json_object,
    extract_text_content,
    request_json_chat_completion,
)


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = SimpleNamespace(content=content)
        self.finish_reason = finish_reason

    def model_dump(self, mode="json"):
        return {
            "message": {"content": self.message.content},
            "finish_reason": self.finish_reason,
        }


class FakeClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(choices=[FakeChoice(content)])


def test_build_chat_completion_kwargs_disables_thinking_for_qwen():
    kwargs = build_chat_completion_kwargs("Qwen/Qwen3.5-9B", [{"role": "user", "content": "hi"}], 512)
    assert kwargs["extra_body"]["enable_thinking"] is False
    assert kwargs["response_format"]["type"] == "json_object"


def test_build_chat_completion_kwargs_skips_extra_body_for_non_qwen():
    kwargs = build_chat_completion_kwargs("gpt-4.1-mini", [{"role": "user", "content": "hi"}], 512)
    assert "extra_body" not in kwargs
    assert kwargs["response_format"]["type"] == "json_object"


def test_extract_text_content_reads_structured_parts():
    content = [
        {"type": "output_text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert extract_text_content(content) == "first\nsecond"


def test_extract_json_object_reads_fenced_json():
    text = '```json\n{"answer":"yes","confidence":1.0}\n```'
    assert extract_json_object(text)["answer"] == "yes"


def test_request_json_chat_completion_retries_empty_response():
    client = FakeClient(["", '{"answer":"yes","confidence":1.0,"reason":"ok"}'])
    result = request_json_chat_completion(
        client,
        "Qwen/Qwen3.5-9B",
        [{"role": "user", "content": "hi"}],
        max_tokens=512,
        max_attempts=3,
    )
    assert result["parsed_json"]["answer"] == "yes"
    assert result["attempts"] == 2
    assert client.calls[0]["extra_body"]["enable_thinking"] is False


def test_request_json_chat_completion_retries_truncated_json():
    client = FakeClient(['{"answer":"no"', '{"answer":"yes","confidence":1.0,"reason":"ok"}'])
    result = request_json_chat_completion(
        client,
        "gpt-4.1-mini",
        [{"role": "user", "content": "hi"}],
        max_tokens=512,
        max_attempts=3,
    )
    assert result["parsed_json"]["answer"] == "yes"
    assert result["attempts"] == 2
