"""Inkling message channels when the server falls back to delimiter parsing (no usable response template):
thinking -> reasoning_content, answer markers stripped, tool calls and argument streaming intact.
"""

import json

import pytest

from mlx_vlm.server.inkling_channels import (
    InklingChannelRewriter,
    has_inkling_markers,
    is_inkling_tokenizer,
)
from mlx_vlm.server.responses_state import (
    InklingThinkingStreamState,
    ThinkingStreamState,
    _split_thinking,
    make_response_stream_state,
    parse_assistant_tool_calls,
)
from mlx_vlm.tools import process_tool_calls
from mlx_vlm.tools.parsers import inkling
from mlx_vlm.tools.streaming import ToolCallDeltaState

THINK = "<|message_model|><|content_thinking|>{}<|end_message|>"
TEXT = "<|message_model|><|content_text|>{}<|end_message|>"
CALL = (
    "<|message_model|>write_file<|content_invoke_tool_json|>"
    '{{"name":"write_file","args":{{"path":"a.py","content":{}}}}}<|end_message|>'
)
END = "<|content_model_end_sampling|>"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }
]


class _Tokenizer:
    """An Inkling tokenizer without a response template (forces the fallback)."""

    response_template = None

    def get_vocab(self):
        return {"<|content_thinking|>": 1, "<|content_text|>": 2, "<|end_message|>": 3}


class _Processor:
    tokenizer = _Tokenizer()


def rewrite(text, chunk_size=None):
    rewriter = InklingChannelRewriter()
    if not chunk_size:
        return rewriter.feed(text, final=True)
    parts = []
    for offset in range(0, len(text), chunk_size):
        last = offset + chunk_size >= len(text)
        parts.append(rewriter.feed(text[offset : offset + chunk_size], final=last))
    return "".join(parts)


@pytest.mark.parametrize("chunk_size", [None, 1, 7, 31])
def test_rewriter_is_chunk_independent(chunk_size):
    raw = THINK.format("Need answer.") + TEXT.format("Soki soba.") + END
    assert rewrite(raw, chunk_size) == "<think>Need answer.</think>Soki soba."


@pytest.mark.parametrize("thinking", ["The user asks. Need answer.", ""])
def test_split_thinking_fallback(thinking):
    raw = THINK.format(thinking) + TEXT.format("Soki soba.") + END
    reasoning, content = _split_thinking(raw)
    assert reasoning == (thinking or None)
    assert content == "Soki soba."


def test_answer_without_thinking():
    assert _split_thinking(TEXT.format("Hello.") + END) == (None, "Hello.")


@pytest.mark.parametrize("chunk_size", [1, 7, 31])
def test_stream_fallback_reasoning_and_content(chunk_size):
    raw = THINK.format("Plan first.") + TEXT.format("Soki soba.") + END
    state = make_response_stream_state(_Processor())
    assert isinstance(state, InklingThinkingStreamState)
    reasoning, content = [], []
    for offset in range(0, len(raw), chunk_size):
        last = offset + chunk_size >= len(raw)
        delta = state.feed(raw[offset : offset + chunk_size], last=last)
        reasoning.append(delta.reasoning or "")
        content.append(delta.content or "")
    assert "".join(reasoning).strip() == "Plan first."
    assert "".join(content) == "Soki soba."


def test_other_models_keep_the_plain_state():
    class Plain:
        response_template = None

        def get_vocab(self):
            return {"<think>": 1}

    class Proc:
        tokenizer = Plain()

    state = make_response_stream_state(Proc())
    assert type(state) is ThinkingStreamState


def test_tool_call_after_thinking():
    raw = THINK.format("Write the file.") + CALL.format(json.dumps("x = 1\n"))
    reasoning, _ = _split_thinking(raw)
    assert reasoning == "Write the file."
    result = parse_assistant_tool_calls(raw, inkling, TOOLS)
    assert result.calls[0]["function"]["name"] == "write_file"
    assert json.loads(result.calls[0]["function"]["arguments"]) == {
        "path": "a.py",
        "content": "x = 1\n",
    }
    assert result.remaining_text == ""


def test_answer_after_tool_result():
    raw = THINK.format("The tests passed.") + TEXT.format("All 12 tests pass.") + END
    assert _split_thinking(raw) == ("The tests passed.", "All 12 tests pass.")


@pytest.mark.parametrize("chunk_size", [1, 7, 31])
def test_tool_call_stream_like_the_server(chunk_size):
    # The server parses tool deltas from the raw text through a plain state and
    # shows the content from the response stream state: arguments still stream
    # and no Inkling markup reaches the visible content.
    body = 'print("hello")\n' * 40
    raw = THINK.format("Write it.") + CALL.format(json.dumps(body)) + END
    visible = make_response_stream_state(_Processor())
    plain = ThinkingStreamState()
    tools = ToolCallDeltaState(inkling, TOOLS)
    arguments, reasoning, content = "", "", ""
    for offset in range(0, len(raw), chunk_size):
        last = offset + chunk_size >= len(raw)
        piece = raw[offset : offset + chunk_size]
        delta = visible.feed(piece, last=last)
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for tool_delta in tools.feed(plain.feed(piece, last=last).content or ""):
            arguments += tool_delta["function"].get("arguments", "")
    final = process_tool_calls(CALL.format(json.dumps(body)), inkling, TOOLS)
    assert tools.finish(final.calls) == []
    assert arguments == final.calls[0]["function"]["arguments"]
    assert reasoning.strip() == "Write it."
    assert content == ""


def test_detection_helpers():
    assert is_inkling_tokenizer(_Tokenizer())
    assert has_inkling_markers(TEXT.format("x"))
    assert not has_inkling_markers("<think>x</think>y")
