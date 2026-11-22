"""Inkling message channels for the server.

Inkling models write each turn as messages:
``<|message_model|><|content_thinking|>...<|end_message|>`` (thinking),
``<|message_model|><|content_text|>...<|end_message|>`` (the answer) and
``<|message_model|>NAME<|content_invoke_tool_json|>{...}<|end_message|>`` (a tool call).

The model's response template (tokenizer_config.json) normally splits these channels. When it
cannot be used, the server falls back to delimiter parsing; there ``InklingChannelRewriter``
turns the text, incrementally, into the form that parsing understands: thinking becomes
``<think>...</think>`` (reasoning_content), the answer channel loses its markers, and
tool-call messages pass through unchanged for the Inkling tool parser.
"""

from __future__ import annotations

MESSAGE = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
END = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
_MARKERS = (MESSAGE, THINKING, TEXT, END, END_SAMPLING)


def is_inkling_tokenizer(tokenizer) -> bool:
    """True when the tokenizer has Inkling's message-channel tokens."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return False
    return THINKING in vocab and TEXT in vocab and END in vocab


def has_inkling_markers(text: str) -> bool:
    """True when raw text contains Inkling's thinking or answer channel markers."""
    return bool(text) and (THINKING in text or TEXT in text)


def _held_suffix(text: str) -> int:
    """Length of the longest suffix of ``text`` that may start a marker."""
    hold = 0
    for marker in _MARKERS:
        for length in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:length]):
                hold = max(hold, length)
                break
    return hold


class InklingChannelRewriter:
    """Incremental rewrite of Inkling message channels (see module docstring)."""

    def __init__(self, keep_tool_calls: bool = True):
        # keep_tool_calls=False drops tool-call messages (for user-visible content
        # streams; the server parses tool calls from the raw text separately).
        self.keep_tool_calls = keep_tool_calls
        self.buffer = ""
        self.state = "top"  # top | message | thinking | text | raw

    def feed(self, text: str, final: bool = False) -> str:
        self.buffer += text or ""
        out = []
        while self.buffer:
            idx, marker = -1, ""
            for m in _MARKERS:
                i = self.buffer.find(m)
                if i >= 0 and (idx < 0 or i < idx):
                    idx, marker = i, m
            if idx < 0:
                hold = 0 if final else _held_suffix(self.buffer)
                plain = self.buffer[: len(self.buffer) - hold]
                self.buffer = self.buffer[len(self.buffer) - hold :]
                self._text(plain, out)
                break
            self._text(self.buffer[:idx], out)
            self.buffer = self.buffer[idx + len(marker) :]
            self._marker(marker, out)
        if final and self.state == "message":
            self.state = "top"
        return "".join(out)

    def _text(self, text: str, out: list) -> None:
        if not text:
            return
        if self.state == "message":
            # A message that is neither thinking nor text: a tool call. Keep it raw.
            if self.keep_tool_calls:
                out.append(MESSAGE)
            self.state = "raw"
        if self.state == "raw" and not self.keep_tool_calls:
            return
        out.append(text)

    def _marker(self, marker: str, out: list) -> None:
        state = self.state
        if marker == MESSAGE:
            if state == "message" and self.keep_tool_calls:
                out.append(MESSAGE)
            elif state == "thinking":
                out.append("</think>")
            elif state == "raw" and self.keep_tool_calls:
                out.append(END)
            self.state = "message"
        elif marker == THINKING:
            if state == "message":
                out.append("<think>")
                self.state = "thinking"
            else:
                out.append(marker)
        elif marker == TEXT:
            if state == "message":
                self.state = "text"
            elif state == "thinking":
                out.append("</think>")
                self.state = "text"
            else:
                out.append(marker)
        elif marker == END:
            if state == "thinking":
                out.append("</think>")
            elif state == "raw" and self.keep_tool_calls:
                out.append(END)
            elif state == "message" and self.keep_tool_calls:
                out.append(MESSAGE + END)
            self.state = "top"
        elif marker == END_SAMPLING:
            if state == "thinking":
                out.append("</think>")
            elif state == "raw" and self.keep_tool_calls:
                out.append(END)
            self.state = "top"
