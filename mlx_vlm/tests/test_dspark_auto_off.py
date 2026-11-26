"""inkling2-mlx: DSpark drafting turns itself off while plain decode is faster.

The controller decides from tokens per second; the plain spans must keep the
drafter's captures so drafting resumes exactly, and greedy output must equal
plain greedy decode whatever the switching does.
"""

import mlx.core as mx
import pytest

from mlx_vlm.speculative import dflash as D


def test_controller_switches_and_probes():
    a = D._DraftAutoOff()
    assert a.plain_span(10, 500) == 0  # warm-up
    for _ in range(a.SPEC_ROUNDS + 1):  # first round is not timed
        a.spec_done(2, 0.03)  # ~67 tok/s drafting
    assert a.plain_span(40, 500) == a.FIRST  # measure plain once
    a.plain_done(15, 15 / 86.0)
    assert a.plain_span(56, 500) == 0  # needs probe rounds first
    for _ in range(a.PROBE):
        a.spec_done(2, 0.03)
    assert a.plain_span(62, 500) == a.SPAN
    a.plain_done(31, 31 / 86.0)
    for _ in range(a.PROBE):
        a.spec_done(2, 0.03)
    assert a.plain_span(100, 500) == 2 * a.SPAN  # spans grow while plain wins
    a.plain_done(63, 63 / 86.0)
    for _ in range(a.PROBE):
        a.spec_done(4, 0.03)  # ~133 tok/s: drafting pays again
    assert a.plain_span(200, 500) == 0
    assert a.span == a.SPAN
    assert a.plain_span(498, 500) == 0  # too few tokens left


def test_env_switch(monkeypatch):
    monkeypatch.setenv(D._AUTO_OFF_ENV, "0")
    assert not D._auto_off_enabled()
    monkeypatch.delenv(D._AUTO_OFF_ENV)
    assert D._auto_off_enabled()


class _Always(D._DraftAutoOff):
    """Switches as often as allowed: plain spans of 3 after every 2 rounds."""

    WARM = 2
    SPEC_ROUNDS = 1
    PROBE = 1
    FIRST = 3
    SPAN = 3
    SPAN_MAX = 3
    MARGIN = 1e9

    def plain_span(self, emitted, max_tokens):
        span = super().plain_span(emitted, max_tokens + 8)
        return min(span, max_tokens - emitted)


@pytest.mark.parametrize("seed", [41, None])
def test_switching_keeps_greedy_output(monkeypatch, seed):
    from mlx_vlm.tests.test_speculative import generated, inkling2_dspark_case

    mx.random.seed(73)
    target, draft = inkling2_dspark_case(mx.float32)
    plain = generated(target, None, temperature=0, seed=seed)
    spans = []
    real = D._plain_decode_block

    def counting(*a, **k):
        spans.append(a[4])
        return real(*a, **k)

    monkeypatch.setattr(D, "_DraftAutoOff", _Always)
    monkeypatch.setattr(D, "_plain_decode_block", counting)
    got = generated(target, draft, temperature=0, seed=seed)
    assert got == plain
    assert len(spans) >= 2 and draft.accept_lens


def test_switching_sampled_runs(monkeypatch):
    from mlx_vlm.tests.test_speculative import generated, inkling2_dspark_case

    mx.random.seed(73)
    target, draft = inkling2_dspark_case(mx.float32)
    spans = []
    real = D._plain_decode_block
    monkeypatch.setattr(D, "_DraftAutoOff", _Always)
    monkeypatch.setattr(
        D, "_plain_decode_block", lambda *a, **k: spans.append(1) or real(*a, **k)
    )
    a = generated(target, draft, temperature=0.7, seed=41)
    b = generated(target, draft, temperature=0.7, seed=41)
    assert a == b and len(a) == 12 and spans  # seeded: repeatable


def test_short_requests_never_switch(monkeypatch):
    from mlx_vlm.tests.test_speculative import generated, inkling2_dspark_case

    mx.random.seed(73)
    target, draft = inkling2_dspark_case(mx.float32)
    spans = []
    real = D._plain_decode_block
    monkeypatch.setattr(
        D, "_plain_decode_block", lambda *a, **k: spans.append(1) or real(*a, **k)
    )
    generated(target, draft, temperature=0, seed=None)  # 24 tokens < WARM
    assert not spans
