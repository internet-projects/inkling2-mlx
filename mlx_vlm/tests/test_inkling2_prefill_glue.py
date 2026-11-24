"""inkling2-mlx: prefill glue. The message controller as one compiled graph and the
depth reads of a whole chunk through the per-row attention-residual kernel give the
results of the plain ops (the kernel part runs only where Metal is available)."""

import types

import mlx.core as mx
import pytest

from mlx_vlm.models.inkling2 import language as L
from mlx_vlm.tests.test_inkling2_fast_decode import _model


def test_message_controller_compiled_matches(monkeypatch):
    mx.random.seed(1)
    cfg = types.SimpleNamespace(
        hidden_size=64,
        rms_norm_eps=1e-6,
        loop_config=types.SimpleNamespace(gate_rank=8),
    )
    m = L.MessageController(cfg)
    m.down_weight = mx.random.normal(m.down_weight.shape).astype(mx.bfloat16)
    m.forget_weight = mx.random.normal(m.forget_weight.shape) * 0.1
    m.gain_weight = mx.random.normal(m.gain_weight.shape) * 0.1
    x = mx.random.normal((1, 40, 64)).astype(mx.bfloat16)
    msg = mx.random.normal((1, 40, 64))
    monkeypatch.setattr(L, "PREFILL_CTRL_COMPILE", False)
    ref = m(x, msg)
    monkeypatch.setattr(L, "PREFILL_CTRL_COMPILE", True)
    out = m(x, msg)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="kernel needs Metal")
def test_attention_residual_rows_kernel_matches():
    mx.random.seed(2)
    for normalize in (True, False):
        blocks = [mx.random.normal((1, 300, 64)).astype(mx.bfloat16) for _ in range(3)]
        partial = mx.random.normal((1, 300, 64))
        query = mx.random.normal((64,))
        ref = L._attention_residual(
            mx.stack([*blocks, partial], axis=-2), query, normalize
        )
        out = L._attention_residual_fast(blocks, partial, query, normalize)
        assert out is not None
        assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("loop", [False, True])
def test_prefill_glue_same_logits(loop, monkeypatch):
    model = _model(loop)
    lm = model.language_model
    tokens = [(i * 7 + 5) % 50 + 1 for i in range(40)]

    def run(on):
        monkeypatch.setattr(L, "PREFILL_CTRL_COMPILE", on)
        monkeypatch.setattr(L, "PREFILL_ATTN_RES", on)
        monkeypatch.setattr(L, "PREFILL_MASK_SINK", on)
        cache = lm.make_cache()
        out = lm(mx.array([tokens]), cache=cache).logits
        nxt = lm(mx.array([[3]]), cache=cache).logits
        mx.eval(out, nxt)
        return mx.concatenate([out[0], nxt[0]])

    a, b = run(False), run(True)
    assert mx.allclose(a, b, atol=1e-4, rtol=1e-4).item(), float(mx.abs(a - b).max())


@pytest.mark.skipif(not mx.metal.is_available(), reason="kernel needs Metal")
def test_mask_with_sink_matches_concat():
    from mlx_vlm.models.inkling.language import banded_additive_mask

    mx.random.seed(4)
    B, LQ, H, d_rel, S, extent = 1, 40, 6, 16, 70, 32
    rel = mx.random.normal((B, LQ, H, d_rel)).astype(mx.bfloat16)
    proj = mx.random.normal((d_rel, extent)).astype(mx.bfloat16)
    kref = mx.zeros((B, 2, S, 8), mx.bfloat16)
    sinks = mx.random.normal((B, LQ, H)).astype(mx.bfloat16)
    for sliding in (0, 24):
        ref = banded_additive_mask(
            rel, proj, S - LQ, S, sliding, extent, shape_ref=kref
        )
        ref = mx.concatenate([ref, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
        out = L._mask_with_sink(rel, proj, kref, sliding, extent, sinks)
        assert out is not None and out.shape == ref.shape
        assert mx.array_equal(out, ref).item()
