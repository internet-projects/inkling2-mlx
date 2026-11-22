import types

import mlx.core as mx

from mlx_vlm.models.inkling2 import language as L


def _cfg(normalize):
    return types.SimpleNamespace(hidden_size=64, attn_res_normalize_values=normalize)


def test_attention_residual_fast_matches_reference():
    mx.random.seed(0)
    for normalize in (True, False):
        for nq in (1, 3):
            for count in (0, 1, 3):
                for rows in ((1, 1), (1, 4)):
                    m = L.AttentionResidual(_cfg(normalize), nq)
                    m.query = mx.random.normal(m.query.shape)
                    blocks = [
                        mx.random.normal((*rows, 64)).astype(mx.bfloat16)
                        for _ in range(count)
                    ]
                    partial = mx.random.normal((*rows, 64))
                    ref = L._attention_residual(
                        mx.stack([*blocks, partial], axis=-2), m.query, normalize
                    )
                    out = m(blocks, partial)
                    assert out.shape == ref.shape
                    assert out.dtype == ref.dtype
                    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()
