import mlx.core as mx
import pytest

from mlx_vlm.models.inkling2 import language as lang


def _gpu():
    return mx.metal.is_available() and mx.default_device() == mx.gpu


@pytest.mark.parametrize(
    "size,length,roll,sliding",
    [
        (8, 1, 0, 0),
        (40, 1, 0, 0),
        (40, 4, 0, 0),
        (1024, 1, 300, 1024),
        (1024, 4, 0, 1024),
    ],
)
def test_raw_mask_matches_clipped(size, length, roll, sliding):
    if not _gpu():
        pytest.skip("needs Metal")
    mx.random.seed(0)
    heads, kv_heads, head_dim, d_rel = 4, 2, 32, 16
    dt = mx.bfloat16
    q = (mx.random.normal((1, length, heads * head_dim)) * 3).astype(dt)
    k = (mx.random.normal((1, length, kv_heads * head_dim)) * 3).astype(dt)
    v = (mx.random.normal((1, length, kv_heads * head_dim)) * 3).astype(dt)
    rp = (mx.random.normal((1, length, heads * d_rel)) * 4).astype(dt)
    q_scale = mx.random.uniform(0.5, 1.5, (heads, head_dim)).astype(dt)
    rel_scale = mx.random.uniform(0.5, 1.5, (heads, d_rel)).astype(dt)
    post = lang._qkv_norm_fast(
        q,
        k,
        v,
        rp,
        q_scale,
        rel_scale,
        heads,
        kv_heads,
        head_dim,
        d_rel,
        22.6,
        8.0,
        1e-6,
    )
    assert post is not None
    r = post[3].reshape(1, length, heads, d_rel)
    extent = 64
    proj = mx.random.normal((d_rel, extent)).astype(dt)
    kref = mx.zeros((1, kv_heads, size, head_dim), dt)
    ref = lang._row_mask_fast(r, proj, kref, size, sliding, extent, roll)
    out = lang._row_mask_raw(
        rp.reshape(1, length, heads, d_rel),
        rel_scale,
        8.0,
        proj,
        size,
        sliding,
        extent,
        roll,
    )
    assert ref is not None and out is not None
    assert out.shape == ref.shape and out.dtype == ref.dtype
    a, b = out.astype(mx.float32), ref.astype(mx.float32)
    assert mx.array_equal(a <= -1e29, b <= -1e29).item()
    assert mx.allclose(a, b, atol=2e-2, rtol=2e-2).item()


def test_topk_rows_matches_argpartition():
    if not _gpu():
        pytest.skip("needs Metal")
    mx.random.seed(1)
    scores = mx.sigmoid(mx.random.normal((1, 5, 128)))
    bias = mx.random.normal((128,)) * 0.1
    idx, wts = lang._topk_route(scores, bias, lang._route_scale_array(2.5), 8)
    ref = mx.argpartition(-(scores + bias), 7, axis=-1)[..., :8]
    rw = mx.take_along_axis(scores, ref, axis=-1)
    rw = rw / (rw.sum(axis=-1, keepdims=True) + 1e-20) * 2.5
    assert idx.shape == (1, 5, 8) and wts.shape == (1, 5, 8)
    for row in range(5):
        a = sorted(zip(idx[0, row].tolist(), wts[0, row].tolist()))
        b = sorted(zip(ref[0, row].tolist(), rw[0, row].tolist()))
        assert [x for x, _ in a] == [x for x, _ in b]
        for (_, x), (_, y) in zip(a, b):
            assert abs(x - y) <= 1e-5 * max(1.0, abs(y))
