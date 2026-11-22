"""inkling2-mlx: the one-token decode path that skips the per-step key/value copies.

``language.FAST_DECODE`` on must give the logits of the stock path (``FAST_DECODE``
off) at every step, including long after the sliding ring has wrapped, on the plain
config and with the loop. The fast path only changes the order in which the keys are
summed, so the comparison allows float32 rounding.
"""

import copy

import mlx.core as mx
import pytest

from mlx_vlm.models.inkling2 import Model, ModelConfig
from mlx_vlm.models.inkling2 import language
from mlx_vlm.tests.test_models import model_case


def _model(loop):
    config = copy.deepcopy(model_case("inkling2")["config"])
    if loop:  # the looped Mini layout of the shared test fixture
        config["model_type"] = "inkling2_loop"
        config["text_config"].update(
            model_type="inkling2_loop_text",
            num_hidden_layers=4,
            n_layers_per_attn_res_block=1,
            loop_config={
                "core_start": 1,
                "core_end": 3,
                "num_passes": 3,
                "gate_rank": 8,
            },
        )
    model = Model(ModelConfig.from_dict(config))
    model.set_dtype(mx.float32)
    mx.random.seed(3)
    params = model.parameters()
    from mlx.utils import tree_flatten, tree_unflatten

    flat = [
        (k, (mx.random.normal(v.shape) * 0.2).astype(v.dtype))
        for k, v in tree_flatten(params)
        if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating)
    ]
    model.update(tree_unflatten(flat))
    for layer in model.language_model.model.layers:
        layer.mlp.e_score_correction_bias = mx.arange(
            layer.mlp.e_score_correction_bias.size, dtype=mx.float32
        )
    model.eval()
    return model


def _decode(model, tokens, steps, fast, monkeypatch):
    monkeypatch.setattr(language, "FAST_DECODE", fast)
    lm = model.language_model
    cache = lm.make_cache()
    out = lm(mx.array([tokens]), cache=cache).logits[:, -1]
    rows = [out]
    nxt = tokens[-1]
    for i in range(steps):
        nxt = (nxt * 7 + 3 + i) % lm.config.unpadded_vocab_size
        out = lm(mx.array([[nxt]]), cache=cache).logits[:, -1]
        rows.append(out)
    mx.eval(rows)
    return mx.concatenate(rows)


@pytest.mark.parametrize("loop", [False, True])
def test_fast_decode_matches_stock(loop, monkeypatch):
    model = _model(loop)
    window = model.language_model.config.sliding_window_size
    steps = 5 * window + 3  # the ring wraps several times
    prompt = [1, 5, 9, 2, 7, 3]
    stock = _decode(model, prompt, steps, False, monkeypatch)
    fast = _decode(model, prompt, steps, True, monkeypatch)
    assert stock.shape == fast.shape
    assert mx.allclose(stock, fast, atol=1e-4, rtol=1e-4).item(), float(
        mx.abs(stock - fast).max()
    )
    assert (mx.argmax(stock, -1) == mx.argmax(fast, -1)).all().item()


def test_fast_decode_is_used_only_for_one_row(monkeypatch):
    """A prompt (several rows) and a batch take the stock path unchanged."""
    model = _model(False)
    lm = model.language_model
    monkeypatch.setattr(language, "FAST_DECODE", True)
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def spy(*args, **kwargs):
        calls.append(kwargs.get("sinks") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    cache = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3]]), cache=cache).logits)
    assert calls and not any(calls)
    calls.clear()
    mx.eval(lm(mx.array([[4]]), cache=cache).logits)
    assert calls and all(calls)


def _decode_rows(model, tokens, rounds, width, fast, monkeypatch, batch=1):
    monkeypatch.setattr(language, "FAST_DECODE", fast)
    lm = model.language_model
    cache = lm.make_cache()
    rows = [lm(mx.array([tokens] * batch), cache=cache).logits]
    nxt = tokens[-1]
    for i in range(rounds):
        block = []
        for j in range(width):
            nxt = (nxt * 7 + 3 + i + j) % lm.config.unpadded_vocab_size
            block.append(nxt)
        rows.append(lm(mx.array([block] * batch), cache=cache).logits)
    mx.eval(rows)
    return mx.concatenate(rows, axis=1)


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_fast_check_rows_match_stock(loop, batch, monkeypatch):
    """Several rows at a time (a drafted check, or a batch): the global layers read
    the sink's zero key from the cache's spare row instead of a copied history."""
    model = _model(loop)
    window = model.language_model.config.sliding_window_size
    prompt = [1, 5, 9, 2, 7, 3]
    rounds = 3 * window + 2
    stock = _decode_rows(model, prompt, rounds, 3, False, monkeypatch, batch)
    fast = _decode_rows(model, prompt, rounds, 3, True, monkeypatch, batch)
    assert mx.allclose(stock, fast, atol=1e-4, rtol=1e-4).item(), float(
        mx.abs(stock - fast).max()
    )


def test_fast_slot_is_used_for_global_rows(monkeypatch):
    """The global layers' attention gets the cache buffer row, not a concatenation."""
    model = _model(False)
    lm = model.language_model
    monkeypatch.setattr(language, "FAST_DECODE", True)
    cache = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3]]), cache=cache).logits)
    from mlx_vlm.speculative.cache_state import iter_leaf_caches

    plain = [c for c in iter_leaf_caches(cache) if type(c) is language.KVCache]
    assert plain and all(c.offset == 3 for c in plain)
    mx.eval(lm(mx.array([[4, 5]]), cache=cache).logits)
    assert all(c.offset == 5 for c in plain)
    for c in plain:  # the spare row after the history holds the zero sink key
        assert mx.all(c.keys[..., c.offset, :] == 0).item()
