"""Inkling 2 Mini with a DFlash2 draft (incoai/Inkling-2-Mini-DFlash2 layout).

Greedy drafted decode must equal plain greedy decode; the selector softcaps
after the edge correction; the draft reads the target's normed embedding and
its raw head over the real vocabulary.
"""

import mlx.core as mx
import pytest

from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config


def _case(dtype=mx.float32, mode="after_correction", block=4):
    from mlx_vlm.tests.test_speculative import generated, inkling2_dspark_case

    mx.random.seed(73)
    target, _ = inkling2_dspark_case(dtype)
    text = target.language_model.config
    visits = len(target.language_model.model.layer_indices)
    config = DFlash2Config.from_dict(
        {
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": text.hidden_size,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "sliding_window": 6,
            "vocab_size": text.vocab_size,
            "max_position_embeddings": 128,
            "is_causal": False,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "dflash_config": {
                "block_size": block,
                "conv_group_size": 8,
                "conv_kernel_size": 2,
                "final_logit_softcapping": 30.0,
                "mask_token_id": 3,
                "selector_rank": 8,
                "selector_softcap_mode": mode,
                "selector_top_k": 4,
                "target_layer_ids": [1, visits],
            },
        }
    )
    draft = DFlash2DraftModel(config)
    draft.set_dtype(dtype)
    draft.eval()
    return target, draft, generated


def test_config_reads_inkling_taps():
    target, draft, _ = _case()
    assert draft.config.num_target_layers == draft.config.target_layer_ids[-1] + 1
    assert draft.config.selector_softcap_mode == "after_correction"
    assert draft.candidate_selector.softcap == 30.0
    draft.bind(target)
    assert draft.argmax_from_hidden is None
    ids = mx.array([[1, 2, 3]])
    lm = target.language_model
    assert mx.allclose(draft._embed_input_tokens(ids), lm.model.embed(ids)).item()


@pytest.mark.parametrize("seed", [41, None])
def test_drafted_greedy_equals_plain(seed):
    target, draft, generated = _case()
    plain = generated(target, None, temperature=0, seed=seed)
    got = generated(target, draft, temperature=0, seed=seed)
    assert got == plain
    assert draft.accept_lens


def test_rejects_other_tap_count():
    target, draft, _ = _case()
    draft.config.target_layer_ids = [1, 2]
    with pytest.raises(ValueError):
        draft.bind(target)


def test_selector_softcap_after_correction():
    target, draft, _ = _case()
    sel = draft.candidate_selector
    logits = mx.zeros((1, 2, draft.config.vocab_size))
    logits[..., 5] = 100.0
    hidden = mx.zeros((1, 2, draft.config.hidden_size))
    path = sel.select(hidden, logits, mx.array([1]), lambda s: mx.argmax(s, -1))
    assert path.tolist() == [[5, 5]]


PUBLISHED = {  # incoai/Inkling-2-Mini-DFlash2 config.json (as downloaded 2026-11-26)
    "architectures": ["DFlash2DraftModel"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": None, "eos_token_id": None, "pad_token_id": None, "dtype": "bfloat16",
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                      "final_logit_softcapping": 30.0, "mask_token_id": 200056, "selector_rank": 256,
                      "selector_softcap_mode": "after_correction", "selector_top_k": 16,
                      "target_layer_ids": [5, 23, 41, 48]},
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 2304, "initializer_range": 0.02,
    "intermediate_size": 6912, "is_causal": False, "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576, "max_window_layers": 5, "model_type": "qwen3",
    "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
    "num_target_layers": 48, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "sliding_window": 2048,
    "tie_word_embeddings": False, "use_cache": False, "use_sliding_window": True, "vocab_size": 201024,
}


def test_published_config_loads():
    config = DFlash2Config.from_dict(dict(PUBLISHED))
    assert config.target_layer_ids == [5, 23, 41, 48]
    assert config.num_target_layers == 49  # 48 visits + the final-norm tap
    assert config.selector_softcap_mode == "after_correction"


@pytest.mark.parametrize("seed", [41, None])
def test_auto_off_switching_keeps_greedy_output(monkeypatch, seed):
    from mlx_vlm.speculative import dflash as D
    from mlx_vlm.tests.test_dspark_auto_off import _Always

    target, draft, generated = _case()
    plain = generated(target, None, temperature=0, seed=seed)
    spans = []
    real = D._plain_decode_block
    monkeypatch.setattr(D, "_DraftAutoOff", _Always)
    monkeypatch.setattr(
        D, "_plain_decode_block", lambda *a, **k: spans.append(a[4]) or real(*a, **k)
    )
    got = generated(target, draft, temperature=0, seed=seed)
    assert draft.supports_auto_off
    assert got == plain
    assert len(spans) >= 2 and draft.accept_lens
