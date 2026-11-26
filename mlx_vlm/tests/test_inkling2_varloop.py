"""inkling2-mlx: variable-loop Inkling 2 checkpoints (loop_config.is_varloop).

TML-d24-varloop-research has the Inkling 2 Mini layout, but its core layers can run
any number of passes at inference (loop_num_passes, or INKLING2_LOOP_PASSES). The
logits were checked against Transformers on a random tiny model at 1-5 passes
(max |diff| 3.6e-6, full and cached); these tests pin the config rules and the
pass count the decoder actually runs.
"""

import copy

import mlx.core as mx
import pytest

from mlx_vlm.models.inkling2 import Model, ModelConfig
from mlx_vlm.models.inkling2.config import TextConfig
from mlx_vlm.speculative.drafters.inkling2_dspark.config import (
    ModelConfig as DraftConfig,
)
from mlx_vlm.tests.test_models import model_case

LOOP = {"core_start": 1, "core_end": 3, "num_passes": 3, "gate_rank": 8}


def _config(varloop=True, **text):
    config = copy.deepcopy(model_case("inkling2")["config"])
    config["model_type"] = "inkling2_loop"
    config["text_config"].update(
        model_type="inkling2_loop_text",
        num_hidden_layers=4,
        n_layers_per_attn_res_block=1,
        loop_config={**LOOP, "is_varloop": varloop},
        **text,
    )
    return config


def _text(config):
    return TextConfig.from_dict(config["text_config"])


def test_varloop_defaults_to_checkpoint_passes(monkeypatch):
    monkeypatch.delenv("INKLING2_LOOP_PASSES", raising=False)
    text = _text(_config())
    assert text.num_loop_passes == 3
    assert text.layer_indices == (0, 1, 2, 1, 2, 1, 2, 3)


@pytest.mark.parametrize("passes", [1, 2, 5])
def test_varloop_pass_count_from_config_and_env(monkeypatch, passes):
    monkeypatch.delenv("INKLING2_LOOP_PASSES", raising=False)
    text = _text(_config(loop_num_passes=passes))
    assert text.layer_indices == (0,) + (1, 2) * passes + (3,)
    monkeypatch.setenv("INKLING2_LOOP_PASSES", str(passes))
    assert _text(_config()).num_loop_passes == passes


def test_pass_override_needs_varloop(monkeypatch):
    monkeypatch.delenv("INKLING2_LOOP_PASSES", raising=False)
    with pytest.raises(ValueError, match="variable-loop"):
        _text(_config(varloop=False, loop_num_passes=2))
    # Transformers saves loop_num_passes equal to the checkpoint's count: accepted.
    assert _text(_config(varloop=False, loop_num_passes=3)).num_loop_passes == 3
    # The environment variable leaves fixed-loop checkpoints alone.
    monkeypatch.setenv("INKLING2_LOOP_PASSES", "1")
    assert _text(_config(varloop=False)).num_loop_passes == 3


@pytest.mark.parametrize("bad", ["0", "x"])
def test_bad_pass_count_is_rejected(monkeypatch, bad):
    monkeypatch.setenv("INKLING2_LOOP_PASSES", bad)
    with pytest.raises(ValueError):
        _text(_config())


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_varloop_model_runs_each_pass_count(monkeypatch, passes):
    monkeypatch.setenv("INKLING2_LOOP_PASSES", str(passes))
    model = Model(ModelConfig.from_dict(_config()))
    lm = model.language_model
    cache = lm.make_cache()
    assert len(cache) == 2 + 2 * passes
    out = lm(mx.array([[1, 2, 3, 4]]), cache=cache).logits
    out = lm(mx.array([[5]]), cache=cache).logits
    assert out.shape[1] == 1 and bool(mx.isfinite(out).all())


def _draft(passes, monkeypatch):
    monkeypatch.setenv("INKLING2_LOOP_PASSES", str(passes))
    text = _config()["text_config"]
    text["mtp_config"] = {
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 1024,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "local_layer_ids": [0],
        "sliding_window_size": 4,
        "block_size": 4,
        "markov_rank": 4,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 8],
    }
    return DraftConfig.from_dict({"text_config": text})


def test_drafter_only_at_trained_pass_count(monkeypatch):
    assert _draft(3, monkeypatch).target_layer_ids == [1, 8]
    with pytest.raises(ValueError, match="trained with 3 loop passes"):
        _draft(2, monkeypatch)
