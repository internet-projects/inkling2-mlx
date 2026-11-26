"""Inkling 2 Mini builds made for mlx-lm (PR #2086) load and give the same output.

These builds use mlx-lm tensor names, fuse gate/up with alternating rows, store
an NVFP4 tensor scale as `global_scale` and can understate expert formats in
config.json.
"""

import json
import re

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.tests.test_models import model_case
from mlx_vlm.utils import load_model

_TO_MLX_LM = [
    (r"^language_model\.lm_head\.", "lm_head."),
    (r"^language_model\.model\.", "model.language_model."),
    (r"^vision_tower\.proj_in\.", "model.vision_tower.merger.linear_1."),
    (r"^vision_tower\.proj_out\.", "model.vision_tower.merger.linear_2."),
    (r"^vision_tower\.", "model.vision_tower."),
    (r"\.embeddings\.position_embedding$", ".embeddings.position_embedding.weight"),
    (r"\.attn_sconv\.conv\.", ".post_attention_convolution.conv1d."),
    (r"\.mlp_sconv\.conv\.", ".post_mlp_convolution.conv1d."),
    (r"_sconv\.conv\.", "_sconv.conv1d."),
    (r"\.self_attn\.q_scale$", ".self_attn.q_l2_clip.scale"),
    (r"\.self_attn\.rel_scale$", ".self_attn.rel_l2_clip.scale"),
    (r"\.self_attn\.rel_proj$", ".self_attn.rel_logits_proj.proj"),
    (r"\.mlp\.gate_weight$", ".mlp.gate.weight"),
    (r"\.mlp\.e_score_correction_bias$", ".mlp.gate.e_score_correction_bias"),
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.(k|v|mlp)\.", r".\1_message_gate."),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"\.model\.message_queries$", ".model.message_input.query"),
    (r"\.mlp\.switch_mlp\.down_proj\.", ".mlp.experts.down_proj."),
]


def _mini(quantize):
    from mlx_vlm.models.inkling2 import Model, ModelConfig

    mx.random.seed(11)
    config = model_case("inkling2")["config"]
    config.update(model_type="inkling2_loop")
    config["text_config"].update(
        model_type="inkling2_loop_text",
        num_hidden_layers=4,
        n_layers_per_attn_res_block=1,
        loop_config={"core_start": 1, "core_end": 3, "num_passes": 3, "gate_rank": 8},
    )
    model = Model(ModelConfig.from_dict(config))
    noisy = (".query", ".rel_proj", "gate_weight", "message_queries", ".controllers.")
    model.load_weights(
        [
            (p, mx.random.normal(v.shape) * 0.1)
            for p, v in tree_flatten(model.parameters())
            if any(n in p for n in noisy)
        ],
        strict=False,
    )
    if quantize:
        # Experts NVFP4 (first two layers) and MXFP8, attention 4-bit affine.
        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".switch_mlp." in path:
                layer = int(path.split(".layers.")[1].split(".")[0])
                if layer < 2:
                    return {"group_size": 16, "bits": 4, "mode": "nvfp4"}
                return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            if path.startswith("language_model.") and path.endswith(
                ("q_proj", "k_proj", "o_proj", "lm_head", "embed_tokens")
            ):
                return {"group_size": 32, "bits": 4, "mode": "affine"}
            return False

        nn.quantize(model, class_predicate=predicate)
    model.eval()
    return model, config


def _to_mlx_lm(weights):
    out, fused = {}, {}
    for key, value in weights.items():
        m = re.search(r"\.mlp\.switch_mlp\.(gate|up)_proj\.(\w+)$", key)
        if m:
            base = key[: m.start()] + ".mlp.experts.gate_up_proj." + m.group(2)
            fused.setdefault(base, {})[m.group(1)] = value
            continue
        for pattern, replacement in _TO_MLX_LM:
            key = re.sub(pattern, replacement, key)
        out[key] = value
    for key, parts in fused.items():
        g, u = parts["gate"], parts["up"]
        out[key] = mx.stack([g, u], axis=-2).reshape(*g.shape[:-2], -1, g.shape[-1])
    return out


@pytest.mark.parametrize("quantize", [False, True])
def test_mlx_lm_layout_loads_same_output(tmp_path, quantize):
    model, config = _mini(quantize)
    weights = dict(tree_flatten(model.parameters()))
    foreign = _to_mlx_lm(weights)
    assert any(k.startswith("model.language_model.") for k in foreign)
    if quantize:
        # mlx-lm configs can say affine for everything.
        config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), foreign)

    loaded = load_model(tmp_path)
    got = dict(tree_flatten(loaded.parameters()))
    assert set(got) == set(weights)
    for key, value in weights.items():
        assert got[key].dtype == value.dtype, key
        assert mx.array_equal(got[key], value).item(), key
    ids = mx.array([[1, 5, 9, 2, 7, 3]])
    assert mx.array_equal(
        loaded.language_model(ids).logits, model.language_model(ids).logits
    ).item()


def test_global_scale_maps_to_weight_scale_2():
    from mlx_vlm.models.inkling2.inkling2 import _from_mlx_lm_layout

    w = mx.zeros((2, 4, 2), dtype=mx.uint32)
    out = _from_mlx_lm_layout(
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj.weight": w,
            "model.language_model.layers.0.mlp.experts.gate_up_proj.global_scale": mx.array(
                [2688.0, 5376.0]
            ),
        }
    )
    base = "language_model.model.layers.0.mlp.switch_mlp."
    assert out[base + "gate_proj.weight"].shape == (2, 2, 2)
    for p in ("gate_proj", "up_proj"):
        assert out[base + p + ".weight_scale_2"].tolist() == [1.0, 2.0]


def test_own_layout_untouched():
    from mlx_vlm.models.inkling2.inkling2 import _is_mlx_lm_layout

    assert not _is_mlx_lm_layout(
        {"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": 0}
    )
    assert _is_mlx_lm_layout(
        {"language_model.model.layers.1.q_message_gate.down_proj.weight": 0}
    )


def _to_omlx(weights, sliding):
    """oMLX (oQ) layout: stacked attention rows, packed loop controllers."""
    out, layers = {}, {}
    for key, value in weights.items():
        if key.startswith("language_model.model.layers."):
            idx, rest = key[len("language_model.model.layers.") :].split(".", 1)
            layers.setdefault(int(idx), {})[rest] = value
        elif key == "language_model.model.message_queries":
            queries = value
        elif key.startswith("language_model.model.embed_tokens."):
            out["model.embed." + key.rsplit(".", 1)[1]] = value
        elif key == "language_model.model.final_input.query":
            out["model.final_input.query"] = value.reshape(-1, value.shape[-1])
        elif key.startswith("language_model."):
            out[key[len("language_model.") :]] = value
        elif key.startswith("vision_tower."):
            sub = key[len("vision_tower.") :]
            for new, old in (
                ("ln_pre.", "pre_layernorm."),
                (".ln_1.", ".layernorm_before."),
                (".ln_2.", ".layernorm_after."),
                (".ls_1.gamma", ".lambda_1"),
                (".ls_2.gamma", ".lambda_2"),
                (".mlp.c_fc.", ".mlp.fc1."),
                (".mlp.c_proj.", ".mlp.fc2."),
            ):
                sub = sub.replace(old, new)
            if not sub.startswith(("proj_in", "proj_out")):
                sub = "backbone." + sub
            out["vision_tower." + sub] = value
    core = []
    for idx, d in sorted(layers.items()):
        pre = f"model.layers.{idx}."
        a = "self_attn."

        def stack(dst, names):
            for s in ("weight", "scales", "biases"):
                parts = [d.pop(f"{a}{n}.{s}") for n in names if f"{a}{n}.{s}" in d]
                if parts:
                    out[f"{pre}attn.{dst}.{s}"] = mx.concatenate(parts, axis=0)

        if idx in sliding:
            stack("in_proj", ["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"])
            for n in ("k", "v"):
                out[pre + f"attn.{n}_conv"] = d.pop(f"{a}{n}_sconv.conv.weight")[..., 0]
            out[pre + "attn_norm.weight"] = d.pop("input_layernorm.weight")
        else:
            stack("q_proj", ["q_proj", "r_proj", "gate_proj", "sink_proj"])
            for n in ("k", "v"):
                stack(f"{n}_proj", [f"{n}_proj", f"{n}_sconv.weight_proj"])
                out[pre + f"attn.{n}_static"] = d.pop(f"{a}{n}_sconv.static_weight")
            for n in ("q", "k", "v"):
                out[pre + f"{n}_input_norm.weight"] = d.pop(f"{n}_input_layernorm.weight")
        stack("o_proj", ["o_proj"])
        out[pre + "attn.q_scale"] = d.pop(a + "q_scale")
        out[pre + "attn.r_scale"] = d.pop(a + "rel_scale")
        out[pre + "attn.rel_proj"] = d.pop(a + "rel_proj")
        out[pre + "attn_conv"] = d.pop("attn_sconv.conv.weight")[..., 0]
        out[pre + "mlp_conv"] = d.pop("mlp_sconv.conv.weight")[..., 0]
        out[pre + "mlp_norm.weight"] = d.pop("post_attention_layernorm.weight")
        out[pre + "mlp.router_weight"] = d.pop("mlp.gate_weight")
        out[pre + "mlp.router_bias"] = d.pop("mlp.e_score_correction_bias")
        for src, dst in (("attention_input", "attn_input"), ("mlp_input", "mlp_input")):
            if f"{src}.query" in d:
                q = d.pop(f"{src}.query")
                out[pre + f"{dst}.query"] = q.reshape(-1, q.shape[-1])
        sw = "mlp.switch_mlp."
        for s in ("weight", "scales", "biases"):
            if f"{sw}gate_proj.{s}" in d:
                g, u = d.pop(f"{sw}gate_proj.{s}"), d.pop(f"{sw}up_proj.{s}")
                out[pre + "mlp.experts.w13." + s] = mx.stack([g, u], axis=-2).reshape(
                    *g.shape[:-2], -1, g.shape[-1]
                )
                out[pre + "mlp.experts.w2." + s] = d.pop(f"{sw}down_proj.{s}")
        if "controllers.attn.down_weight" in d:
            downs, gates = [], []
            for m in ("attn", "k", "v", "mlp"):
                down = d.pop(f"controllers.{m}.down_weight")
                rank = down.shape[0]
                downs.append(down.reshape(rank, 2, -1).transpose(1, 2, 0))
                gain = d.pop(f"controllers.{m}.gain_weight")
                forget = d.pop(f"controllers.{m}.forget_weight")
                gates.append(mx.concatenate([gain.T, forget.T], axis=1))
            out[pre + "loop_down"] = mx.stack(downs)
            out[pre + "loop_gain_forget"] = mx.stack(gates)
            core.append(idx)
        assert not d, sorted(d)
    for i, idx in enumerate(core):
        out[f"model.layers.{idx}.loop_message.query"] = queries[i]
    return out


@pytest.mark.parametrize("quantize", [False, True])
def test_omlx_layout_loads_same_output(tmp_path, quantize):
    model, config = _mini(False)
    if quantize:
        # oQ: experts NVFP4 / MXFP8, attention and head 8-bit affine.
        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".switch_mlp." in path:
                layer = int(path.split(".layers.")[1].split(".")[0])
                if layer < 2:
                    return {"group_size": 16, "bits": 4, "mode": "nvfp4"}
                return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            if path.startswith("language_model.") and (
                ".self_attn." in path or path.endswith(("lm_head", "embed_tokens"))
            ):
                return {"group_size": 32, "bits": 8, "mode": "affine"}
            return False

        nn.quantize(model, class_predicate=predicate)
    weights = dict(tree_flatten(model.parameters()))
    sliding = set(config["text_config"]["local_layer_ids"])
    foreign = _to_omlx(weights, sliding)
    assert "model.embed.weight" in foreign
    config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), foreign)

    loaded = load_model(tmp_path)
    got = dict(tree_flatten(loaded.parameters()))
    assert set(got) == set(weights)
    for key, value in weights.items():
        assert mx.array_equal(got[key], value).item(), key
    ids = mx.array([[1, 5, 9, 2, 7, 3]])
    assert mx.array_equal(
        loaded.language_model(ids).logits, model.language_model(ids).logits
    ).item()
