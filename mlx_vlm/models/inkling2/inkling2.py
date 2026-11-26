import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..base import InputEmbeddingsFeatures
from ..inkling.inkling import _split_gate_up, _to_mlx_conv, masked_scatter
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel

# mlx-lm (PR #2086) and nativ-community builds name tensors their own way.
_MLX_LM_RENAMES = [
    (r"^lm_head\.", "language_model.lm_head."),
    (r"^model\.language_model\.", "language_model.model."),
    (r"^model\.vision_tower\.", "vision_tower."),
    (r"\.post_attention_convolution\.conv1d\.", ".attn_sconv.conv."),
    (r"\.post_mlp_convolution\.conv1d\.", ".mlp_sconv.conv."),
    (r"_sconv\.conv1d\.", "_sconv.conv."),
    (r"\.self_attn\.q_l2_clip\.scale$", ".self_attn.q_scale"),
    (r"\.self_attn\.rel_l2_clip\.scale$", ".self_attn.rel_scale"),
    (r"\.self_attn\.rel_logits_proj\.proj$", ".self_attn.rel_proj"),
    (r"\.mlp\.gate\.weight$", ".mlp.gate_weight"),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.e_score_correction_bias"),
    (r"\.q_message_gate\.", ".controllers.attn."),
    (r"\.(k|v|mlp)_message_gate\.", r".controllers.\1."),
    (r"(\.controllers\.\w+)\.down_proj\.weight$", r"\1.down_weight"),
    (r"(\.controllers\.\w+)\.forget_gate_proj\.weight$", r"\1.forget_weight"),
    (r"(\.controllers\.\w+)\.message_gate_proj\.weight$", r"\1.gain_weight"),
    (r"\.model\.message_input\.query$", ".model.message_queries"),
    (r"\.mlp\.experts\.down_proj\.", ".mlp.switch_mlp.down_proj."),
    (r"^vision_tower\.merger\.linear_1\.", "vision_tower.proj_in."),
    (r"^vision_tower\.merger\.linear_2\.", "vision_tower.proj_out."),
    (r"\.embeddings\.position_embedding\.weight$", ".embeddings.position_embedding"),
]
# mlx-lm stores the NVFP4 tensor scale times the E2M1 and E4M3 maxima.
_NVFP4_GLOBAL_TO_SCALE_2 = 1.0 / (6.0 * 448.0)


def _is_mlx_lm_layout(weights):
    return any(
        k.startswith(("model.language_model.", "lm_head."))
        or "_message_gate." in k
        or k.endswith("message_input.query")
        for k in weights
    )


def _from_mlx_lm_layout(weights):
    """Rename an mlx-lm style Inkling 2 checkpoint to this model's names."""
    out = {}
    for key, value in weights.items():
        for pattern, replacement in _MLX_LM_RENAMES:
            key = re.sub(pattern, replacement, key)
        if key.endswith(".global_scale"):
            key = key[: -len("global_scale")] + "weight_scale_2"
            value = value.astype(mx.float32) * _NVFP4_GLOBAL_TO_SCALE_2
        if ".mlp.experts.gate_up_proj." in key:
            base, leaf = key.split(".mlp.experts.gate_up_proj.")
            base += ".mlp.switch_mlp."
            # The rows alternate between gate and up.
            parts = (
                (value, value) if leaf == "weight_scale_2" else _split_gate_up(value)
            )
            out[base + "gate_proj." + leaf], out[base + "up_proj." + leaf] = parts
            continue
        out[key] = value
    return out


def _infer_quantization(path, weight, scales, in_dims):
    bits = weight.shape[-1] * 32 // in_dims
    group_size = in_dims // scales.shape[-1]
    mode = "affine"
    if scales.dtype == mx.uint8:
        mode = {(4, 16): "nvfp4", (4, 32): "mxfp4", (8, 32): "mxfp8"}.get(
            (bits, group_size)
        )
        if mode is None:
            raise ValueError(f"Unknown packed format for {path}: {bits}/{group_size}")
    return {"group_size": group_size, "bits": bits, "mode": mode}


_OMLX_SLIDING_IN_PROJ = ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj")
_OMLX_GLOBAL_Q_PROJ = ("q_proj", "r_proj", "gate_proj", "sink_proj")
_OMLX_LOOP_INPUTS = ("attn", "k", "v", "mlp")
_OMLX_VISION_RENAMES = (
    ("ln_pre.", "pre_layernorm."),
    (".ln_1.", ".layernorm_before."),
    (".ln_2.", ".layernorm_after."),
    (".ls_1.gamma", ".lambda_1"),
    (".ls_2.gamma", ".lambda_2"),
    (".mlp.c_fc.", ".mlp.fc1."),
    (".mlp.c_proj.", ".mlp.fc2."),
)


def _is_omlx_layout(weights):
    return "model.embed.weight" in weights or any(
        k.startswith("model.layers.") and ".attn." in k for k in weights
    )


def _split_rows(tensors, prefix, names, sizes, out):
    """Split stacked output rows of one (quantized) linear into its parts."""
    for suffix in ("weight", "scales", "biases"):
        value = tensors.pop(f"{prefix}.{suffix}", None)
        if value is None:
            continue
        start = 0
        for name, size in zip(names, sizes):
            out[f"{name}.{suffix}"] = value[start : start + size]
            start += size
        if start != value.shape[0]:
            raise ValueError(f"{prefix}: {value.shape[0]} rows, parts sum to {start}")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        config.vision_config.text_hidden_size = config.text_config.hidden_size
        config.vision_config.out_hidden_size = config.text_config.hidden_size
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        h = self.language_model.model.embed(input_ids)
        if pixel_values is not None:
            features = self.vision_tower(
                pixel_values, image_grid_thw=kwargs.get("image_grid_thw")
            ).astype(h.dtype)
            slots = input_ids == self.config.image_token_id
            if int(slots.sum()) != features.shape[0]:
                raise ValueError("Image tokens do not match image features")
            h = masked_scatter(h, mx.broadcast_to(slots[..., None], h.shape), features)
        return InputEmbeddingsFeatures(inputs_embeds=h)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        h = self.get_input_embeddings(input_ids, pixel_values, **kwargs).inputs_embeds
        return self.language_model(inputs_embeds=h, cache=cache, **kwargs)

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group=None):
        self.language_model.shard(group)

    @property
    def layers(self):
        return self.language_model.layers

    def quantization_from_metadata(self, metadata):
        if metadata.get("quant_method") != "inkling2":
            return None
        result = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        for group in metadata["model"]["parameter_groups"]:
            kind = group.get("weight_transform")
            if kind == "cast":
                # Safetensors already stores these groups in their declared
                # FP32/BF16 dtypes; loading must not apply a global BF16 cast.
                # The unembedding's FP32 forward output is handled by the LM.
                continue
            if kind not in ("nvfp4_nearest", "mxfp8_nearest"):
                raise ValueError(f"Unsupported Inkling 2 weight transform: {kind}")
            settings = (
                {"group_size": 16, "bits": 4, "mode": "nvfp4"}
                if kind == "nvfp4_nearest"
                else {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            )
            for tensor in group["tensors"]:
                layer = tensor.split("layers.")[1].split(".")[0]
                projections = (
                    ("gate_proj", "up_proj")
                    if tensor.endswith("w13_weight")
                    else ("down_proj",)
                )
                for projection in projections:
                    result[
                        f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}"
                    ] = settings.copy()
        return result

    def quantization_from_weights(self, weights, quantization):
        """Per-module formats read from tensor shapes, for other builds.

        mlx-lm style configs can understate the expert formats (NVFP4, MXFP8),
        so the shapes are the reference.
        """
        if not getattr(self, "_layout_from_weights", False):
            return None
        modules = dict(self.named_modules())
        base = {
            k: v
            for k, v in (quantization or {}).items()
            if k in ("group_size", "bits", "mode")
        } or {"group_size": 64, "bits": 4, "mode": "affine"}
        result = dict(base)
        for key, scales in weights.items():
            if not key.endswith(".scales"):
                continue
            path = key[: -len(".scales")]
            module = modules.get(path)
            if module is None or not hasattr(module, "weight"):
                continue
            result[path] = _infer_quantization(
                path, weights[path + ".weight"], scales, module.weight.shape[-1]
            )
        return result

    @staticmethod
    def cast_predicate(path):
        # Preserve the native FP32 queries, routing bias, and NVFP4 outer
        # scales during conversion. Router weights may also arrive in FP32.
        return not any(
            s in path
            for s in (
                ".query",
                "message_queries",
                "gate_weight",
                "e_score_correction_bias",
                "weight_scale_2",
            )
        )

    def _from_omlx_layout(self, weights):
        """Map an oMLX (oQ) Inkling 2 checkpoint to this model's names."""
        modules = dict(self.named_modules())
        shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}

        def rows(path):
            return modules[path].weight.shape[0]

        out, layers, messages = {}, {}, {}
        for key, value in weights.items():
            if key.startswith(("mtp.", "model.mtp.")):
                continue
            if key.startswith("model.layers."):
                idx, rest = key[len("model.layers.") :].split(".", 1)
                layers.setdefault(int(idx), {})[rest] = value
            elif key.startswith("vision_tower."):
                sub = key[len("vision_tower.") :].removeprefix("backbone.")
                for old, new in _OMLX_VISION_RENAMES:
                    sub = sub.replace(old, new)
                out["vision_tower." + sub] = value
            elif key.startswith("lm_head."):
                out["language_model." + key] = value
            elif key.startswith("model.embed."):
                out["language_model.model.embed_tokens." + key[12:]] = value
            elif key == "model.final_input.query":
                target = "language_model.model.final_input.query"
                out[target] = value.reshape(shapes[target])
            else:
                out["language_model." + key] = value
        for idx, d in sorted(layers.items()):
            base = f"language_model.model.layers.{idx}."
            a = base + "self_attn."
            if "attn.in_proj.weight" in d:
                names = [a + n for n in _OMLX_SLIDING_IN_PROJ]
                _split_rows(d, "attn.in_proj", names, [rows(n) for n in names], out)
                for n in ("k", "v"):
                    out[a + f"{n}_sconv.conv.weight"] = d.pop(f"attn.{n}_conv")[..., None]
                out[base + "input_layernorm.weight"] = d.pop("attn_norm.weight")
            else:
                names = [a + n for n in _OMLX_GLOBAL_Q_PROJ]
                _split_rows(d, "attn.q_proj", names, [rows(n) for n in names], out)
                for n in ("k", "v"):
                    names = [a + f"{n}_proj", a + f"{n}_sconv.weight_proj"]
                    _split_rows(d, f"attn.{n}_proj", names, [rows(n) for n in names], out)
                    out[a + f"{n}_sconv.static_weight"] = d.pop(f"attn.{n}_static")
                for n in ("q", "k", "v"):
                    out[base + f"{n}_input_layernorm.weight"] = d.pop(
                        f"{n}_input_norm.weight"
                    )
            for suffix in ("weight", "scales", "biases"):
                if f"attn.o_proj.{suffix}" in d:
                    out[a + "o_proj." + suffix] = d.pop(f"attn.o_proj.{suffix}")
            out[a + "q_scale"] = d.pop("attn.q_scale")
            out[a + "rel_scale"] = d.pop("attn.r_scale")
            out[a + "rel_proj"] = d.pop("attn.rel_proj")
            out[base + "attn_sconv.conv.weight"] = d.pop("attn_conv")[..., None]
            out[base + "mlp_sconv.conv.weight"] = d.pop("mlp_conv")[..., None]
            out[base + "post_attention_layernorm.weight"] = d.pop("mlp_norm.weight")
            out[base + "mlp.gate_weight"] = d.pop("mlp.router_weight")
            out[base + "mlp.e_score_correction_bias"] = d.pop("mlp.router_bias")
            for src, dst in (("attn_input", "attention_input"), ("mlp_input", "mlp_input")):
                if f"{src}.query" in d:
                    target = base + f"{dst}.query"
                    out[target] = d.pop(f"{src}.query").reshape(shapes[target])
            sw = base + "mlp.switch_mlp."
            for suffix in ("weight", "scales", "biases"):
                if f"mlp.experts.w13.{suffix}" in d:
                    g, u = _split_gate_up(d.pop(f"mlp.experts.w13.{suffix}"))
                    out[sw + "gate_proj." + suffix], out[sw + "up_proj." + suffix] = g, u
                if f"mlp.experts.w2.{suffix}" in d:
                    out[sw + "down_proj." + suffix] = d.pop(f"mlp.experts.w2.{suffix}")
            for which, names in (("w13", ("gate_proj", "up_proj")), ("w2", ("down_proj",))):
                gs = d.pop(f"mlp.experts.{which}.global_scale", None)
                if gs is not None:
                    for n in names:
                        out[sw + n + ".weight_scale_2"] = (
                            gs.astype(mx.float32) * _NVFP4_GLOBAL_TO_SCALE_2
                        )
            if "loop_down" in d:
                down, gain_forget = d.pop("loop_down"), d.pop("loop_gain_forget")
                width = down.shape[2]
                for i, m in enumerate(_OMLX_LOOP_INPUTS):
                    c = base + f"controllers.{m}."
                    out[c + "down_weight"] = down[i].transpose(2, 0, 1).reshape(
                        down.shape[3], -1
                    )
                    out[c + "gain_weight"] = gain_forget[i][:, :width].T
                    out[c + "forget_weight"] = gain_forget[i][:, width:].T
            if "loop_message.query" in d:
                messages[idx] = d.pop("loop_message.query")
            if d:
                raise ValueError(f"Unmapped oMLX Inkling 2 tensors in layer {idx}: {sorted(d)}")
        if messages:
            out["language_model.model.message_queries"] = mx.stack(
                [messages[i] for i in sorted(messages)]
            )
        return out

    def sanitize(self, weights):
        if _is_omlx_layout(weights):
            self._layout_from_weights = True
            return self._from_omlx_layout(weights)
        self._layout_from_weights = _is_mlx_lm_layout(weights)
        if self._layout_from_weights:
            return _from_mlx_lm_layout(weights)
        out, experts = {}, {}
        attn_names = {
            "wq_du": "q_proj",
            "wk_dv": "k_proj",
            "wv_dv": "v_proj",
            "wr_du": "r_proj",
            "wo_ud": "o_proj",
            "gate_attn": "gate_proj",
            "attn_sink_proj": "sink_proj",
        }
        norms = {
            "attn_norm": "input_layernorm",
            "mlp_norm": "post_attention_layernorm",
            "q_input_norm": "q_input_layernorm",
            "k_input_norm": "k_input_layernorm",
            "v_input_norm": "v_input_layernorm",
        }
        for key, value in weights.items():
            if key.startswith("model.llm.layers."):
                layer, sub = key[len("model.llm.layers.") :].split(".", 1)
                base = f"language_model.model.layers.{layer}."
                if sub.startswith("mlp.experts."):
                    experts.setdefault(layer, {})[sub[len("mlp.experts.") :]] = value
                    continue
                if sub.startswith("attn."):
                    sub = sub[len("attn.") :]
                    name, _, leaf = sub.partition(".")
                    sub = attn_names.get(name, name) + ("." + leaf if leaf else "")
                    sub = {
                        "q_l2_clip.scale": "q_scale",
                        "rel_l2_clip.scale": "rel_scale",
                        "rel_logits_proj.proj": "rel_proj",
                    }.get(sub, sub)
                    for conv in ("k_sconv", "v_sconv"):
                        if sub == conv + ".weight":
                            sub = conv + ".conv.weight"
                            value = _to_mlx_conv(value)
                        elif sub == conv + ".static_w":
                            sub = conv + ".static_weight"
                        elif sub == conv + ".weight_proj":
                            sub += ".weight"
                    sub = "self_attn." + sub
                elif sub in ("attn_sconv.weight", "mlp_sconv.weight"):
                    sub = sub.replace(".weight", ".conv.weight")
                    value = _to_mlx_conv(value)
                elif sub in ("mlp.gate.weight", "mlp.gate.bias"):
                    sub = (
                        "mlp.gate_weight"
                        if sub.endswith("weight")
                        else "mlp.e_score_correction_bias"
                    )
                else:
                    name, _, leaf = sub.partition(".")
                    sub = norms.get(name, name) + "." + leaf
                out[base + sub] = value
            elif key.startswith("model.llm.attn_res."):
                sub = key[len("model.llm.attn_res.") :]
                if sub == "output_query":
                    target = "final_input.query"
                else:
                    kind, index = sub.split(".")
                    layer = int(index) + (kind == "attn_queries")
                    target = f"layers.{layer}.{'attention_input' if kind == 'attn_queries' else 'mlp_input'}.query"
                out["language_model.model." + target] = value
            elif key.startswith("model.llm."):
                sub = key[len("model.llm.") :]
                target = (
                    "language_model.lm_head.weight"
                    if sub == "unembed.weight"
                    else "language_model.model."
                    + sub.replace("embed.weight", "embed_tokens.weight")
                )
                out[target] = value
            elif key.startswith("model.visual."):
                sub = key[len("model.visual.") :].removeprefix("backbone.")
                sub = (
                    sub.replace("ln_pre.", "pre_layernorm.")
                    .replace(".ln_1.", ".layernorm_before.")
                    .replace(".ln_2.", ".layernorm_after.")
                )
                sub = (
                    sub.replace(".ls_1.gamma", ".lambda_1")
                    .replace(".ls_2.gamma", ".lambda_2")
                    .replace(".mlp.c_fc.", ".mlp.fc1.")
                    .replace(".mlp.c_proj.", ".mlp.fc2.")
                )
                out["vision_tower." + sub] = value
            elif key.startswith("model.mtp.") or key.startswith("mtp."):
                continue
            else:
                out[key] = value
        for layer, tensors in experts.items():
            base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
            for name, projections in (
                ("w13_weight", ("gate_proj", "up_proj")),
                ("w2_weight", ("down_proj",)),
            ):
                weight = tensors[name]
                packed = name + "_scale" in tensors
                values = {"weight": weight.view(mx.uint32) if packed else weight}
                if packed:
                    values["scales"] = tensors[name + "_scale"]
                if name + "_amax" in tensors:
                    values["weight_scale_2"] = (
                        tensors[name + "_amax"]
                        / tensors[name + "_outer_scale_denominator"]
                    )
                for suffix, v in values.items():
                    parts = (
                        _split_gate_up(v)
                        if len(projections) == 2 and suffix != "weight_scale_2"
                        else (v,) * len(projections)
                    )
                    for projection, part in zip(projections, parts):
                        out[base + projection + "." + suffix] = part
        base = "language_model.model."
        if base + "middle_cycle_cross_pass_queries" in out:
            width = self.config.text_config.hidden_size
            attn_mlp = out.pop(base + "middle_cycle_cross_pass_queries").reshape(
                -1, 2, width
            )
            kv = out.pop(base + "middle_cycle_cross_pass_kv_queries").reshape(
                -1, 2, width
            )
            out[base + "message_queries"] = mx.concatenate(
                [attn_mlp[:, :1], kv, attn_mlp[:, 1:]], axis=1
            )
        return out
