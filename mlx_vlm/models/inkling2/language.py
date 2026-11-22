# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for MLX from Transformers' Inkling2 implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import (
    ArraysCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from ..inkling.language import InklingShortConvolution
from ..inkling.language import LanguageModel as InklingLanguageModel
from ..inkling.language import (
    _cache_padding_mask,
    _next_conv_state,
    banded_additive_mask,
)
from ..switch_layers import SwitchGLU
from .config import TextConfig, validate_sharding

# One-token decode without per-step copies of the key/value history (inkling2-mlx).
# Off: the attention below runs exactly as upstream. On: a one-row step of a single
# conversation passes the query-dependent sink to the fused attention as a per-head
# sink (one row, so per query and per head coincide) instead of appending a zero key
# and value to the whole history, and a full sliding ring is read in its physical
# order with the relative-bias mask rolled to match, instead of copying the ring
# into chronological order. Same arithmetic; the keys are summed in another order,
# so results can differ in the last bits.
FAST_DECODE = os.environ.get("INKLING2_FAST_DECODE", "1").strip().lower() not in (
    "0",
    "off",
    "false",
    "no",
)
# Parts of the fast path, for diagnosis only (each needs FAST_DECODE on).
FAST_SINK = FAST_ROLL = FAST_GATE = FAST_SLOT = FAST_CTRL = True
# Compiled elementwise glue (output gate, residual adds, dynamic conv, log scaling).
FAST_GLUE = True
# One Metal kernel per attention-residual mix for short (decode/check) rows.
FAST_ATTN_RES = True
# One Metal kernel for the q/k/r l2 clips and the value RMS norm (short rows).
FAST_QKV_NORM = True
# Cross-pass messages of the looped core through the attention-residual kernel.
FAST_MESSAGES = True
# Banded relative mask for short rows: 1-D threadgroups, sliding-ring roll folded in.
FAST_MASK = True
# attn_sconv / mlp_sconv write the decoder residual sum directly (short rows).
FAST_SCONV_RES = True

_ROW_MASK_SRC = r"""
    const int B  = rel_shape[0];
    const int LQ = rel_shape[1];
    const int H  = rel_shape[2];
    const int S  = kshape_shape[2];
    uint p  = thread_position_in_grid.x;
    uint i  = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;
    if ((int)i >= LQ || (int)p >= S || (int)bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int j = (int)p - shift[0];
    if (j < 0) j += S;
    int dist = ((int)i + (S - LQ)) - j;
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0]
            + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0]
                               + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[(((size_t)b * H + h) * LQ + i) * S + p] = val;
"""
_ROW_MASK_KERNEL = []
_ROW_MASK_SHIFTS = {}


def _row_mask_fast(rel, proj, kref, size, sliding, rel_extent, roll):
    """Banded additive mask [B, H, LQ, S] for a few query rows, with physical
    slot p holding chronological key (p - roll) mod S. Same per-element
    arithmetic as the v2 mask kernel; None when not applicable."""
    if (
        rel.ndim != 4
        or rel.shape[1] > 8
        or kref.ndim < 3
        or kref.shape[2] != size
        or size < 1
        or rel.dtype not in _ATTN_RES_DTYPES
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    if not _ROW_MASK_KERNEL:
        _ROW_MASK_KERNEL.append(
            mx.fast.metal_kernel(
                name="inkling2_banded_mask_rows",
                input_names=["rel", "proj", "kshape", "shift"],
                output_names=["out"],
                source=_ROW_MASK_SRC,
                ensure_row_contiguous=False,
            )
        )
    roll = int(roll) % size
    shift = _ROW_MASK_SHIFTS.get(roll)
    if shift is None:
        if len(_ROW_MASK_SHIFTS) > 8192:
            _ROW_MASK_SHIFTS.clear()
        shift = mx.array([roll], mx.int32)
        _ROW_MASK_SHIFTS[roll] = shift
    b, lq, h, d_rel = rel.shape
    tg = min(256, size)
    return _ROW_MASK_KERNEL[0](
        inputs=[rel, proj, kref, shift],
        template=[
            ("T", rel.dtype),
            ("D_REL", d_rel),
            ("REL_EXTENT", int(rel_extent)),
            ("SLIDING", int(sliding)),
        ],
        grid=(((size + tg - 1) // tg) * tg, lq, b * h),
        threadgroup=(tg, 1, 1),
        output_shapes=[(b, h, lq, size)],
        output_dtypes=[rel.dtype],
    )[0]


def _message_bank(states):
    return mx.fast.rms_norm(
        mx.stack(states, axis=-2).astype(mx.float32), None, 1e-6
    )

_ATTN_RES_TG = 512
_ATTN_RES_SRC = r"""
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    constexpr uint NS = TG / 32;
    constexpr uint NB = @NB@;
    threadgroup float red[NS * NB * NQ];
    threadgroup float tot[NB * NQ];
    size_t base = (size_t)row * D;
    float ss[NB];
    for (uint k = 0; k < NB; ++k) ss[k] = 0.0f;
    for (uint d = tid; d < D; d += TG) {
@SUMSQ@
    }
    for (uint k = 0; k < NB; ++k) {
        float s = simd_sum(ss[k]);
        if (lane == 0) red[sg * NB + k] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NB) {
        float s = 0.0f;
        for (uint g = 0; g < NS; ++g) s += red[g * NB + tid];
        tot[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv[NB];
    for (uint k = 0; k < NB; ++k)
        inv[k] = metal::precise::rsqrt(tot[k] / (float)D + 1e-6f);
    float dots[NB * NQ];
    for (uint j = 0; j < NB * NQ; ++j) dots[j] = 0.0f;
    for (uint d = tid; d < D; d += TG) {
        float qv[NQ];
        for (uint q = 0; q < NQ; ++q) qv[q] = float(query[q * D + d]);
@DOTS@
    }
    for (uint j = 0; j < NB * NQ; ++j) {
        float s = simd_sum(dots[j]);
        if (lane == 0) red[sg * NB * NQ + j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NB * NQ) {
        float s = 0.0f;
        for (uint g = 0; g < NS; ++g) s += red[g * NB * NQ + tid];
        tot[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = metal::precise::rsqrt((float)D);
    float wts[NB * NQ];
    for (uint q = 0; q < NQ; ++q) {
        float m = -INFINITY;
        for (uint k = 0; k < NB; ++k) {
            wts[k * NQ + q] = tot[k * NQ + q] * scale;
            m = metal::max(m, wts[k * NQ + q]);
        }
        float sum = 0.0f;
        for (uint k = 0; k < NB; ++k) {
            wts[k * NQ + q] = metal::precise::exp(wts[k * NQ + q] - m);
            sum += wts[k * NQ + q];
        }
        for (uint k = 0; k < NB; ++k) wts[k * NQ + q] = wts[k * NQ + q] / sum;
    }
    for (uint d = tid; d < D; d += TG) {
        float vals[NB];
@VALUES@
        for (uint q = 0; q < NQ; ++q) {
            float o = 0.0f;
            for (uint k = 0; k < NB; ++k) o += wts[k * NQ + q] * vals[k];
            out[((size_t)row * NQ + q) * D + d] = o;
        }
    }
"""
_ATTN_RES_KERNELS = {}


def _attn_res_kernel(count):
    kernel = _ATTN_RES_KERNELS.get(count)
    if kernel is None:
        sumsq = "\n".join(
            f"        {{ float v = float(x{k}[base + d]); ss[{k}] += v * v; }}"
            for k in range(count)
        )
        dots = "\n".join(
            f"        {{ float key = float(x{k}[base + d]) * inv[{k}];"
            f" for (uint q = 0; q < NQ; ++q) dots[{k} * NQ + q] += key * qv[q]; }}"
            for k in range(count)
        )
        values = "\n".join(
            f"        {{ float v = float(x{k}[base + d]);"
            f" vals[{k}] = NORMALIZE ? v * inv[{k}] : v; }}"
            for k in range(count)
        )
        source = (
            _ATTN_RES_SRC.replace("@NB@", str(count))
            .replace("@SUMSQ@", sumsq)
            .replace("@DOTS@", dots)
            .replace("@VALUES@", values)
        )
        kernel = mx.fast.metal_kernel(
            name=f"inkling2_attention_residual_{count}",
            input_names=[*(f"x{k}" for k in range(count)), "query"],
            output_names=["out"],
            source=source,
        )
        _ATTN_RES_KERNELS[count] = kernel
    return kernel


_ATTN_RES_DTYPES = (mx.float32, mx.bfloat16, mx.float16)


def _attention_residual_fast(blocks, partial, query, normalize_values):
    """Fused attention residual for a few rows; None when not applicable."""
    inputs = [*blocks, partial]
    shape = partial.shape
    if (
        len(shape) != 3
        or shape[0] * shape[1] > 8
        or len(inputs) > 8
        or query.ndim not in (1, 2)
        or query.shape[-1] != shape[-1]
        or query.dtype not in _ATTN_RES_DTYPES
        or any(a.shape != shape or a.dtype not in _ATTN_RES_DTYPES for a in inputs)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    width = shape[-1]
    nq = 1 if query.ndim == 1 else query.shape[0]
    if nq < 1 or nq > 4:
        return None
    rows = shape[0] * shape[1]
    return _attn_res_kernel(len(inputs))(
        inputs=[*inputs, query],
        template=[
            ("TG", _ATTN_RES_TG),
            ("NQ", nq),
            ("D", width),
            ("NORMALIZE", bool(normalize_values)),
        ],
        grid=(_ATTN_RES_TG, rows, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(shape[0], shape[1], nq, width)],
        output_dtypes=[mx.float32],
    )[0]


def _fast_glue(x):
    return FAST_DECODE and FAST_GLUE and x.shape[1] <= 8


_QKV_NORM_SRC = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint slot = threadgroup_position_in_grid.y;
    uint row = threadgroup_position_in_grid.z;
    const device T* src;
    device T* dst;
    uint width;
    uint head;
    int kind;
    if (slot < HQ) {
        kind = 0; head = slot; width = DH;
        src = q + ((size_t)row * HQ + head) * DH;
        dst = q_out + ((size_t)row * HQ + head) * DH;
    } else if (slot < HQ + HKV) {
        kind = 1; head = slot - HQ; width = DH;
        src = k + ((size_t)row * HKV + head) * DH;
        dst = k_out + ((size_t)row * HKV + head) * DH;
    } else if (slot < HQ + 2 * HKV) {
        kind = 2; head = slot - HQ - HKV; width = DH;
        src = v + ((size_t)row * HKV + head) * DH;
        dst = v_out + ((size_t)row * HKV + head) * DH;
    } else {
        kind = 3; head = slot - HQ - 2 * HKV; width = DR;
        src = r + ((size_t)row * HQ + head) * DR;
        dst = r_out + ((size_t)row * HQ + head) * DR;
    }
    float ss = 0.0f;
    for (uint d = lane; d < width; d += 32) {
        float val = float(src[d]);
        if (kind == 0) val = val * float(q_scale[head * DH + d]);
        else if (kind == 3) val = val * float(rel_scale[head * DR + d]);
        ss += val * val;
    }
    ss = simd_sum(ss);
    float f;
    if (kind == 2) {
        f = metal::precise::rsqrt(ss / float(width) + params[2]);
    } else {
        float th = kind == 3 ? params[1] : params[0];
        f = metal::min(th / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
    }
    for (uint d = lane; d < width; d += 32) {
        float val = float(src[d]);
        if (kind == 0) val = val * float(q_scale[head * DH + d]);
        else if (kind == 3) val = val * float(rel_scale[head * DR + d]);
        dst[d] = T(val * f);
    }
"""
_QKV_NORM_KERNEL = []
_QKV_NORM_PARAMS = {}


def _qkv_norm_fast(
    q, k, v, r, q_scale, rel_scale, n_heads, n_kv, head_dim, d_rel, qk_th, rel_th, eps
):
    """l2_clip(q*q_scale), l2_clip(k), rms_norm(v), l2_clip(r*rel_scale) in one
    launch for a few rows; None when not applicable."""
    dtype = q.dtype
    if (
        q.ndim != 3
        or dtype not in _ATTN_RES_DTYPES
        or any(a.dtype != dtype for a in (k, v, r))
        or q_scale.dtype not in _ATTN_RES_DTYPES
        or rel_scale.dtype not in _ATTN_RES_DTYPES
        or q.shape[0] * q.shape[1] > 8
        or q.shape[-1] != n_heads * head_dim
        or k.shape != (q.shape[0], q.shape[1], n_kv * head_dim)
        or v.shape != k.shape
        or r.shape != (q.shape[0], q.shape[1], n_heads * d_rel)
        or q_scale.shape != (n_heads, head_dim)
        or rel_scale.shape != (n_heads, d_rel)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    if not _QKV_NORM_KERNEL:
        _QKV_NORM_KERNEL.append(
            mx.fast.metal_kernel(
                name="inkling2_qkv_norm",
                input_names=["q", "k", "v", "r", "q_scale", "rel_scale", "params"],
                output_names=["q_out", "k_out", "v_out", "r_out"],
                source=_QKV_NORM_SRC,
                compile_options={"math_mode": "safe"},
            )
        )
    key = (float(qk_th), float(rel_th), float(eps))
    params = _QKV_NORM_PARAMS.get(key)
    if params is None:
        params = mx.array(list(key), mx.float32)
        _QKV_NORM_PARAMS[key] = params
    rows = q.shape[0] * q.shape[1]
    return _QKV_NORM_KERNEL[0](
        inputs=[q, k, v, r, q_scale, rel_scale, params],
        template=[
            ("T", dtype),
            ("HQ", n_heads),
            ("HKV", n_kv),
            ("DH", head_dim),
            ("DR", d_rel),
        ],
        grid=(32, 2 * n_heads + 2 * n_kv, rows),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape, k.shape, v.shape, r.shape],
        output_dtypes=[dtype, dtype, dtype, dtype],
    )


@mx.compile
def _output_gate(out, gate):
    return (
        out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
    ).astype(out.dtype)


@mx.compile
def _sum_f32(a, b):
    return a.astype(mx.float32) + b.astype(mx.float32)


_SUM_AS = {}


def _sum_f32_as(a, b, dtype):
    fn = _SUM_AS.get(dtype)
    if fn is None:
        fn = mx.compile(
            lambda p, q: (p.astype(mx.float32) + q.astype(mx.float32)).astype(dtype)
        )
        _SUM_AS[dtype] = fn
    return fn(a, b)


@mx.compile
def _log_scale(q, r, offset, alpha, floor):
    positions = mx.arange(q.shape[1]) + offset[..., None] + 1
    tau = 1 + alpha * mx.log(mx.maximum(positions / floor, 1))
    tau = tau[..., None, None]
    return (
        (q.astype(mx.float32) * tau).astype(q.dtype),
        (r.astype(mx.float32) * tau).astype(r.dtype),
    )


@mx.compile
def _dynamic_conv(x, context, dynamic, static_weight, num_heads):
    b, length, channels = x.shape
    k = static_weight.shape[0]
    xf = x.astype(mx.float32)
    dynamic = dynamic.astype(mx.float32).reshape(b, length, num_heads, 1, k)
    weights = static_weight.T.reshape(num_heads, -1, k) + dynamic
    out = mx.zeros_like(xf).reshape(b, length, num_heads, -1)
    for lag in range(k):
        values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
        out = out + values * weights[..., lag]
    return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


@mx.compile
def _blend_controllers(x, message, down, gates, eps):
    """Several MessageControllers at once: row i of the second-to-last axis uses
    controller i (stacked weights). Same per-row arithmetic as MessageController."""
    x = mx.broadcast_to(x, message.shape)
    width = x.shape[-1]
    count = message.shape[-2]
    pair = mx.concatenate(
        [mx.fast.rms_norm(x, None, eps), mx.fast.rms_norm(message, None, eps)],
        axis=-1,
    )
    pair = pair.reshape(-1, count, 2 * width).swapaxes(0, 1).astype(down.dtype)
    hidden = (pair @ down.swapaxes(-1, -2)).astype(mx.float32)
    fg = mx.tanh(hidden @ gates.swapaxes(-1, -2))
    fg = fg.swapaxes(0, 1).reshape(*message.shape[:-1], 2 * width)
    forget, gain = fg[..., :width], fg[..., width:]
    return (1 - forget) * x + gain * message


@mx.compile
def l2_clip(x, threshold, scale=None):
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


@mx.compile
def _softcapped_swiglu(up, gate, gate_cap, up_cap):
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    return (
        mx.tanh(gate / gate_cap)
        * gate_cap
        * mx.sigmoid(gate)
        * mx.tanh(up / up_cap)
        * up_cap
    ).astype(dtype)


@mx.compile
def _route(x, weight, bias, top_k, route_scale):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


@mx.compile
def _combine_routes(out, weights):
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


@mx.compile
def _attention_residual(states, query, normalize_values):
    states = states.astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (keys @ query.reshape(-1, states.shape[-1]).T) * states.shape[-1] ** -0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ (keys if normalize_values else states)


_conv_history_kernel = mx.fast.metal_kernel(
    name="inkling_masked_conv_history",
    input_names=["state", "inputs", "mask"],
    output_names=["out"],
    ensure_row_contiguous=False,
    source=r"""
    uint index = thread_position_in_grid.x;
    int B = inputs_shape[0], L = inputs_shape[1], C = inputs_shape[2];
    int W = state_shape[1];
    if (index >= B * L * W * C) return;
    int c = index % C, w = (index / C) % W;
    int t = (index / (C * W)) % L, b = index / (C * W * L);
    int count = 0, start = 0;
    for (int j = 0; j <= t; ++j) {
        if (mask[b * mask_strides[0] + j * mask_strides[1]]) {
            if (count == 0) start = j;
            ++count;
        }
    }
    int logical = count - W + w;
    out[index] = logical < 0
        ? state[b * state_strides[0] + (W + logical) * state_strides[1]
                + c * state_strides[2]]
        : inputs[b * inputs_strides[0] + (start + logical) * inputs_strides[1]
                 + c * inputs_strides[2]];
    """,
)


@mx.compile
def _conv_state_history_fallback(state, inputs, mask):
    width = state.shape[1]
    count = mx.cumsum(mask.astype(mx.int32), axis=1)
    start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(width)[None, None, :] + count[:, :, None] - width
    indices = mx.where(
        logical >= 0,
        width + start[:, None, None] + mx.maximum(logical, 0),
        width + logical,
    )
    combined = mx.concatenate([state, inputs], axis=1)
    return mx.take_along_axis(combined[:, None], indices[..., None], axis=2)


def _conv_state_history(state, inputs, mask):
    """Select every masked prefix's state without repeated prefix forwards."""
    batch, length, channels = inputs.shape
    width = state.shape[1]
    if (
        width > 0
        and length <= 8
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        return _conv_history_kernel(
            inputs=[state, inputs, mask],
            grid=(batch * length * width * channels, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, length, width, channels)],
            output_dtypes=[state.dtype],
        )[0]
    return _conv_state_history_fallback(state, inputs, mask)


def _update_conv_cache(cache, index, state, inputs, mask):
    if cache is None:
        return
    speculating = getattr(cache, "is_speculating", False)
    if speculating and mask is not None:
        # Physical prefixes can retain different numbers of valid tokens.
        # Avoid rebuilding a separate reduction, concat and gather per prefix.
        states = _conv_state_history(state, inputs, mask)
        final = states[:, -1]
        cache.record_speculative_states(
            index, states[:, :-1] if inputs.shape[1] > 1 else None, final
        )
    else:
        final = _next_conv_state(state, inputs, mask)
        if speculating:
            cache.record_speculative_window(
                index, mx.concatenate([state, inputs], axis=1), state.shape[1]
            )
    cache[index] = final


class ShortConvolution(InklingShortConvolution):
    def fused(self, x, cache, mask, residual, mode):
        """Decode kernel with the decoder's residual add folded in; None when the
        stock kernel path would not run. mode 1: fp32 out = conv (+ residual);
        mode 2: model-dtype out = residual + conv. Same roundings as the glue."""
        if not (
            FAST_DECODE
            and FAST_SCONV_RES
            and not self.training
            and x.ndim == 3
            and x.dtype == mx.bfloat16
            and self.kernel_size == 4
            and x.shape[0] <= 4
            and x.shape[1] <= 8
            and x.shape[-1] % 128 == 0
            and (
                residual is None
                or (
                    residual.shape == x.shape
                    and residual.dtype in (mx.float32, mx.bfloat16, mx.float16)
                )
            )
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            return None
        from .kernels import short_convolution_residual

        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]))
        _update_conv_cache(cache, self.conv_idx, state, x, mask)
        return short_convolution_residual(
            state, self.conv.weight, x, mask, self.kernel_size, residual, mode
        )

    def __call__(self, x, cache=None, mask=None):
        # Keep the convolution, residual, and scaling in FP32. The checkpoint
        # stores BF16 taps, but rounding the convolution before adding the
        # residual changes both reference parity and prefill/decode consistency.
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]))
        if (
            not self.training
            and x.dtype == mx.bfloat16
            and self.kernel_size == 4
            and x.shape[0] <= 4
            and x.shape[1] <= 8
            and x.shape[-1] % 128 == 0
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            from .kernels import short_convolution

            # Temporal selection promotes selected BF16 values into FP32 state.
            # Invalid positions never enter that state. The convolution itself
            # reads masked inputs directly, without materializing a concat.
            _update_conv_cache(cache, self.conv_idx, state, x, mask)
            return short_convolution(state, self.conv.weight, x, mask, self.kernel_size)
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        context = mx.concatenate([state, inputs], axis=1)
        _update_conv_cache(cache, self.conv_idx, state, inputs, mask)
        out = mx.conv1d(
            context, self.conv.weight.astype(mx.float32), groups=x.shape[-1]
        )
        return ((out + xf) / math.sqrt(self.kernel_size + 1)).astype(x.dtype)


class DynamicShortConvolution(nn.Module):
    def __init__(self, config, channels, num_heads, conv_idx):
        super().__init__()
        self.kernel_size = config.sconv_kernel_size
        self.num_heads = num_heads
        self.conv_idx = conv_idx
        self.static_weight = mx.zeros((self.kernel_size, channels))
        self.weight_proj = nn.Linear(
            config.hidden_size, num_heads * self.kernel_size, bias=False
        )

    def __call__(self, x, dynamic_input, cache=None, mask=None):
        b, length, channels = x.shape
        k = self.kernel_size
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((b, k - 1, channels), mx.float32)
        context = mx.concatenate([state, inputs], axis=1)
        _update_conv_cache(cache, self.conv_idx, state, inputs, mask)
        if _fast_glue(x) and not self.training:
            return _dynamic_conv(
                x,
                context,
                self.weight_proj(dynamic_input),
                self.static_weight,
                self.num_heads,
            )
        dynamic = (
            self.weight_proj(dynamic_input)
            .astype(mx.float32)
            .reshape(b, length, self.num_heads, 1, k)
        )
        weights = self.static_weight.T.reshape(self.num_heads, -1, k) + dynamic
        out = mx.zeros_like(xf).reshape(b, length, self.num_heads, -1)
        for lag in range(k):
            values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
            out = out + values * weights[..., lag]
        return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = self.sliding or config.rel_extent
        self.sharding_group = None
        self.head_start = 0
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.r_proj = nn.Linear(
            config.hidden_size, self.n_heads * config.d_rel, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.sink_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.rel_scale = mx.ones((self.n_heads, config.d_rel))
        self.rel_proj = mx.zeros((config.d_rel, self.rel_extent))
        for idx, name in enumerate(("k_sconv", "v_sconv")):
            channels = self.n_kv * self.head_dim
            self[name] = (
                ShortConvolution(channels, config.sconv_kernel_size, idx)
                if self.is_sliding
                else DynamicShortConvolution(config, channels, self.n_kv, idx)
            )

    def shard(self, group):
        """Split complete heads and output rows without splitting dot products."""
        if self.sharding_group is not None:
            return
        projections = ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "o_proj")
        # Preserve the dense kernels' output-row tiles. Other configurations
        # retain replicated attention while the experts are still partitioned.
        if (
            group.size() != 2
            or self.n_heads % 2
            or self.n_kv % 2
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or any(
                type(self[name]) is not nn.Linear
                or "bias" in self[name]
                or self[name].weight.shape[0] % 256
                for name in projections
            )
            or (
                not self.is_sliding
                and any(
                    type(self[name].weight_proj) is not nn.Linear
                    or "bias" in self[name].weight_proj
                    for name in ("k_sconv", "v_sconv")
                )
            )
        ):
            return
        rank = group.rank()
        for name in projections:
            projection = self[name]
            width = projection.weight.shape[0] // 2
            projection.weight = mx.contiguous(
                projection.weight[rank * width : (rank + 1) * width]
            )
        for name in ("q_scale", "rel_scale"):
            value = self[name]
            width = value.shape[0] // 2
            self[name] = mx.contiguous(value[rank * width : (rank + 1) * width])
        for name in ("k_sconv", "v_sconv"):
            conv = self[name]
            if self.is_sliding:
                width = conv.conv.weight.shape[0] // 2
                conv.conv.weight = mx.contiguous(
                    conv.conv.weight[rank * width : (rank + 1) * width]
                )
            else:
                conv.num_heads //= 2
                width = conv.static_weight.shape[1] // 2
                conv.static_weight = mx.contiguous(
                    conv.static_weight[:, rank * width : (rank + 1) * width]
                )
                width = conv.weight_proj.weight.shape[0] // 2
                conv.weight_proj.weight = mx.contiguous(
                    conv.weight_proj.weight[rank * width : (rank + 1) * width]
                )
        self.n_heads //= 2
        self.n_kv //= 2
        self.head_start = rank * self.n_heads
        self.sharding_group = group

    def _gather_features(self, x):
        if self.sharding_group is None:
            return x
        shape = x.shape
        return (
            mx.distributed.all_gather(
                x.reshape(1, -1, shape[-1]), group=self.sharding_group
            )
            .transpose(1, 0, 2)
            .reshape(*shape[:-1], shape[-1] * self.sharding_group.size())
        )

    def _project(self, projection, x):
        if self.sharding_group is None:
            return projection(x)
        # Both GEMV and GEMM dispatch can depend on N, including short batches
        # on older GPUs. Preserve the original N with zero rows for the other
        # rank, then select our complete rows. No remote weights are retained.
        width = projection.weight.shape[0]
        start = self.sharding_group.rank() * width
        weight = mx.pad(projection.weight, ((start, width - start), (0, 0)))
        return (x @ weight.T)[..., start : start + width]

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        k = self._project(self.k_proj, key_input)
        v = self._project(self.v_proj, value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        qp = self._project(self.q_proj, x)
        rp = self._project(self.r_proj, x)
        post = None
        if (
            FAST_DECODE
            and FAST_QKV_NORM
            and not self.training
            and cfg.use_value_norm_no_gamma
            and qp.dtype == x.dtype
        ):
            post = _qkv_norm_fast(
                qp,
                k,
                v,
                rp,
                self.q_scale,
                self.rel_scale,
                self.n_heads,
                self.n_kv,
                self.head_dim,
                cfg.d_rel,
                cfg.qk_l2_clip_threshold,
                cfg.rel_l2_clip_threshold,
                cfg.rms_norm_eps,
            )
        if post is not None:
            q, k, v, r = post
            q = q.reshape(b, length, self.n_heads, self.head_dim)
            k = k.reshape(b, length, self.n_kv, self.head_dim)
            v = v.reshape(b, length, self.n_kv, self.head_dim)
            r = r.reshape(b, length, self.n_heads, cfg.d_rel)
        else:
            q = l2_clip(
                qp.reshape(b, length, self.n_heads, self.head_dim),
                cfg.qk_l2_clip_threshold,
                self.q_scale,
            )
            k = l2_clip(
                k.reshape(b, length, self.n_kv, self.head_dim),
                cfg.qk_l2_clip_threshold,
            )
            v = v.reshape(b, length, self.n_kv, self.head_dim)
            if cfg.use_value_norm_no_gamma:
                v = mx.fast.rms_norm(
                    v.astype(mx.float32), None, cfg.rms_norm_eps
                ).astype(x.dtype)
            r = l2_clip(
                rp.reshape(b, length, self.n_heads, cfg.d_rel),
                cfg.rel_l2_clip_threshold,
                self.rel_scale,
            )
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset if kv is not None else 0
            if _fast_glue(x) and q.dtype == x.dtype and r.dtype == x.dtype:
                q, r = _log_scale(
                    q,
                    r,
                    mx.array(offset),
                    cfg.log_scaling_alpha,
                    cfg.log_scaling_n_floor,
                )
            else:
                positions = mx.arange(length) + mx.array(offset)[..., None] + 1
                tau = 1 + cfg.log_scaling_alpha * mx.log(
                    mx.maximum(positions / cfg.log_scaling_n_floor, 1)
                )
                q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
                r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        fast = FAST_DECODE and length == 1 and b == 1
        roll = 0
        # A plain growing cache (global layers) stores the zero sink key/value in its
        # spare row after the new rows, so the attention reads it in place instead of
        # copying the whole history to append it (prompt, check rows, batches).
        slot = (
            FAST_DECODE
            and FAST_SLOT
            and not (fast and FAST_SINK)
            and type(kv) is KVCache
        )
        if slot:
            zero_k = mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)
            zero_v = mx.zeros((b, self.n_kv, 1, self.head_dim), v.dtype)
            k, v = kv.update_and_fetch(
                mx.concatenate([k, zero_k], axis=2), mx.concatenate([v, zero_v], axis=2)
            )
            kv.trim(1)
        elif kv is not None:
            k, v = kv.update_and_fetch(k, v)
            # Relative logits index backward distance, so ring buffers must be
            # read in chronological order (ordinary RoPE attention does not care).
            if (
                fast
                and FAST_ROLL
                and type(kv) is RotatingKVCache
                and kv.keep == 0
                and k.shape[2] == kv.max_size
                and 0 < kv._idx < kv.offset
            ):
                # Physical slot p holds chronological row (p - _idx) mod S: roll the
                # mask (heads x S) instead of copying the keys and values.
                roll = int(kv._idx)
            elif isinstance(kv, RotatingKVCache):
                k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        size = k.shape[2] - (1 if slot else 0)
        padding = getattr(kv, "left_padding", None)
        kref = k[..., :size, :] if slot else k
        proj = self.rel_proj.astype(x.dtype)
        mask = None
        if FAST_DECODE and FAST_MASK and padding is None and not self.training:
            mask = _row_mask_fast(
                r, proj, kref, size, self.sliding, self.rel_extent, roll
            )
            if mask is not None:
                roll = 0
        if mask is None:
            mask = banded_additive_mask(
                r,
                proj,
                size - length,
                size,
                self.sliding,
                self.rel_extent,
                left_padding=padding,
                shape_ref=kref,
            )
        # A query-dependent sink is represented by a zero K/V and its own logit.
        # Keep this tiny projection's original width: splitting its 48 output
        # rows changes the prefill GEMM reduction and hence BF16 rounding.
        if roll:
            mask = mx.roll(mask, roll, axis=-1)
        sinks = self.sink_proj(x)
        if self.sharding_group is not None:
            sinks = sinks[..., self.head_start : self.head_start + self.n_heads]
        if fast and FAST_SINK:
            # One row: the sink logit is one value per head, as the fused kernel takes it.
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.head_dim**-0.5, mask=mask, sinks=sinks.reshape(-1)
            )
            return self._finish(out, x, b, length)
        sinks = sinks.transpose(0, 2, 1)[..., None]
        mask = mx.concatenate([mask, sinks], axis=-1)
        if slot:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.head_dim**-0.5, mask=mask
            )
            return self._finish(out, x, b, length)
        k = mx.concatenate(
            [k, mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)], axis=2
        )
        v = mx.concatenate(
            [v, mx.zeros((b, self.n_kv, 1, self.head_dim), v.dtype)], axis=2
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        return self._finish(out, x, b, length)

    def _finish(self, out, x, b, length):
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        gate = self._project(self.gate_proj, x)
        if _fast_glue(x) and not self.training and out.dtype == x.dtype:
            out = _output_gate(out, gate)
        else:
            out = (
                out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
            ).astype(x.dtype)
        # Every output dot still consumes all heads on one rank. Gather the
        # heads first, project complete output rows, then restore hidden order.
        return self._gather_features(
            self._project(self.o_proj, self._gather_features(out))
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap

    def __call__(self, up, gate):
        return _softcapped_swiglu(up, gate, self.gate_cap, self.up_cap)


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.sharding_group = None
        self.expert_start = 0
        self.expert_end = config.n_routed_experts
        self.gate_weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.intermediate_size,
            config.n_routed_experts,
            activation=SoftcappedSwiGLU(config),
        )

    def _gate_f32(self):
        # The router multiplies in FP32; keep the converted weight instead of
        # converting it again on every call (the same values).
        weight = self.gate_weight
        if weight.dtype == mx.float32:
            return weight
        cached = self.get("_gate_weight_f32")
        if cached is None or cached[0] is not weight:
            self._gate_weight_f32 = (weight, weight.astype(mx.float32))
        return self._gate_weight_f32[1]

    def __call__(self, x):
        indices, weights = _route(
            x,
            self._gate_f32() if FAST_DECODE and FAST_GATE else self.gate_weight,
            self.e_score_correction_bias,
            self.top_k,
            self.route_scale,
        )
        if self.sharding_group is None:
            if (
                not self.training
                and x.shape[0] <= 4
                and 1 < x.shape[1] <= 8
                and getattr(self.switch_mlp.down_proj, "mode", None) == "nvfp4"
            ):
                out = self.switch_mlp.masked(x, indices.astype(mx.int32))
            else:
                out = self.switch_mlp(x, indices)
        else:
            owned = (indices >= self.expert_start) & (indices < self.expert_end)
            local = mx.where(owned, indices.astype(mx.int32) - self.expert_start, -1)
            out = self.switch_mlp.masked(x, local)
            # Exactly one rank owns each route. Reduce route outputs before
            # weighting to retain the unsharded FP32 accumulation order.
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return _combine_routes(out, weights)


class AttentionResidual(nn.Module):
    def __init__(self, config, num_queries=1):
        super().__init__()
        self.query = mx.zeros(
            (config.hidden_size,)
            if num_queries == 1
            else (num_queries, config.hidden_size)
        )
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks, partial):
        if FAST_DECODE and FAST_ATTN_RES and not self.training:
            out = _attention_residual_fast(
                blocks, partial, self.query, self.normalize_values
            )
            if out is not None:
                return out
        return _attention_residual(
            mx.stack([*blocks, partial], axis=-2), self.query, self.normalize_values
        )


class MessageController(nn.Module):
    """Blend a previous pass's message into one sublayer input."""

    def __init__(self, config):
        super().__init__()
        rank, width = config.loop_config.gate_rank, config.hidden_size
        self.down_weight = mx.zeros((rank, 2 * width))
        self.forget_weight = mx.zeros((width, rank))
        self.gain_weight = mx.zeros((width, rank))
        self.eps = config.rms_norm_eps

    def __call__(self, x, message):
        pair = mx.concatenate(
            [mx.fast.rms_norm(v, None, self.eps) for v in (x, message)], axis=-1
        )
        hidden = (pair.astype(self.down_weight.dtype) @ self.down_weight.T).astype(
            mx.float32
        )
        forget = mx.tanh(hidden @ self.forget_weight.T)
        gain = mx.tanh(hidden @ self.gain_weight.T)
        return (1 - forget) * x + gain * message


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.attention_input = (
            AttentionResidual(config, 1 if self.is_sliding else 3)
            if layer_idx
            else None
        )
        self.mlp_input = AttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoE(config)
        names = (
            ("input_layernorm",)
            if self.is_sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        for name in (*names, "post_attention_layernorm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 2
        )
        self.mlp_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 3
        )
        loop = config.loop_config
        if loop is not None and loop.core_start <= layer_idx < loop.core_end:
            self.controllers = {
                name: MessageController(config) for name in ("attn", "k", "v", "mlp")
            }

    def _fast_controllers(self, x):
        return (
            FAST_DECODE
            and FAST_CTRL
            and not self.training
            and x.shape[1] <= 8
            and "controllers" in self
        )

    def _controller_stack(self, names):
        # Stacked controller weights, rebuilt only when the weights change.
        ctrls = [self.controllers[n] for n in names]
        sources = tuple(
            a for c in ctrls for a in (c.down_weight, c.forget_weight, c.gain_weight)
        )
        stacks = vars(self).get("_ctrl_stacks")
        if stacks is None:
            stacks = {}
            object.__setattr__(self, "_ctrl_stacks", stacks)
        cached = stacks.get(names)
        if (
            cached is None
            or len(cached[0]) != len(sources)
            or any(a is not b for a, b in zip(cached[0], sources))
        ):
            down = mx.stack([c.down_weight for c in ctrls])
            gates = mx.stack(
                [
                    mx.concatenate([c.forget_weight, c.gain_weight], axis=0).astype(
                        mx.float32
                    )
                    for c in ctrls
                ]
            )
            mx.eval(down, gates)
            cached = (sources, down, gates)
            stacks[names] = cached
        return cached[1], cached[2]

    def __call__(self, x, blocks, cache=None, messages=None):
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = (
            self.attention_input(blocks, x)
            if self.attention_input is not None
            else x[..., None, :]
        )
        if self.is_sliding and messages is None:
            attn = self.self_attn(
                self.input_layernorm(mixed[..., 0, :]).astype(x.dtype),
                cache=cache,
                conv_mask=mask,
            )
        else:
            fast_ctrl = messages is not None and self._fast_controllers(x)
            if fast_ctrl:
                down, gates = self._controller_stack(("attn", "k", "v"))
                blended = _blend_controllers(
                    mixed, messages[..., :3, :], down, gates,
                    self.controllers["attn"].eps,
                )
                parts = [blended[..., i, :] for i in range(3)]
            else:
                parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            if messages is not None and not fast_ctrl:
                parts = [
                    self.controllers[name](part, messages[..., i, :])
                    for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))
                ]
            norms = (
                (self.input_layernorm,) * 3
                if self.is_sliding
                else (
                    self.q_input_layernorm,
                    self.k_input_layernorm,
                    self.v_input_layernorm,
                )
            )
            parts = [norm(part).astype(x.dtype) for norm, part in zip(norms, parts)]
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        glue = _fast_glue(x) and not self.training
        residual = None
        if glue:
            residual = self.attn_sconv.fused(
                attn, conv, mask, None if self.is_block_boundary else x, 1
            )
        if residual is None:
            attn_out = self.attn_sconv(attn, conv, mask)
            if self.is_block_boundary:
                residual = attn_out.astype(mx.float32)
            elif glue:
                residual = _sum_f32(attn_out, x)
            else:
                residual = attn_out.astype(mx.float32) + x.astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        if messages is not None and self._fast_controllers(x):
            down, gates = self._controller_stack(("mlp",))
            hidden = _blend_controllers(
                hidden[..., None, :], messages[..., 3:, :], down, gates,
                self.controllers["mlp"].eps,
            )[..., 0, :]
        elif messages is not None:
            hidden = self.controllers["mlp"](hidden, messages[..., 3, :])
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = None
        if glue and hidden.dtype == x.dtype and residual.dtype == mx.float32:
            out = self.mlp_sconv.fused(hidden, conv, mask, residual, 2)
        if out is None:
            mlp_out = self.mlp_sconv(hidden, conv, mask)
            if glue:
                out = _sum_f32_as(residual, mlp_out, x.dtype)
            else:
                out = (residual + mlp_out.astype(mx.float32)).astype(x.dtype)
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.layer_indices = config.layer_indices
        self.loop = config.loop_config
        if self.loop is not None:
            self.message_queries = mx.zeros(
                (self.loop.core_end - self.loop.core_start, 4, config.hidden_size)
            )
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        skip_final_norm=False,
        capture_layer_ids=None,
        speculative_verify=False,
    ):
        captures = {}
        if capture_layer_ids is not None and (
            len(capture_layer_ids) != len(set(capture_layer_ids))
            or any(
                type(i) is not int or not 0 <= i <= len(self.layer_indices)
                for i in capture_layer_ids
            )
        ):
            raise ValueError("Invalid Inkling 2 capture_layer_ids")
        requested = set(capture_layer_ids or [])
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        blocks = []
        bank = None
        bank_states = None
        if cache and len(cache) != len(self.layer_indices):
            raise ValueError(
                "Inkling 2 requires a separate cache for every layer visit"
            )
        for visit, index in enumerate(self.layer_indices):
            if self.loop is not None and index == self.loop.core_start:
                if visit == index:
                    prelude, prelude_blocks = x, blocks.copy()
                else:
                    # The first block appended in the core is the prelude partial.
                    # Only completed core blocks and the last partial supply messages.
                    bank_states = [*blocks[len(prelude_blocks) + 1 :], x]
                    bank = None
                    if not (
                        FAST_DECODE
                        and FAST_MESSAGES
                        and not self.training
                        and x.ndim == 3
                        and x.shape[0] * x.shape[1] <= 8
                    ):
                        bank = _message_bank(bank_states)
                    x, blocks = prelude, prelude_blocks.copy()
            messages = None
            if bank_states is not None and index < self.loop.core_end:
                query = self.message_queries[index - self.loop.core_start]
                if bank is None:
                    messages = _attention_residual_fast(
                        bank_states[:-1], bank_states[-1], query, True
                    )
                    if messages is None:
                        bank = _message_bank(bank_states)
                if bank is not None:
                    scores = (query @ bank.swapaxes(-1, -2)) * x.shape[-1] ** -0.5
                    messages = mx.softmax(scores, axis=-1) @ bank
            # DSpark taps the partial block entering the layer, not its output.
            if visit in requested:
                captures[visit] = x
            x = self.layers[index](x, blocks, cache[visit] if cache else None, messages)
            if speculative_verify and (visit + 1) % 4 == 0:
                mx.async_eval(x)
        h = self.final_input(blocks, x)[..., 0, :]
        normalized = self.norm(h).astype(x.dtype)
        if len(self.layer_indices) in requested:
            captures[len(self.layer_indices)] = normalized
        result = h if skip_final_norm else normalized
        if capture_layer_ids is not None:
            return result, [captures[i] for i in capture_layer_ids]
        return result


class LanguageModel(InklingLanguageModel):
    def __init__(self, config: TextConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sharding_group = None

    def chunked_prefill_policy(self, **kwargs):
        # Captures are per-token partial blocks; the shared prefill collector
        # retains them across chunks. Preserve the caller's target chunk size.
        return True

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        cache=None,
        return_hidden=False,
        return_shared_kv=False,
        skip_logits=False,
        capture_layer_ids=None,
        speculative_verify=False,
        **kwargs,
    ):
        from ...speculative.cache_state import start_speculative_cache

        transaction = None
        if speculative_verify:
            length = inputs.shape[1] if inputs is not None else inputs_embeds.shape[1]
            transaction = start_speculative_cache(cache, length)
        try:
            result = self.model(
                inputs,
                cache=cache,
                input_embeddings=inputs_embeds,
                skip_final_norm=True,
                capture_layer_ids=capture_layer_ids,
                speculative_verify=speculative_verify,
            )
            pre_norm, captured = (
                result if capture_layer_ids is not None else (result, None)
            )
            return LanguageModelOutput(
                logits=(
                    None
                    if skip_logits
                    else self._logits_from_norm(self.model.norm(pre_norm))
                ),
                hidden_states=(
                    captured
                    if capture_layer_ids is not None
                    else ([pre_norm] if return_hidden else None)
                ),
                shared_kv_states={} if return_shared_kv else None,
                gdn_states=transaction,
            )
        except BaseException:
            if transaction is not None:
                transaction.abort()
            raise

    def shard(self, group=None):
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        count = self.config.n_routed_experts
        validate_sharding(vars(self.config), size)
        if not 0 <= rank < size:
            raise ValueError("Each distributed rank must own at least one expert")
        if self.sharding_group is not None:
            raise ValueError("Inkling 2 is already sharded")
        if size == 1:
            return
        start, end = count * rank // size, count * (rank + 1) // size
        scalar_scales = {"weight_scale_2", "weight_global_scale"}
        # Check every layer before changing any weights, including native
        # packed weights, group scales, and per-expert FP32 outer scales.
        for layer in self.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer.mlp.switch_mlp, name)
                for key, value in projection.parameters().items():
                    if not isinstance(value, mx.array):
                        continue
                    if key in scalar_scales and value.size == 1:
                        continue
                    if not value.ndim or value.shape[0] != count:
                        raise ValueError(
                            f"Invalid expert partition for {name}.{key}: "
                            f"expected {count} experts, got {value.shape}"
                        )
        for layer in self.layers:
            moe = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(moe.switch_mlp, name)
                projection.update(
                    {
                        key: mx.contiguous(value[start:end])
                        for key, value in projection.parameters().items()
                        if isinstance(value, mx.array)
                        and not (key in scalar_scales and value.size == 1)
                    }
                )
            # Materialize one layer at a time and release the full expert
            # buffers; lazy slices alone retain the unpartitioned allocation.
            mx.eval(moe.switch_mlp.parameters())
            mx.clear_cache()
            moe.expert_start, moe.expert_end = start, end
            moe.sharding_group = group
            layer.self_attn.shard(group)
            mx.eval(layer.self_attn.parameters())
            mx.clear_cache()
        self.sharding_group = group

    def _logits_from_norm(self, h):
        # hf_quant_config.json stores BF16 unembedding weights but requires
        # fwd_output_dtype=float32. Promote the product before multiplication;
        # casting BF16 scores afterwards loses the accumulator's low bits.
        h = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
        if (
            not self.training
            and type(self.lm_head) is nn.Linear
            and "bias" not in self.lm_head
            and h.ndim == 3
            and h.shape[0] * h.shape[1] <= 15
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            # Retain the exact conversion instead of reading and promoting the
            # entire vocabulary matrix again on every decode step. Whole output
            # rows can be shared across the two expert ranks without splitting
            # any dot product or changing its FP32 accumulation.
            weight = self.lm_head.weight
            group = self.sharding_group
            partition = (
                group is not None
                and group.size() == 2
                and weight.shape[0] % group.size() == 0
            )
            size, rank = (group.size(), group.rank()) if partition else (1, 0)
            cached = self.get("_head_projection_cache")
            if cached is None or cached[0] is not weight or cached[1:3] != (size, rank):
                width = weight.shape[0] // size
                projected = weight[rank * width : (rank + 1) * width].astype(mx.float32)
                self._head_projection_cache = (weight, size, rank, projected)
            projected = self._head_projection_cache[-1]
            logits = h @ projected.T
            if partition:
                width = projected.shape[0]
                logits = (
                    mx.distributed.all_gather(logits.reshape(1, -1, width), group=group)
                    .transpose(1, 0, 2)
                    .reshape(*h.shape[:-1], weight.shape[0])
                )
        else:
            logits = self.lm_head(h)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    def make_cache(self):
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                    if self.model.layers[i].is_sliding
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for i in self.model.layer_indices
        ]

    def speculative_verify_hidden(self, inputs, cache):
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            skip_logits=True,
            speculative_verify=True,
        )
        return out.hidden_states[-1], {}, out.gdn_states

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        from ...speculative.cache_state import rollback_speculative_cache

        return rollback_speculative_cache(caches, gdn_states, accepted, block_size)
