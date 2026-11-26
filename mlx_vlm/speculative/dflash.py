import copy
import os
import time
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .. import apc
from ..sample_utils import apply_top_k, apply_top_p
from .cache_state import abort_speculative_round, commit_speculative_round
from .common import (
    _apply_thinking_budget,
    _dflash_block_total,
    _filter_rope_deltas,
    _record_speculative_round,
    _requires_uniform_batch_acceptance,
    _speculative_walk,
    _speculative_walk_batch,
    _SpeculativeSamplerRNG,
    generation_stream,
    verify_forward,
)


def _dflash_next_block_size(
    draft_model: nn.Module,
    requested_block_total: int,
    remaining_budget: int,
    initial_block_size: Optional[int] = None,
) -> int:
    """Choose the next DFlash verify block size from recent acceptance.

    DFlash checkpoints advertise a trained block size, usually 16. Treat that
    as the ceiling and back off quickly when deeper positions are mostly
    rejected. When acceptance is strong at the current depth, grow back toward
    the configured ceiling.
    """
    block_total = min(requested_block_total, remaining_budget)
    if block_total <= 1:
        return block_total
    if getattr(draft_model, "prefer_requested_block_size", False):
        return block_total

    accept_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    recent = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-8:], draft_lens[-8:])
        if int(d) > 0
    ]
    if not recent:
        if initial_block_size is not None:
            return min(block_total, max(2, int(initial_block_size)))
        return block_total

    current = min(block_total, max(2, recent[-1][1] + 1))
    min_total = min(
        block_total,
        max(2, int(getattr(draft_model, "dflash_min_block_size", 4))),
    )
    drafted = sum(d for _, d in recent)
    accepted = sum(a for a, _ in recent)
    accept_rate = accepted / drafted
    mean_accept = accepted / len(recent)

    if accept_rate < 0.30 or mean_accept < 2.0:
        if current >= 8:
            return max(min_total, min(block_total, current // 2))
        return max(min_total, min(block_total, current - 2))

    if accept_rate < 0.50:
        return max(min_total, min(block_total, current - 2))

    full_hits = sum(1 for a, d in recent if a >= d)
    full_hit_rate = full_hits / len(recent)
    if accept_rate >= 0.85 and full_hit_rate >= 0.75:
        return min(block_total, current + 2)

    return min(block_total, current)


def _dflash_committed_hidden_segments(
    hidden_full: mx.array, new_tokens_list: List[List[int]]
) -> List[mx.array]:
    return [
        hidden_full[i : i + 1, : len(new_tokens), :]
        for i, new_tokens in enumerate(new_tokens_list)
    ]


def _reserve_dflash_target_cache(prompt_cache: List[Any], block_size: int) -> None:
    pending = []
    for entry in prompt_cache:
        reserve = getattr(entry, "prefix_cache_reserve", None)
        offset = getattr(entry, "offset", None)
        if not callable(reserve) or not isinstance(offset, int):
            continue
        pending.extend(reserve(offset + block_size))
    if pending:
        mx.async_eval(*pending)


def _dflash_verify(lm, inputs, cache, capture_layer_ids):
    # Older architectures still own their verifier entry point. Models using
    # ordinary forward calls need only the shared cache transaction.
    if callable(getattr(lm, "rollback_speculative_cache", None)):
        output = lm(
            inputs,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            speculative_verify=True,
        )
        return output, output.gdn_states
    return verify_forward(lm, inputs, cache, capture_layer_ids=capture_layer_ids)


def _dflash_verify_greedy(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
    target_layer_ids: List[int],
    sampler: Callable[[mx.array], mx.array],
):
    gdn_states = None
    try:
        verify_hidden = getattr(lm, "speculative_verify_dflash_hidden", None)
        argmax_from_hidden = getattr(lm, "speculative_argmax_from_hidden", None)
        if callable(verify_hidden) and callable(argmax_from_hidden):
            captured, final_hidden, gdn_states = verify_hidden(
                verify_input, prompt_cache, target_layer_ids
            )
            target_tokens = argmax_from_hidden(final_hidden)
            if target_tokens is None:
                raise RuntimeError(
                    "speculative_argmax_from_hidden returned no greedy target tokens"
                )
            return captured, gdn_states, target_tokens

        verify_out, gdn_states = _dflash_verify(
            lm, verify_input, prompt_cache, target_layer_ids
        )
        return verify_out.hidden_states, gdn_states, sampler(verify_out.logits)

    except BaseException:
        abort_speculative_round(gdn_states)
        raise


def _supports_positioned_target_sampling(sampler: Callable) -> bool:
    return callable(getattr(sampler, "sample_target", None))


class _PositionedDraftSampler:
    def __init__(
        self,
        sampler: Callable,
        *,
        row_ids: List[int],
        positions: List[int],
    ):
        self.sampler = sampler
        self.row_ids = [int(row_id) for row_id in row_ids]
        self.positions = [int(position) for position in positions]

    def __call__(self, logits: mx.array) -> mx.array:
        if logits.ndim == 1:
            batch, length = 1, 1
        elif logits.ndim == 2:
            batch, length = logits.shape[0], 1
        else:
            batch, length = logits.shape[0], logits.shape[1]
        if batch != len(self.row_ids):
            raise ValueError(
                "Draft sampler row count does not match logits batch size."
            )

        rows = [row_id for row_id in self.row_ids for _ in range(length)]
        positions = [
            position + offset for position in self.positions for offset in range(length)
        ]
        sampled = self.sampler.sample_target(
            logits.reshape(batch * length, logits.shape[-1]),
            row_ids=rows,
            positions=positions,
        )
        self.positions = [position + length for position in self.positions]
        return sampled.reshape(logits.shape[:-1])

    def sample_proposal(self, logits: mx.array) -> mx.array:
        sample_proposal = getattr(self.sampler, "sample_proposal", None)
        if not callable(sample_proposal):
            return mx.argmax(logits, axis=-1)
        if logits.ndim == 1:
            batch, length = 1, 1
        elif logits.ndim == 2:
            batch, length = logits.shape[0], 1
        else:
            batch, length = logits.shape[0], logits.shape[1]
        if batch != len(self.row_ids):
            raise ValueError(
                "Draft sampler row count does not match logits batch size."
            )
        rows = [row_id for row_id in self.row_ids for _ in range(length)]
        positions = [
            position + offset for position in self.positions for offset in range(length)
        ]
        sampled = sample_proposal(
            logits.reshape(batch * length, logits.shape[-1]),
            row_ids=rows,
            positions=positions,
        )
        self.positions = [position + length for position in self.positions]
        return sampled.reshape(logits.shape[:-1])


def _sample_dflash_target_block(
    logits: mx.array,
    sampler: Callable[[mx.array], mx.array],
    *,
    row_ids: List[int],
    base_positions: List[int],
) -> mx.array:
    batch, length, vocab_size = logits.shape
    logprobs = _dflash_target_logprobs(logits)
    flat_logprobs = logprobs.reshape(batch * length, vocab_size)
    positions = [
        int(base_position) + position
        for base_position in base_positions
        for position in range(length)
    ]
    rows = [int(row_id) for row_id in row_ids for _ in range(length)]
    return sampler.sample_target(
        flat_logprobs,
        row_ids=rows,
        positions=positions,
    ).reshape(batch, length)


def _dflash_target_logprobs(logits: mx.array) -> mx.array:
    return mx.stack(
        [
            row - mx.logsumexp(row, axis=-1, keepdims=True)
            for row in (logits[:, position, :] for position in range(logits.shape[1]))
        ],
        axis=1,
    )


# Drafters that set ``records_draft_logits`` keep the logits they sampled
# their proposals from, so a sampled round can use speculative (rejection)
# sampling: accept draft x with probability min(1, p(x) / q(x)) and, on the
# first rejection, draw from the normalized residual max(0, p - q). The
# committed tokens then follow the target's filtered distribution exactly,
# while acceptance is 1 - TV(p, q) instead of the chance that two independent
# (or key-coupled) draws agree. Set MLX_VLM_DSPARK_REJECTION=0 to disable.
_REJECTION_ENV = "MLX_VLM_DSPARK_REJECTION"
_ACCEPT_SALT = 0x5A17ACC3
_RESIDUAL_SALT = 0x2E51D0A1


def _rejection_enabled() -> bool:
    return os.environ.get(_REJECTION_ENV, "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _rejection_params(sampler: Callable) -> Optional[Tuple[float, float, int]]:
    """(temperature, top_p, top_k) when the sampler's law is reproducible here."""
    temperature = getattr(sampler, "temperature", None)
    if temperature is None or not _supports_positioned_target_sampling(sampler):
        return None
    if not getattr(sampler, "_top_p_after_temperature", True):
        return None  # top-n-sigma / p-less / typical filters: keep the old walk
    if float(temperature) <= 0:
        return None
    return (
        float(temperature),
        float(getattr(sampler, "top_p", 1.0)),
        int(getattr(sampler, "top_k", 0) or 0),
    )


def _filtered_logprobs(
    logits: mx.array, temperature: float, top_p: float, top_k: int
) -> mx.array:
    """Normalized log-law that the positioned samplers draw from."""
    x = logits.astype(mx.float32)
    if top_k > 0:
        x = apply_top_k(x, min(top_k, x.shape[-1]))
    x = x / temperature
    x = x - mx.logsumexp(x, axis=-1, keepdims=True)
    if 0 < top_p < 1:
        x = apply_top_p(x, top_p)
        x = x - mx.logsumexp(x, axis=-1, keepdims=True)
    return x


def _mix32(seed: int, row: int, position: int) -> int:
    x = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    x = (x + (int(row) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ ((int(position) + 1) * 0xC2B2AE35)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return int(x & 0xFFFFFFFF)


def _round_keys(seed, salt, row_ids, base_positions, length):
    return mx.stack(
        [
            mx.random.key(_mix32(int(seed) ^ salt, row, base + i))
            for row, base in zip(row_ids, base_positions)
            for i in range(length)
        ]
    )


def take_draft_logits(draft_model, rows: int, count: int) -> Optional[mx.array]:
    """Pop the logits a recording drafter sampled this round's rows from."""
    log = getattr(draft_model, "_draft_logits_log", None)
    if log is None:
        return None
    draft_model._draft_logits_log = None
    if not log:
        return None
    logits = mx.concatenate(log, axis=0) if len(log) > 1 else log[0]
    if logits.ndim != 3 or logits.shape[0] != rows or logits.shape[1] != count:
        return None
    return logits


def begin_draft_logits(draft_model, sampler: Callable, greedy: bool) -> None:
    if (
        not greedy
        and getattr(draft_model, "records_draft_logits", False)
        and _rejection_enabled()
        and _rejection_params(sampler) is not None
    ):
        draft_model._draft_logits_log = []
    else:
        draft_model._draft_logits_log = None


def _rejection_dflash_target_walk(
    logits: mx.array,
    draft_tokens: mx.array,
    draft_logits: mx.array,
    params: Tuple[float, float, int],
    budgets: List[int],
    *,
    row_ids: List[int],
    base_positions: List[int],
    seed: Optional[int] = None,
) -> Tuple[List[int], List[List[int]]]:
    temperature, top_p, top_k = params
    batch, length, vocab = logits.shape
    count = int(draft_tokens.shape[1])
    p = _filtered_logprobs(logits, temperature, top_p, top_k)  # [B, L, V]
    q = draft_logits
    if q.shape[-1] < vocab:
        pad = mx.full((*q.shape[:-1], vocab - q.shape[-1]), -mx.inf, q.dtype)
        q = mx.concatenate([q, pad], axis=-1)
    elif q.shape[-1] > vocab:
        q = q[..., :vocab]
    q = _filtered_logprobs(q, temperature, top_p, top_k)  # [B, count, V]

    fallback_lp = p
    if count:
        ids = draft_tokens.astype(mx.int32)[..., None]
        p_x = mx.exp(mx.take_along_axis(p[:, :count], ids, axis=-1)[..., 0])
        q_x = mx.exp(mx.take_along_axis(q, ids, axis=-1)[..., 0])
        if seed is None:
            u = mx.random.uniform(shape=(batch, count))
        else:
            keys = _round_keys(seed, _ACCEPT_SALT, row_ids, base_positions, count)
            u = mx.vmap(lambda k: mx.random.uniform(key=k))(keys).reshape(batch, count)
        accept = u * q_x < p_x
        residual = mx.maximum(mx.exp(p[:, :count]) - mx.exp(q), 0.0)
        mass = residual.sum(axis=-1, keepdims=True)
        residual_lp = mx.where(
            mass > 0,
            mx.log(residual) - mx.log(mx.maximum(mass, 1e-30)),
            p[:, :count],
        )
        fallback_lp = mx.concatenate([residual_lp, p[:, count:]], axis=1)
    flat = fallback_lp.reshape(batch * length, vocab)
    if seed is None:
        fallback = mx.random.categorical(flat)
    else:
        keys = _round_keys(seed, _RESIDUAL_SALT, row_ids, base_positions, length)
        fallback = mx.vmap(lambda lp, k: mx.random.categorical(lp, key=k))(flat, keys)
    fallback = fallback.reshape(batch, length)
    if count:
        mx.eval(accept, fallback)
        accept_rows = accept.tolist()
    else:
        mx.eval(fallback)
        accept_rows = [[] for _ in range(batch)]
    draft_rows = draft_tokens.tolist()
    fallback_rows = fallback.tolist()

    accepted, new_tokens = [], []
    for row, budget in enumerate(budgets):
        n = 0
        while n < count and accept_rows[row][n]:
            n += 1
        tokens = draft_rows[row][:n]
        if n < length and len(tokens) < budget:
            tokens.append(int(fallback_rows[row][n]))
        accepted.append(n)
        new_tokens.append(tokens[:budget])
    return accepted, new_tokens


def _sample_dflash_target_walk(
    logits: mx.array,
    draft_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    budgets: List[int],
    *,
    row_ids: List[int],
    base_positions: List[int],
    draft_logits: Optional[mx.array] = None,
) -> Tuple[List[int], List[List[int]]]:
    params = _rejection_params(sampler) if draft_logits is not None else None
    if params is not None and _rejection_enabled():
        return _rejection_dflash_target_walk(
            logits,
            draft_tokens,
            draft_logits,
            params,
            budgets,
            row_ids=row_ids,
            base_positions=base_positions,
            seed=getattr(sampler, "seed", None),
        )
    if _supports_positioned_target_sampling(sampler):
        target_tokens = _sample_dflash_target_block(
            logits,
            sampler,
            row_ids=row_ids,
            base_positions=base_positions,
        )
        mx.async_eval(target_tokens)
        return _speculative_walk_batch(draft_tokens, target_tokens, budgets)

    batch, length, _ = logits.shape
    draft_count = int(draft_tokens.shape[1])
    draft_rows = draft_tokens.tolist()
    logprobs = _dflash_target_logprobs(logits)

    for position in range(length):
        target_tokens = sampler(logprobs[:, position, :])
        mx.eval(target_tokens)
        target_rows = [int(token) for token in target_tokens.reshape(-1).tolist()]
        if position < draft_count and all(
            target_rows[row] == draft_rows[row][position] for row in range(batch)
        ):
            continue

        new_tokens = []
        for row, budget in enumerate(budgets):
            tokens = draft_rows[row][:position]
            if len(tokens) < budget:
                tokens.append(target_rows[row])
            new_tokens.append(tokens[:budget])
        return [position] * batch, new_tokens

    return [draft_count] * batch, [
        draft_rows[row][: budgets[row]] for row in range(batch)
    ]


def unpad_hidden(hidden, cache):
    """Return real target context only; padding has no position in a draft cache."""
    batch = hidden.shape[0]
    lengths = [hidden.shape[1]] * batch
    offsets = next(
        (c.offset for c in cache if getattr(c, "offset", None) is not None), None
    )
    if isinstance(offsets, mx.array) and offsets.size == batch:
        lengths = [
            min(hidden.shape[1], max(0, n)) for n in offsets.reshape(-1).tolist()
        ]
    return [
        hidden[i : i + 1, hidden.shape[1] - length :]
        for i, length in enumerate(lengths)
    ]


class DFlashBatch:
    """Own each request's draft state and one shared, mutable target batch.

    A step completes its cache transaction before returning. Admission and
    cancellation can therefore change the batch at every step boundary.
    All device work stays on the caller's serving stream, including drafting,
    verification, and rollback, to preserve distributed collective ordering.
    """

    def __init__(
        self, model, draft, cache, hidden, bonus, block_size, sampler, dtype, greedy
    ):
        self.model = getattr(model, "language_model", model)
        self.draft = draft
        self.cache = cache
        self.hidden = unpad_hidden(hidden, cache)
        self.bonus = bonus.reshape(-1).tolist()
        self.block_size = _dflash_block_total(draft, block_size)
        choose_ceiling = getattr(draft, "choose_block_ceiling", None)
        if callable(choose_ceiling):
            self.block_size = choose_ceiling(int(hidden.shape[1]), self.block_size)
        self.initial_block_size = getattr(draft, "dflash_initial_block_size", None)
        choose_initial = getattr(draft, "choose_initial_block_size", None)
        if callable(choose_initial):
            self.initial_block_size = choose_initial(
                int(hidden.shape[1]), self.block_size
            )
        self.sampler = sampler
        self.dtype = dtype
        self.greedy = greedy
        self.positioned_sampling = _supports_positioned_target_sampling(sampler)
        draft.reset(model)
        self.sampler_rng = _SpeculativeSamplerRNG(
            draft, enabled=not greedy and not self.positioned_sampling
        )
        self.draft_caches = [draft.make_cache() for _ in self.bonus]

    @property
    def state(self):
        # Include drafter buffers in the server's continuation memory budget.
        return self.cache, self.hidden, self.draft_caches

    def snapshot(self, index):
        """Own a single row at a completed verification boundary."""
        caches = apc.snapshot_prompt_cache_row(self.cache, index)
        if caches is None:
            raise ValueError("This model does not support continuation cache snapshots")
        offsets = next(
            (c.offset for c in self.cache if getattr(c, "offset", None) is not None), 0
        )
        length = (
            int(offsets.reshape(-1)[index].item())
            if isinstance(offsets, mx.array)
            else int(offsets)
        )
        caches, _ = apc.make_warm_batch_exact_cache_multi([caches], [length])
        if caches is None:
            raise ValueError("This model does not support continuation cache merging")
        row = copy.copy(self)
        row.cache = caches
        row.hidden = [mx.contiguous(self.hidden[index])]
        row.bonus = [self.bonus[index]]
        # These objects belong exclusively to this row; the caller removes it
        # from the live batch immediately after taking the snapshot.
        row.draft_caches = [self.draft_caches[index]]
        row.sampler_rng = copy.copy(self.sampler_rng)
        mx.eval(row.hidden, [c.state for c in row.draft_caches[0]])
        return row

    def extend(self, other):
        # Finish pending rollback copies before mutating batch layouts.
        mx.eval(
            [c.state for c in self.cache],
            [c.state for c in other.cache],
            self.hidden,
            other.hidden,
        )
        for current, incoming in zip(self.cache, other.cache):
            current.extend(incoming)
        self.hidden.extend(other.hidden)
        self.bonus.extend(other.bonus)
        self.draft_caches.extend(other.draft_caches)

    def filter(self, keep):
        if len(keep) == len(self.bonus):
            return
        if keep:
            mx.eval([c.state for c in self.cache], self.hidden)
            indices = mx.array(keep, mx.int32)
            for cache in self.cache:
                cache.filter(indices)
        else:
            self.cache.clear()
        self.hidden = [self.hidden[i] for i in keep]
        self.bonus = [self.bonus[i] for i in keep]
        self.draft_caches = [self.draft_caches[i] for i in keep]

    def step(self, budgets, positions, thinking_budget_criteria=None):
        if not budgets or min(budgets) <= 0:
            raise ValueError("Only live requests with remaining tokens can decode")
        block_size = _dflash_next_block_size(
            self.draft, self.block_size, max(budgets) + 1, self.initial_block_size
        )
        sampler = self.sampler
        row_ids = [0] * len(budgets)
        if not self.greedy and self.positioned_sampling:
            sampler = _PositionedDraftSampler(
                sampler, row_ids=row_ids, positions=positions
            )

        def draft_rows():
            draft_batch = (
                getattr(self.draft, "draft_batch_greedy", None) if self.greedy else None
            ) or getattr(self.draft, "draft_batch", None)
            if callable(draft_batch):
                return draft_batch(
                    self.bonus,
                    self.hidden,
                    self.draft_caches,
                    block_size,
                    sampler,
                    self.dtype,
                )
            draft_block = (
                getattr(self.draft, "draft_block_greedy", self.draft.draft_block)
                if self.greedy
                else self.draft.draft_block
            )
            return mx.concatenate(
                [
                    draft_block(
                        bonus,
                        hidden,
                        cache,
                        block_size,
                        (
                            _PositionedDraftSampler(
                                self.sampler, row_ids=[0], positions=[position]
                            )
                            if not self.greedy and self.positioned_sampling
                            else self.sampler
                        ),
                        self.dtype,
                    )
                    for bonus, hidden, cache, position in zip(
                        self.bonus, self.hidden, self.draft_caches, positions
                    )
                ]
            )

        begin_draft_logits(self.draft, self.sampler, self.greedy)
        draft_tokens = self.sampler_rng.draft_tokens(draft_rows)
        draft_logits = take_draft_logits(
            self.draft, len(budgets), int(draft_tokens.shape[1])
        )
        inputs = mx.concatenate(
            [mx.array(self.bonus, self.dtype)[:, None], draft_tokens], axis=1
        )
        transaction = None
        try:
            if self.greedy:
                captured, transaction, target_tokens = _dflash_verify_greedy(
                    self.model,
                    inputs,
                    self.cache,
                    list(self.draft.config.target_layer_ids),
                    self.sampler,
                )
            else:
                output, transaction = _dflash_verify(
                    self.model,
                    inputs,
                    self.cache,
                    list(self.draft.config.target_layer_ids),
                )
                captured = output.hidden_states
            hidden = mx.concatenate(captured, axis=-1)
            if self.greedy:
                mx.async_eval(target_tokens, hidden)
                accepted, tokens = _speculative_walk_batch(
                    draft_tokens, target_tokens, budgets
                )
            else:
                mx.async_eval(output.logits, hidden)
                accepted, tokens = _sample_dflash_target_walk(
                    output.logits,
                    draft_tokens,
                    self.sampler,
                    budgets,
                    row_ids=row_ids,
                    base_positions=positions,
                    draft_logits=draft_logits,
                )
                self.sampler_rng.target_sampled(sync_draft=not self.positioned_sampling)
            # A clipped final round must never commit past the emitted prefix.
            accepted = [min(n, len(row) - 1) for n, row in zip(accepted, tokens)]
            uniform = _requires_uniform_batch_acceptance(self.draft, self.model)
            if uniform:
                count = min(accepted)
                tokens = [row[: count + 1] for row in tokens]
                accepted = [count] * len(tokens)
            accepted, tokens = _apply_thinking_budget(
                accepted,
                tokens,
                thinking_budget_criteria,
                list(range(len(tokens))),
                uniform=uniform,
            )
            commit_speculative_round(
                self.model, self.cache, transaction, accepted, block_size
            )
        except BaseException:
            abort_speculative_round(transaction)
            raise
        self.hidden = [hidden[i : i + 1, : len(row)] for i, row in enumerate(tokens)]
        self.bonus = [row[-1] for row in tokens]
        for count in accepted:
            _record_speculative_round(self.draft, count, block_size - 1)
        return tokens


def _dflash_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    use_model_initial_block_size: bool = True,
    greedy_sampling: bool = True,
    thinking_budget_criteria: Optional[Any] = None,
) -> Generator[Tuple[int, None], None, None]:
    """DFlash speculative-decoding **round loop**.

    draft → verify → walk → rollback. ``generate_step`` is responsible
    for prefill, sampling the first bonus token, and packaging the
    captured hidden states into ``hidden``.
    """
    lm = model.language_model if hasattr(model, "language_model") else model

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    choose_block_ceiling = getattr(draft_model, "choose_block_ceiling", None)
    if callable(choose_block_ceiling):
        block_total = choose_block_ceiling(int(hidden.shape[1]), block_total)
    initial_block_size = getattr(draft_model, "dflash_initial_block_size", None)
    choose_initial_block_size = getattr(draft_model, "choose_initial_block_size", None)
    if use_model_initial_block_size and callable(choose_initial_block_size):
        initial_block_size = choose_initial_block_size(
            int(hidden.shape[1]), block_total
        )
    draft_cache = draft_model.reset(model)
    with mx.stream(generation_stream):
        _reserve_dflash_target_cache(prompt_cache, block_total)
    positioned_sampling = _supports_positioned_target_sampling(sampler)
    sampler_rng = _SpeculativeSamplerRNG(
        draft_model,
        enabled=not greedy_sampling and not positioned_sampling,
    )
    prepare_target_hidden = getattr(draft_model, "prepare_target_hidden", None)
    hidden_is_prepared = callable(prepare_target_hidden)
    if hidden_is_prepared:
        hidden = prepare_target_hidden(hidden)
        mx.async_eval(hidden)

    if thinking_budget_criteria is not None:
        thinking_budget_criteria(first_bonus)
    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    auto = None
    if (
        _auto_off_enabled()
        and getattr(draft_model, "supports_auto_off", False)
        and not hidden_is_prepared
        and thinking_budget_criteria is None
        and (greedy_sampling or positioned_sampling)
    ):
        auto = _DraftAutoOff()
    round_start = time.perf_counter()

    while emitted < max_tokens:
        if auto is not None:
            span = auto.plain_span(emitted, max_tokens)
            if span:
                block = _plain_decode_block(
                    lm,
                    prompt_cache,
                    b,
                    hidden,
                    span,
                    target_layer_ids,
                    sampler,
                    greedy_sampling,
                    emitted,
                    token_dtype,
                )
                count = 0
                for tok in block.tokens():
                    yield tok, None
                    emitted += 1
                    count += 1
                    if emitted >= max_tokens:
                        return
                b, hidden = block.last, block.hidden()
                auto.plain_done(*block.timing())
                round_start = time.perf_counter()
                continue

        bs = _dflash_next_block_size(
            draft_model,
            block_total,
            max_tokens - emitted + 1,
            (initial_block_size if use_model_initial_block_size else None),
        )
        if bs <= 1:
            break

        draft_kwargs = {"target_hidden_prepared": True} if hidden_is_prepared else {}
        draft_sampler = (
            _PositionedDraftSampler(
                sampler,
                row_ids=[0],
                positions=[emitted],
            )
            if not greedy_sampling and positioned_sampling
            else sampler
        )
        draft_fn = (
            getattr(draft_model, "draft_block_greedy", draft_model.draft_block)
            if greedy_sampling
            else draft_model.draft_block
        )
        begin_draft_logits(draft_model, sampler, greedy_sampling)
        draft_tokens = sampler_rng.draft_tokens(
            draft_fn,
            b,
            hidden,
            draft_cache,
            bs,
            draft_sampler,
            token_dtype,
            **draft_kwargs,
        )
        mx.async_eval(draft_tokens)

        gdn_states = None
        try:
            with mx.stream(generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[b]], dtype=token_dtype), draft_tokens],
                    axis=1,
                )
                if greedy_sampling:
                    captured, gdn_states, target_tokens = _dflash_verify_greedy(
                        lm,
                        verify_input,
                        prompt_cache,
                        target_layer_ids,
                        sampler,
                    )
                    hidden = mx.concatenate(captured, axis=-1)
                else:
                    verify_out, gdn_states = _dflash_verify(
                        lm, verify_input, prompt_cache, target_layer_ids
                    )
                    hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            if greedy_sampling:
                mx.async_eval(target_tokens, hidden)
            else:
                mx.async_eval(hidden)

            if greedy_sampling:
                accepted, new_tokens = _speculative_walk(
                    draft_tokens, target_tokens, max_tokens - emitted
                )
            else:
                accepted_list, new_tokens_list = _sample_dflash_target_walk(
                    verify_out.logits,
                    draft_tokens,
                    sampler,
                    [max_tokens - emitted],
                    row_ids=[0],
                    base_positions=[emitted],
                    draft_logits=take_draft_logits(
                        draft_model, 1, int(draft_tokens.shape[1])
                    ),
                )
                accepted = accepted_list[0]
                new_tokens = new_tokens_list[0]
                sampler_rng.target_sampled(sync_draft=not positioned_sampling)
            [accepted], [new_tokens] = _apply_thinking_budget(
                [accepted], [new_tokens], [thinking_budget_criteria], [0]
            )
            _record_speculative_round(draft_model, accepted, bs - 1)

            if accepted < bs - 1:
                hidden = hidden[:, : accepted + 1, :]
            b = new_tokens[-1] if new_tokens else b

            with mx.stream(generation_stream):
                commit_speculative_round(lm, prompt_cache, gdn_states, accepted, bs)

        except BaseException:
            abort_speculative_round(gdn_states)
            raise

        if hidden_is_prepared and emitted + len(new_tokens) < max_tokens:
            hidden = prepare_target_hidden(hidden)
            mx.async_eval(hidden)

        # Emit after scheduling the next context projection so its execution
        # can overlap server-side detokenization and response handling.
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return
        if auto is not None:
            now = time.perf_counter()
            auto.spec_done(len(new_tokens), now - round_start)
            round_start = now

        verify_out = None


_AUTO_OFF_ENV = "MLX_VLM_DSPARK_AUTO_OFF"


def _auto_off_enabled() -> bool:
    return os.environ.get(_AUTO_OFF_ENV, "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


class _DraftAutoOff:
    """Turn drafting off while plain decode is faster, from on-line timings.

    Both rates are tokens per wall second, measured the same way (including the
    caller's per-token work). Drafting starts on; after ``WARM`` tokens and
    ``SPEC_ROUNDS`` timed rounds a short plain span measures plain decode. While
    plain is faster, plain spans grow (``SPAN`` doubling up to ``SPAN_MAX``
    tokens) with ``PROBE`` drafting rounds between them, judged on those rounds
    alone, so a stretch where drafting pays (code, a rewrite) turns it back on.
    Drafting is judged on its last ``WINDOW`` rounds.
    """

    WARM = 32
    SPEC_ROUNDS = 4
    PROBE = 3
    WINDOW = 6
    FIRST = 16
    SPAN = 32
    SPAN_MAX = 256
    MARGIN = 0.97

    def __init__(self):
        self.rounds = []  # (tokens, seconds) since the last plain span
        self.plain_rate = None
        self.span = self.SPAN
        self.skipped_first = False

    def spec_done(self, tokens, seconds):
        if not self.skipped_first:  # the first round includes compile time
            self.skipped_first = True
            return
        self.rounds.append((int(tokens), float(seconds)))
        del self.rounds[: -self.WINDOW]

    def spec_rate(self):
        seconds = sum(r[1] for r in self.rounds)
        return sum(r[0] for r in self.rounds) / seconds if seconds > 0 else None

    def plain_span(self, emitted, max_tokens):
        left = max_tokens - emitted
        if left < 8 or emitted < self.WARM:
            return 0
        need = self.SPEC_ROUNDS if self.plain_rate is None else self.PROBE
        if len(self.rounds) < need:
            return 0
        if self.plain_rate is None:
            return min(self.FIRST, left)
        rate = self.spec_rate()
        if rate >= self.plain_rate * self.MARGIN:
            self.span = self.SPAN
            return 0
        span = min(self.span, left)
        self.span = min(self.span * 2, self.SPAN_MAX)
        return span

    def plain_done(self, tokens, seconds):
        if tokens > 1 and seconds > 0:
            rate = tokens / seconds
            self.plain_rate = (
                rate if self.plain_rate is None else 0.5 * (self.plain_rate + rate)
            )
        self.rounds = []


class _plain_decode_block:
    """Plain one-token target steps (pipelined one step ahead), keeping the
    drafter's captures for every token so drafting can resume afterwards."""

    def __init__(
        self, lm, cache, bonus, hidden, count, taps, sampler, greedy, emitted, dtype
    ):
        self.lm, self.cache, self.taps = lm, cache, taps
        self.sampler, self.greedy, self.dtype = sampler, greedy, dtype
        self.count, self.position = int(count), int(emitted)
        self.captures = [hidden]
        self.last = int(bonus)
        self._first = mx.array([[int(bonus)]], dtype)
        self.t_first = self.t_last = None
        self.done = 0

    def timing(self):
        """(tokens, seconds) after the first token, or (0, 0.0)."""
        if self.done < 2:
            return 0, 0.0
        return self.done - 1, self.t_last - self.t_first

    def _step(self, token, position):
        with mx.stream(generation_stream):
            out = self.lm(token, cache=self.cache, capture_layer_ids=self.taps)
            logits = out.logits[:, -1]
            if self.greedy:
                nxt = mx.argmax(logits, axis=-1)
            else:
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                nxt = self.sampler.sample_target(
                    logprobs, row_ids=[0], positions=[position]
                )
            nxt = nxt.reshape(1, 1).astype(self.dtype)
            captured = mx.concatenate(out.hidden_states, axis=-1)
        mx.async_eval(nxt, captured)
        return nxt, captured

    def tokens(self):
        cur, cap = self._step(self._first, self.position)
        for i in range(self.count):
            ahead = None
            if i + 1 < self.count:
                ahead = self._step(cur, self.position + i + 1)
            tok = int(cur.item())
            now = time.perf_counter()
            if self.t_first is None:
                self.t_first = now  # the first step may include kernel compiles
            self.t_last, self.done = now, i + 1
            self.captures.append(cap)
            self.last = tok
            yield tok
            if ahead is None:
                break
            cur, cap = ahead

    def hidden(self):
        return mx.concatenate(self.captures, axis=1)


def _dflash_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    greedy_sampling: bool = True,
    row_ids: Optional[List[int]] = None,
    thinking_budget_criteria: Optional[List[Any]] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DFlash speculative-decoding round loop (B > 1).

    Supports continuous batching: when a sequence finishes (EOS or
    max_tokens), it is filtered out of the target caches and the
    drafter cache is reinitialized for the new batch size.

    ``stop_check(seq_idx, token_id) -> bool`` is an optional callback
    that returns True to stop a sequence (e.g. EOS detection).

    Yields ``(tokens_list, None)`` where ``tokens_list[i]`` is the
    token for sequence ``i`` (or ``None`` if that sequence has nothing
    to emit this step).
    """
    lm = model.language_model if hasattr(model, "language_model") else model

    B = first_bonus.shape[0]
    row_ids = list(range(B)) if row_ids is None else list(row_ids)
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    choose_block_ceiling = getattr(draft_model, "choose_block_ceiling", None)
    if callable(choose_block_ceiling):
        block_total = choose_block_ceiling(int(hidden.shape[1]), block_total)
    initial_block_size = getattr(draft_model, "dflash_initial_block_size", None)
    choose_initial_block_size = getattr(draft_model, "choose_initial_block_size", None)
    if callable(choose_initial_block_size):
        initial_block_size = choose_initial_block_size(
            int(hidden.shape[1]), block_total
        )
    draft_model.reset(model)
    with mx.stream(generation_stream):
        _reserve_dflash_target_cache(prompt_cache, block_total)
    positioned_sampling = _supports_positioned_target_sampling(sampler)
    sampler_rng = _SpeculativeSamplerRNG(
        draft_model,
        enabled=not greedy_sampling and not positioned_sampling,
    )
    draft_caches = [draft_model.make_cache() for _ in range(B)]

    # Per-sequence state tracked by ORIGINAL index so the caller sees
    # stable indices in the yielded token lists.
    b = first_bonus.tolist()  # active bonus tokens
    for state, token in zip(thinking_budget_criteria or [], b):
        if state is not None:
            state(token)
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))  # maps active-slot → original-index
    hidden_by_orig = [hidden[i : i + 1] for i in range(B)]
    draft_batch = (
        getattr(draft_model, "draft_batch_greedy", None) if greedy_sampling else None
    ) or getattr(draft_model, "draft_batch", None)
    if callable(draft_batch):
        offsets = getattr(prompt_cache[0], "offset", None)
        if isinstance(offsets, mx.array) and offsets.size == B:
            lengths = offsets.reshape(-1).tolist()
            hidden_by_orig = [
                h[:, -min(h.shape[1], n) :] for h, n in zip(hidden_by_orig, lengths)
            ]

    total_emitted = sum(emitted)

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = _dflash_next_block_size(
            draft_model,
            block_total,
            min(remaining),
            initial_block_size,
        )
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft rowwise: the DFlash drafter cache is scalar-offset and has
        # proven unsafe as a single batched cache on MLX/Metal. Target verify
        # remains batched below.
        def draft_active_rows():
            if callable(draft_batch) and greedy_sampling:
                return draft_batch(
                    b_active,
                    [hidden_by_orig[i] for i in active_idx],
                    [draft_caches[i] for i in active_idx],
                    bs,
                    sampler,
                    token_dtype,
                )
            return mx.concatenate(
                [
                    (
                        getattr(
                            draft_model,
                            "draft_block_greedy",
                            draft_model.draft_block,
                        )
                        if greedy_sampling
                        else draft_model.draft_block
                    )(
                        int(b_active[j]),
                        hidden_by_orig[active_idx[j]],
                        draft_caches[active_idx[j]],
                        bs,
                        (
                            _PositionedDraftSampler(
                                sampler,
                                row_ids=[row_ids[active_idx[j]]],
                                positions=[emitted[active_idx[j]]],
                            )
                            if not greedy_sampling and positioned_sampling
                            else sampler
                        ),
                        token_dtype,
                    )
                    for j in range(n_active)
                ],
                axis=0,
            )

        # Keep draft collectives on the verifier stream so distributed ranks
        # cannot submit draft and target reductions in different orders.
        with mx.stream(generation_stream):
            begin_draft_logits(draft_model, sampler, greedy_sampling)
            draft_tokens = sampler_rng.draft_tokens(draft_active_rows)
            draft_logits = take_draft_logits(
                draft_model, n_active, int(draft_tokens.shape[1])
            )

        gdn_states = None
        try:
            with mx.stream(generation_stream):
                verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
                if greedy_sampling:
                    captured, gdn_states, target_tokens = _dflash_verify_greedy(
                        lm,
                        verify_input,
                        prompt_cache,
                        target_layer_ids,
                        sampler,
                    )
                    hidden_full = mx.concatenate(captured, axis=-1)
                else:
                    verify_out, gdn_states = _dflash_verify(
                        lm, verify_input, prompt_cache, target_layer_ids
                    )
                    hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            if greedy_sampling:
                mx.async_eval(target_tokens, hidden_full)
            else:
                mx.async_eval(hidden_full)

            budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
            if greedy_sampling:
                accepted_list, new_tokens_list = _speculative_walk_batch(
                    draft_tokens, target_tokens, budgets
                )
            else:
                accepted_list, new_tokens_list = _sample_dflash_target_walk(
                    verify_out.logits,
                    draft_tokens,
                    sampler,
                    budgets,
                    row_ids=[row_ids[active_idx[j]] for j in range(n_active)],
                    base_positions=[emitted[active_idx[j]] for j in range(n_active)],
                    draft_logits=draft_logits,
                )
                sampler_rng.target_sampled(sync_draft=not positioned_sampling)

            if (
                n_active > 1
                and _requires_uniform_batch_acceptance(draft_model, lm)
                and len(set(accepted_list)) > 1
            ):
                uniform = min(len(nt) - 1 for nt in new_tokens_list)
                new_tokens_list = [nt[: uniform + 1] for nt in new_tokens_list]
                accepted_list = [uniform] * n_active

            accepted_list, new_tokens_list = _apply_thinking_budget(
                accepted_list,
                new_tokens_list,
                thinking_budget_criteria,
                active_idx,
                uniform=_requires_uniform_batch_acceptance(draft_model, lm),
            )
            hidden_segments = _dflash_committed_hidden_segments(
                hidden_full, new_tokens_list
            )
            for j in range(n_active):
                orig = active_idx[j]
                if hidden_segments[j].shape[1] > 0:
                    hidden_by_orig[orig] = hidden_segments[j]

            for a in accepted_list:
                _record_speculative_round(draft_model, a, bs - 1)

            with mx.stream(generation_stream):
                commit_speculative_round(
                    lm, prompt_cache, gdn_states, mx.array(accepted_list), bs
                )

        except BaseException:
            abort_speculative_round(gdn_states)
            raise

        # Emit (map active slots back to original indices)
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        # --- Continuous batching: filter out finished sequences ---
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            if len(keep_slots) == 0:
                break
            # Filter target caches (BatchKVCache supports this)
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_mx)
            _filter_rope_deltas(lm, keep_mx, n_active)
            # Update active index mapping
            active_idx = [active_idx[j] for j in keep_slots]

        verify_out = None
        total_emitted = sum(emitted)
