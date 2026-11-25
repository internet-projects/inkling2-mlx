import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.ar import _PositionedTargetSampler
from mlx_vlm.speculative import dflash as D

N = 4000


def _law(logits, temperature, top_p, top_k):
    return np.exp(
        np.array(D._filtered_logprobs(mx.array(logits), temperature, top_p, top_k))
    )


def _setup(temperature, top_p, top_k, count=2, vocab=6, seed=3):
    rng = np.random.default_rng(seed)
    p_logits = rng.normal(size=(count + 1, vocab)).astype(np.float32) * 1.5
    q_logits = (p_logits[:count] + rng.normal(size=(count, vocab)) * 0.8).astype(
        np.float32
    )
    q_law = _law(q_logits, temperature, top_p, top_k)
    draft = np.stack(
        [rng.choice(vocab, size=N, p=q_law[i] / q_law[i].sum()) for i in range(count)],
        axis=1,
    )
    logits = mx.broadcast_to(mx.array(p_logits)[None], (N, count + 1, vocab))
    dlogits = mx.broadcast_to(mx.array(q_logits)[None], (N, count, vocab))
    return p_logits, logits, mx.array(draft, mx.int32), dlogits


@pytest.mark.parametrize(
    "temperature,top_p,top_k,seeded",
    [(1.0, 1.0, 0, False), (0.7, 0.9, 0, True), (1.0, 1.0, 4, True)],
)
def test_rejection_walk_keeps_target_law(temperature, top_p, top_k, seeded):
    mx.random.seed(0)
    p_logits, logits, draft, dlogits = _setup(temperature, top_p, top_k)
    accepted, tokens = D._rejection_dflash_target_walk(
        logits,
        draft,
        dlogits,
        (temperature, top_p, top_k),
        [8] * N,
        row_ids=list(range(N)),
        base_positions=[0] * N,
        seed=1234 if seeded else None,
    )
    p_law = _law(p_logits, temperature, top_p, top_k)
    first = np.bincount([t[0] for t in tokens], minlength=p_law.shape[1]) / N
    assert np.abs(first - p_law[0]).max() < 0.04
    second = [t[1] for t in tokens if len(t) > 1]
    assert len(second) > N // 4
    second = np.bincount(second, minlength=p_law.shape[1]) / len(second)
    assert np.abs(second - p_law[1]).max() < 0.05
    # Acceptance is 1 - TV(p, q) at the first position.
    q_law = _law(np.array(dlogits[0]), temperature, top_p, top_k)
    expected = 1 - 0.5 * np.abs(p_law[0] - q_law[0]).sum()
    assert abs(np.mean([a >= 1 for a in accepted]) - expected) < 0.04


def test_identical_laws_accept_everything():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(4, 9)).astype(np.float32)
    draft = mx.array([[1, 2, 3]], mx.int32)
    accepted, tokens = D._rejection_dflash_target_walk(
        mx.array(p)[None],
        draft,
        mx.array(p[:3])[None],
        (1.0, 1.0, 0),
        [10],
        row_ids=[0],
        base_positions=[5],
        seed=7,
    )
    assert accepted == [3] and tokens[0][:3] == [1, 2, 3] and len(tokens[0]) == 4


def test_budget_and_switch(monkeypatch):
    sampler = _PositionedTargetSampler(temperature=1.0, top_p=1.0, seed=1)
    logits = mx.zeros((1, 3, 5))
    draft = mx.array([[1, 2]], mx.int32)
    dl = mx.zeros((1, 2, 5))
    accepted, tokens = D._sample_dflash_target_walk(
        logits, draft, sampler, [1], row_ids=[0], base_positions=[0], draft_logits=dl
    )
    assert tokens == [[1]]
    monkeypatch.setenv("MLX_VLM_DSPARK_REJECTION", "0")
    calls = []
    real = D._rejection_dflash_target_walk
    monkeypatch.setattr(
        D,
        "_rejection_dflash_target_walk",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    D._sample_dflash_target_walk(
        logits, draft, sampler, [4], row_ids=[0], base_positions=[0], draft_logits=dl
    )
    assert not calls


def test_draft_logits_log_roundtrip():
    class Draft:
        records_draft_logits = True

    sampler = _PositionedTargetSampler(temperature=0.8, top_p=0.95, seed=1)
    d = Draft()
    D.begin_draft_logits(d, sampler, greedy=False)
    assert d._draft_logits_log == []
    d._draft_logits_log.append(mx.zeros((1, 3, 7)))
    d._draft_logits_log.append(mx.zeros((1, 3, 7)))
    out = D.take_draft_logits(d, 2, 3)
    assert out.shape == (2, 3, 7) and d._draft_logits_log is None
    D.begin_draft_logits(d, sampler, greedy=True)
    assert d._draft_logits_log is None
    D.begin_draft_logits(d, lambda x: x, greedy=False)  # unknown sampler law
    assert d._draft_logits_log is None


@pytest.mark.parametrize("seed", [41, None])
def test_inkling2_sampled_rounds_use_rejection(monkeypatch, seed):
    from mlx_vlm.tests.test_speculative import generated, inkling2_dspark_case

    mx.random.seed(73)
    target, draft = inkling2_dspark_case(mx.float32)
    calls = []
    real = D._rejection_dflash_target_walk
    monkeypatch.setattr(
        D,
        "_rejection_dflash_target_walk",
        lambda *a, **k: calls.append(a[2].shape) or real(*a, **k),
    )
    tokens = generated(target, draft, temperature=0.7, seed=seed)
    assert len(tokens) == (12 if seed else 24) and calls
    assert all(shape[0] == 1 and shape[1] >= 1 for shape in calls)
    assert getattr(draft, "_draft_logits_log", None) is None
