import numpy as np
import pytest

from sabana_lc.classes import DYNAMIC_WORLD, NODATA, WORLDCOVER, fmt_area, remap
from sabana_lc.evaluate import Design, assess, proportion, ratio


def test_fmt_area_matches_house_style():
    assert fmt_area(210_067) == "21 ha + 0067 m²"
    assert fmt_area(131_352_086) == "13.135 ha + 2086 m²"
    assert fmt_area(99.6) == "0 ha + 0100 m²"
    assert fmt_area(-25_000) == "-2 ha + 5000 m²"


def test_remaps_cover_every_source_code():
    wc = remap(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0]), WORLDCOVER)
    assert wc.tolist() == [1, 2, 3, 4, 5, 6, NODATA, 0, 3, 1, 3, NODATA]
    dw = remap(np.arange(9), DYNAMIC_WORLD)
    assert dw.tolist() == [0, 1, 3, 3, 4, 2, 5, 6, NODATA]


# ---------------------------------------------------------------- Stehman (2014) estimators
def test_reduces_to_olofsson_when_strata_are_map_classes():
    """With strata = map classes, user's accuracy is n_kk/n_k. and areas follow Olofsson et al. (2014)."""
    rng = np.random.default_rng(1)
    N_h = {0: 20_000, 1: 180_000, 2: 800_000}
    strata, mapped, ref = [], [], []
    for h, n in ((0, 60), (1, 80), (2, 100)):
        strata += [h] * n
        mapped += [h] * n
        ref += list(np.where(rng.random(n) < 0.85, h, rng.integers(0, 3, n)))
    s, m, r = map(np.array, (strata, mapped, ref))
    res = assess(Design(s, N_h), r, m, pixel_area_m2=1.0)
    W = np.array([N_h[h] for h in range(3)]) / sum(N_h.values())
    n_ij = np.array([[((m == i) & (r == j)).sum() for j in range(3)] for i in range(3)], float)
    p_ij = W[:, None] * n_ij / n_ij.sum(1, keepdims=True)  # Olofsson eq. (4)
    np.testing.assert_allclose(res["error_matrix"][:3, :3], p_ij, atol=1e-12)
    ua = np.diag(n_ij) / n_ij.sum(1)
    np.testing.assert_allclose(res["per_class"]["users_acc"][:3], ua, atol=1e-12)
    pa = np.diag(p_ij) / p_ij.sum(0)
    np.testing.assert_allclose(res["per_class"]["producers_acc"][:3], pa, atol=1e-12)
    assert res["overall_acc"] == pytest.approx(np.trace(p_ij), abs=1e-12)
    # Olofsson eq. (10): SE of the area proportion of class j
    j = 0
    se = np.sqrt(sum(W[i] ** 2 * (n_ij[i, j] / n_ij[i].sum()) * (1 - n_ij[i, j] / n_ij[i].sum()) / (n_ij[i].sum() - 1)
                     for i in range(3)))
    fpc_free = proportion(Design(s, {k: v * 10**6 for k, v in N_h.items()}), (r == j).astype(int))[1]
    assert fpc_free == pytest.approx(se, rel=1e-4)


def test_estimators_are_unbiased_when_strata_differ_from_map():
    """Population known exactly; strata unrelated to the map; average estimate -> truth."""
    rng = np.random.default_rng(7)
    N = 60_000
    ref = rng.choice(3, N, p=[0.15, 0.25, 0.6])
    mapped = np.where(rng.random(N) < 0.8, ref, rng.integers(0, 3, N))
    stratum = np.where(rng.random(N) < 0.9, ref, rng.integers(0, 3, N))  # like WorldCover vs truth
    N_h = {h: int((stratum == h).sum()) for h in range(3)}
    true_oa = (ref == mapped).mean()
    true_ua1 = ((mapped == 1) & (ref == 1)).sum() / (mapped == 1).sum()
    oas, uas, z = [], [], []
    for _ in range(300):
        idx = np.concatenate([rng.choice(np.nonzero(stratum == h)[0], 50, replace=False) for h in range(3)])
        d = Design(stratum[idx], N_h)
        oa, se = proportion(d, (ref[idx] == mapped[idx]).astype(int))
        ua, _ = ratio(d, ((mapped[idx] == 1) & (ref[idx] == 1)).astype(int), (mapped[idx] == 1).astype(int))
        oas.append(oa)
        uas.append(ua)
        z.append(abs(oa - true_oa) / se)
    assert np.mean(oas) == pytest.approx(true_oa, abs=0.006)
    assert np.mean(uas) == pytest.approx(true_ua1, abs=0.01)
    assert 0.90 < np.mean(np.array(z) < 1.96) < 0.99  # 95 % intervals cover the truth ~95 % of the time
