"""
test_empty_class.py -- checks the empty-class convention of Algorithm 1 (v2.0.1).
Run:  python test_empty_class.py
"""
import numpy as np
import dmm_lib_v2 as L

def test_empty_class():
    M = L.synthetic_model(10, [5]*10, 3, list(range(4)), 0.5, seed=1)
    C, X, y = L.sample(M, 2000, np.random.default_rng(5))
    tr = np.flatnonzero(y != 2)                     # class 2 absent from the training sample
    emb = L.DMMEmbed(3).fit(X[tr], y[tr], 3, 10)
    for use_priors in (True, False):                # MAP and ML rules
        clf = L.LatentKDE(use_priors=use_priors).fit(emb.transform(X[tr]), y[tr], 3)
        S = clf.scores(emb.transform(X))
        assert np.all(np.isneginf(S[:, 2])), "empty class must receive log-score -inf"
        assert np.all(np.isfinite(S[:, :2])), "non-empty classes must have finite log-scores"
        assert not np.any(clf.predict(emb.transform(X)) == 2), "empty class must never be predicted"
    Z = L.DCAEmbed(3).fit(X[tr], y[tr], 3, 10).transform(X)
    assert np.all(np.isfinite(Z)), "DCA must be well defined with an empty class"
    res = L.run_methods(X[tr], y[tr], C[tr], X, y, C, 3, 10, M.m, 3,
                        methods=["DMM+KDE", "DMM-CN+KDE", "DCA+KDE", "HellPCA+KDE", "MCA+KDE"])
    assert all(np.isfinite(v["accuracy"]) for v in res.values())

if __name__ == "__main__":
    test_empty_class(); print("empty-class convention: OK")
