"""
dmm_lib_v2.py (v2.0.1) -- reproducible code for the numerical section of
"Density-Matrix Spectral Embeddings for Categorical Data" (v2).

Contents
  * blockwise categorical generator (compatible with the v1 notebook generator)
  * DMM operator (count-based / class-normalized), Gram-route embedding
  * latent KDE classifier (MAP or ML), Scott bandwidth
  * baselines: Hellinger-PCA+KDE, MCA+KDE, DCA+KDE, PCA+KNN, shrinkage LDA,
    categorical naive Bayes, multinomial logistic regression, random forest
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# ----------------------------------------------------------------------------- generator
def peaked(m, peak, delta):
    p = (1.0 - delta) * np.ones(m) / m
    p[peak] += delta
    return p / p.sum()

@dataclass
class BlockModel:
    """p[b][c] = categorical law of block b in class c."""
    p: List[List[np.ndarray]]
    priors: np.ndarray
    @property
    def m(self): return [len(pb[0]) for pb in self.p]
    @property
    def q(self): return len(self.p)
    @property
    def d(self): return int(sum(self.m))
    @property
    def k(self): return len(self.priors)
    def profiles(self):
        """(d,k) matrix of one-hot profiles p_y = (1/q)(p^(1)_y,...,p^(q)_y)."""
        return np.stack([np.concatenate([pb[c] for pb in self.p]) / self.q for c in range(self.k)], 1)
    def rho_D_star(self):
        Psi = np.sqrt(self.profiles())
        return Psi @ np.diag(self.priors) @ Psi.T
    def rho_star(self):
        Psi = np.sqrt(self.profiles())
        return Psi @ Psi.T / self.k

def synthetic_model(q, m_list, k, informative, delta, noise_blocks=0, priors=None, seed=0):
    """Same construction as the v1 notebook (peaked laws on informative blocks)."""
    rng = np.random.default_rng(seed)
    p = []
    for b in range(q):
        m = m_list[b]
        if b in informative:
            peaks = rng.integers(0, m, size=k)
            p.append([peaked(m, int(peaks[c]), delta) for c in range(k)])
        else:
            p.append([np.ones(m) / m for _ in range(k)])
    for _ in range(noise_blocks):
        m = int(rng.integers(3, 10))
        p.append([np.ones(m) / m for _ in range(k)])
    pr = np.ones(k) / k if priors is None else np.asarray(priors, float) / np.sum(priors)
    return BlockModel(p, pr)

def dirichlet_model(m_list, k, alpha, priors=None, seed=0):
    """Random class profiles with Dirichlet(alpha) block laws (small alpha => rare categories)."""
    rng = np.random.default_rng(seed)
    p = [[rng.dirichlet(np.full(m, alpha)) for _ in range(k)] for m in m_list]
    pr = np.ones(k) / k if priors is None else np.asarray(priors, float) / np.sum(priors)
    return BlockModel(p, pr)

def sample(model: BlockModel, n, rng):
    """Returns codes (n,q) int, one-hot X (n,d) float32, labels y (n,)."""
    y = rng.choice(model.k, size=n, p=model.priors)
    codes = np.empty((n, model.q), dtype=np.int64)
    for b, pb in enumerate(model.p):
        for c in range(model.k):
            idx = np.flatnonzero(y == c)
            if idx.size:
                codes[idx, b] = rng.choice(len(pb[c]), size=idx.size, p=pb[c])
    return codes, onehot(codes, model.m), y

def onehot(codes, m_list):
    n, q = codes.shape
    off = np.concatenate([[0], np.cumsum(m_list)[:-1]])
    X = np.zeros((n, int(sum(m_list))), dtype=np.float32)
    X[np.repeat(np.arange(n), q), (codes + off).ravel()] = 1.0
    return X

# ----------------------------------------------------------------------------- DMM
def count_matrix(X, y, k):
    F = np.zeros((X.shape[1], k))
    for c in range(k):
        F[:, c] = X[y == c].sum(0)
    return F

def dmm_operator(X, y, k, normalized=False):
    F = count_matrix(X, y, k)
    if normalized:
        s = F.sum(0); s[s == 0] = 1.0; F = F / s
    A = np.sqrt(F)
    return A @ A.T / np.trace(A.T @ A), A

def dmm_basis(X, y, k, r, normalized=False):
    """Top-r eigenvectors of rho_D (or rho_CN) via the k x k Gram matrix."""
    F = count_matrix(X, y, k)
    if normalized:
        s = F.sum(0); s[s == 0] = 1.0; F = F / s
    A = np.sqrt(F)
    lam, V = np.linalg.eigh(A.T @ A)
    o = np.argsort(lam)[::-1][:r]
    lam, V = np.maximum(lam[o], 1e-12), V[:, o]
    return (A @ V) / np.sqrt(lam)

def top_eigvecs(M, r):
    w, V = np.linalg.eigh(M)
    return V[:, np.argsort(w)[::-1][:r]]

def sin_theta(U, V):
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))

def scott_bandwidth(Z):
    n, r = Z.shape
    return float(np.mean(Z.std(0)) * n ** (-1.0 / (r + 4)) + 1e-12)

class LatentKDE:
    """Class-conditional Gaussian KDE + MAP (use_priors=True) or ML rule (use_priors=False).

    Empty-class convention (Definition of the KDE and Algorithm 1 of the manuscript): if a class has no
    training observation (n_y = 0), its density estimate is identically zero, so its log-score is -inf under
    both the MAP and the ML rule and it is never predicted. Ties are broken by the smallest class index.
    """
    def __init__(self, bandwidth=None, use_priors=True):
        self.h, self.use_priors = bandwidth, use_priors
    def fit(self, Z, y, k):
        Z = np.asarray(Z); y = np.asarray(y)
        if Z.shape[0] == 0:
            raise ValueError("LatentKDE.fit requires at least one training observation")
        self.k = k
        self.h_ = self.h if self.h is not None else scott_bandwidth(Z)
        self.n_y = np.array([np.sum(y == c) for c in range(k)])
        self.kdes = [KernelDensity(bandwidth=self.h_).fit(Z[y == c]) if self.n_y[c] > 0 else None for c in range(k)]
        with np.errstate(divide="ignore"):
            self.logpri = np.log(self.n_y / len(Z))          # -inf for empty classes
        return self
    def scores(self, Z):
        """(n, k) matrix of log-scores; -inf for empty classes."""
        Z = np.asarray(Z)
        S = np.full((Z.shape[0], self.k), -np.inf)
        for c, kd in enumerate(self.kdes):
            if kd is not None:
                S[:, c] = kd.score_samples(Z) + (self.logpri[c] if self.use_priors else 0.0)
        return S
    def predict(self, Z):
        return self.scores(Z).argmax(1)

# ----------------------------------------------------------------------------- embeddings
class DMMEmbed:
    def __init__(self, r, normalized=False): self.r, self.normalized = r, normalized
    def fit(self, X, y, k, q):
        self.q = q; self.U = dmm_basis(X, y, k, min(self.r, k), self.normalized); return self
    def transform(self, X): return (X / np.sqrt(self.q)) @ self.U

class HellingerPCAEmbed:
    """PCA of the Hellinger-transformed individual profiles sqrt(x/q) = x/sqrt(q) (Rao 1995)."""
    def __init__(self, r): self.r = r
    def fit(self, X, y, k, q):
        self.q = q; Xt = X / np.sqrt(q); self.mu = Xt.mean(0)
        _, _, Vt = np.linalg.svd(Xt - self.mu, full_matrices=False); self.V = Vt[:self.r].T; return self
    def transform(self, X): return (X / np.sqrt(self.q) - self.mu) @ self.V

class MCAEmbed:
    """Multiple correspondence analysis of the indicator matrix; rows projected as supplementary profiles."""
    def __init__(self, r): self.r = r
    def fit(self, X, y, k, q):
        self.q = q; n = X.shape[0]; P = X / (n * q); c = P.sum(0); keep = c > 0
        self.keep = keep; c = c[keep]; rr = np.full(n, 1.0 / n)
        S = (P[:, keep] - np.outer(rr, c)) / np.sqrt(np.outer(rr, c))
        _, _, Vt = np.linalg.svd(S, full_matrices=False)
        self.G = Vt[:self.r].T / np.sqrt(c)[:, None]      # column standard coordinates
        return self
    def transform(self, X): return (X[:, self.keep] / self.q) @ self.G

class DCAEmbed:
    """Discriminant correspondence analysis: CA of the class x modality table; individuals as barycentres."""
    def __init__(self, r): self.r = r
    def fit(self, X, y, k, q):
        self.q = q; F = count_matrix(X, y, k).T
        F = F[F.sum(1) > 0]                                   # empty classes carry no mass: drop their rows
        P = F / F.sum(); rr = P.sum(1); c = P.sum(0)
        keep = c > 0; self.keep = keep; c = c[keep]; P = P[:, keep]
        S = (P - np.outer(rr, c)) / np.sqrt(np.outer(rr, c))
        _, _, Vt = np.linalg.svd(S, full_matrices=False)
        rd = max(1, min(self.r, F.shape[0] - 1)); self.G = Vt[:rd].T / np.sqrt(c)[:, None]
        return self
    def transform(self, X): return (X[:, self.keep] / self.q) @ self.G

# ----------------------------------------------------------------------------- evaluation
def metrics(yt, yp):
    return dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                balanced_acc=balanced_accuracy_score(yt, yp))

def run_methods(Xtr, ytr, Ctr, Xte, yte, Cte, k, q, m_list, r, methods=None, seed=0, use_priors=True):
    """Fit/evaluate all methods on one split. C* are integer codes (for naive Bayes)."""
    out = {}
    emb = {"DMM+KDE": DMMEmbed(r), "DMM-CN+KDE": DMMEmbed(r, normalized=True),
           "HellPCA+KDE": HellingerPCAEmbed(r), "MCA+KDE": MCAEmbed(r), "DCA+KDE": DCAEmbed(r)}
    methods = methods or list(emb) + ["PCA+KNN", "LDA-shrink", "CatNB", "LogReg", "RF"]
    for name in methods:
        if name in emb:
            e = emb[name].fit(Xtr, ytr, k, q)
            clf = LatentKDE(use_priors=use_priors).fit(e.transform(Xtr), ytr, k)
            yp = clf.predict(e.transform(Xte))
        elif name == "PCA+KNN":
            svd = TruncatedSVD(n_components=min(max(10, 5 * k), Xtr.shape[1] - 1), random_state=seed)
            yp = KNeighborsClassifier(15).fit(svd.fit_transform(Xtr), ytr).predict(svd.transform(Xte))
        elif name == "LDA-shrink":
            yp = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte)
        elif name == "CatNB":
            yp = CategoricalNB(alpha=1.0, min_categories=np.asarray(m_list)).fit(Ctr, ytr).predict(Cte)
        elif name == "LogReg":
            yp = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr).predict(Xte)
        elif name == "RF":
            yp = RandomForestClassifier(300, random_state=seed, n_jobs=-1).fit(Xtr, ytr).predict(Xte)
        out[name] = metrics(yte, yp)
    return out
