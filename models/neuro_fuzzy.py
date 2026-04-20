"""
models/neuro_fuzzy.py
Model B — Sistem Neuro-Fuzzy berbasis Artificial Neural Network (ANN).

Pendekatan:
─────────────
Kami mengimplementasikan arsitektur ANFIS-inspired (Adaptive Neuro-Fuzzy
Inference System) yang disederhanakan menggunakan numpy murni agar tidak
bergantung pada library deep-learning besar.

Arsitektur (5 layer):
  Layer 1 (Fuzzification)  : setiap input diubah menjadi 3 derajat keanggotaan
                              (rendah, sedang, tinggi) menggunakan fungsi Gaussian.
  Layer 2 (Rule Firing)    : perkalian derajat keanggotaan (T-norm product)
                             menghasilkan kekuatan setiap aturan.
  Layer 3 (Normalization)  : normalisasi kekuatan aturan.
  Layer 4 (Consequence)    : setiap aturan punya parameter linear (a·x + b).
  Layer 5 (Aggregation)    : weighted sum → skor akhir (0–100).

Parameter yang dioptimasi (belajar):
  - Pusat (c) dan lebar (σ) fungsi keanggotaan Gaussian (premise params)
  - Koefisien linear di layer 4 (consequent params)

Training menggunakan simulasi preferensi pengguna (synthetic supervision)
karena tidak ada data rating eksplisit — ini adalah pendekatan umum
dalam sistem rekomendasi cold-start.

Mengapa ANN lebih baik dari FIS Manual?
  → Parameter fungsi keanggotaan dipelajari dari data, bukan ditebak pakar.
  → Dapat menangkap pola non-linear kompleks antar fitur.
  → Adaptif: jika data berubah, model bisa di-retrain.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ════════════════════════════════════════════════════════════════════════════
# KONFIGURASI ARSITEKTUR
# ════════════════════════════════════════════════════════════════════════════

# Fitur input yang digunakan
INPUT_FEATURES = [
    "energy", "danceability", "valence",
    "speechiness", "acousticness",
]
N_INPUTS    = len(INPUT_FEATURES)   # 5
N_MF        = 3                     # Jumlah MF per input (rendah/sedang/tinggi)
N_RULES     = N_MF ** 2             # Kita pakai 2 input terpenting untuk rules: 9
                                    # (dikombinasikan energy × danceability)


def _gaussian_mf(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """
    Fungsi keanggotaan Gaussian: μ(x) = exp(-((x-c)/σ)²)
    Gaussian dipilih karena smooth dan mudah didiferensiasikan (gradient-friendly).
    """
    return np.exp(-((x - c) / (sigma + 1e-9)) ** 2)


class NeuroFuzzySystem:
    """
    ANFIS-inspired Neuro-Fuzzy System yang dilatih dengan backpropagation
    sederhana menggunakan numpy.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 300):
        self.lr     = learning_rate
        self.epochs = epochs
        self._trained = False

        # ── Inisialisasi parameter premise (c dan σ untuk tiap MF) ──────────
        # Untuk N_INPUTS fitur, masing-masing punya N_MF buah MF.
        # Pusat awal: merata di [0.1, 0.5, 0.9] — intuisi: rendah/sedang/tinggi
        # Lebar awal: 0.25 — cukup lebar untuk overlap yang baik
        self.centers = np.tile(
            np.array([0.15, 0.50, 0.85]),   # 3 pusat MF
            (N_INPUTS, 1)                   # untuk setiap input
        )  # shape: (N_INPUTS, N_MF)

        # σ per MF (semua diinisialisasi sama, akan berubah saat training)
        self.sigmas = np.full((N_INPUTS, N_MF), 0.25)

        # ── Inisialisasi consequent parameters ──────────────────────────────
        # Setiap rule punya: koefisien linear untuk setiap input + bias
        # shape: (N_RULES, N_INPUTS + 1)
        np.random.seed(42)
        self.consequents = np.random.randn(N_RULES, N_INPUTS + 1) * 0.1

    # ── FORWARD PASS ─────────────────────────────────────────────────────────

    def _fuzzify(self, X: np.ndarray) -> np.ndarray:
        """
        Layer 1: Fuzzifikasi.
        Input  : X shape (n_samples, N_INPUTS)
        Output : mu shape (n_samples, N_INPUTS, N_MF)
        """
        n = X.shape[0]
        mu = np.zeros((n, N_INPUTS, N_MF))
        for i in range(N_INPUTS):
            for m in range(N_MF):
                mu[:, i, m] = _gaussian_mf(X[:, i],
                                            self.centers[i, m],
                                            self.sigmas[i, m])
        return mu

    def _compute_rule_strengths(self, mu: np.ndarray) -> np.ndarray:
        """
        Layer 2: Rule Firing Strength via T-norm (product).
        Kita gunakan kombinasi 2 input terpenting (energy × danceability)
        untuk 9 rules (3×3), lalu kalikan dengan rata-rata input lain.

        Mengapa product T-norm?
        → Lebih sensitif terhadap input yang sangat rendah (mempertegas perbedaan)
        dibandingkan min T-norm.
        """
        n = mu.shape[0]
        # Rules: kombinasi MF energy (idx 0) × danceability (idx 1)
        w = np.zeros((n, N_RULES))
        r = 0
        for i in range(N_MF):
            for j in range(N_MF):
                w[:, r] = mu[:, 0, i] * mu[:, 1, j]
                r += 1

        # Kalikan dengan rata-rata derajat MF input lainnya (val, speech, acoustic)
        # sebagai modulator global — agar semua fitur ikut berkontribusi
        other_avg = np.mean(mu[:, 2:, :], axis=(1, 2), keepdims=False)
        w = w * other_avg[:, np.newaxis]
        return w  # (n, N_RULES)

    def _normalize(self, w: np.ndarray) -> np.ndarray:
        """Layer 3: Normalisasi rule strengths."""
        total = w.sum(axis=1, keepdims=True) + 1e-9
        return w / total

    def _compute_output(self, w_norm: np.ndarray,
                        X: np.ndarray) -> np.ndarray:
        """
        Layer 4 & 5: Consequent linear functions + aggregasi.
        f_r = a_r0 + a_r1*x1 + ... + a_rn*xn
        output = Σ w_norm_r * f_r
        """
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])   # add bias
        # (n, N_RULES, N_INPUTS+1) dot → (n, N_RULES)
        f = X_aug @ self.consequents.T   # (n, N_RULES)
        output = (w_norm * f).sum(axis=1)  # (n,)
        return output

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Full forward pass — kembalikan semua intermediate values."""
        mu     = self._fuzzify(X)
        w      = self._compute_rule_strengths(mu)
        w_norm = self._normalize(w)
        out    = self._compute_output(w_norm, X)
        return mu, w, w_norm, out

    # ── TRAINING ─────────────────────────────────────────────────────────────

    def _generate_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Membuat target skor (pseudo-label) berbasis formula musikal
        untuk setiap lagu di dataset. Ini mensimulasikan preferensi pengguna
        tanpa memerlukan rating eksplisit (self-supervised).

        Formula target:
          score = 30·energy + 30·danceability + 20·valence
                + 10·(1-speechiness) + 10·popularity_norm
        Lagu yang energik, bisa diajak dance, mood positif, non-spoken
        mendapat skor tinggi — sesuai preferensi umum pengguna musik pop.
        """
        feats = INPUT_FEATURES
        X = df[feats].values.astype(float)

        pop_norm = (df["popularity"].values / 100.0) if "popularity" in df.columns \
                   else np.full(len(df), 0.5)

        y = (30.0 * X[:, 0]            # energy
           + 30.0 * X[:, 1]            # danceability
           + 20.0 * X[:, 2]            # valence
           + 10.0 * (1.0 - X[:, 3])    # 1 - speechiness (semakin sedikit spoken, makin baik)
           + 10.0 * pop_norm)          # popularity

        y = np.clip(y, 0.0, 100.0)
        return X, y

    def train(self, df: pd.DataFrame):
        """
        Latih ANFIS dengan gradient descent sederhana.

        Strategi optimasi:
        - Loss: MSE antara output ANN dan target pseudo-label
        - Update: Least-squares untuk consequent params (lebih stabil)
        - Update: Gradient numerik untuk premise params (c dan σ)
        - Mini-batch size 64 untuk stabilitas
        """
        X, y = self._generate_training_data(df)
        n     = X.shape[0]
        bs    = min(64, n)
        idx   = np.arange(n)

        for epoch in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, bs):
                batch_idx = idx[start:start + bs]
                Xb = X[batch_idx]
                yb = y[batch_idx]

                mu, w, w_norm, out = self._forward(Xb)

                error = out - yb   # (batch,)

                # ── Update consequent params (layer 4) via gradient ──────────
                X_aug = np.hstack([Xb, np.ones((len(Xb), 1))])
                # grad wrt consequents: (n_rules, n_inputs+1)
                grad_c = (w_norm.T @ (error[:, np.newaxis] * X_aug)) / len(Xb)
                self.consequents -= self.lr * grad_c

                # ── Update premise params (c, σ) via numerical gradient ──────
                eps   = 1e-4
                scale = 0.3   # Reduce step size untuk stability premise params
                for i in range(N_INPUTS):
                    for m in range(N_MF):
                        # Gradien numerik untuk center c
                        c_orig = self.centers[i, m]
                        self.centers[i, m] = c_orig + eps
                        _, _, wn_p, out_p = self._forward(Xb)
                        loss_p = np.mean((out_p - yb) ** 2)
                        self.centers[i, m] = c_orig - eps
                        _, _, wn_m, out_m = self._forward(Xb)
                        loss_m = np.mean((out_m - yb) ** 2)
                        self.centers[i, m] = c_orig
                        grad_ci = (loss_p - loss_m) / (2 * eps)
                        self.centers[i, m] -= self.lr * scale * grad_ci

                        # Gradien numerik untuk sigma σ
                        s_orig = self.sigmas[i, m]
                        self.sigmas[i, m] = s_orig + eps
                        _, _, _, out_p = self._forward(Xb)
                        loss_p = np.mean((out_p - yb) ** 2)
                        self.sigmas[i, m] = max(s_orig - eps, 0.05)
                        _, _, _, out_m = self._forward(Xb)
                        loss_m = np.mean((out_m - yb) ** 2)
                        self.sigmas[i, m] = s_orig
                        grad_si = (loss_p - loss_m) / (2 * eps)
                        self.sigmas[i, m] -= self.lr * scale * grad_si
                        # Jaga σ tidak terlalu kecil (min 0.05) agar overlap wajar
                        self.sigmas[i, m] = max(self.sigmas[i, m], 0.05)

            # Clamp centers ke [0,1]
            self.centers = np.clip(self.centers, 0.0, 1.0)

        self._trained = True

    # ── INFERENCE ─────────────────────────────────────────────────────────────

    def score_song(self, row: pd.Series, user_prefs: Dict[str, float]) -> float:
        """
        Hitung skor kecocokan satu lagu.
        Gabungkan output ANN dengan proximity ke preferensi pengguna.
        """
        song_x = np.array([[float(row.get(f, 0.5)) for f in INPUT_FEATURES]])

        # ANN score (sudah dalam skala 0–100 setelah training)
        _, _, _, ann_out = self._forward(song_x)
        ann_score = float(np.clip(ann_out[0], 0.0, 100.0))

        # Proximity score
        diffs = []
        for f in INPUT_FEATURES:
            diffs.append(abs(float(row.get(f, 0.5)) - float(user_prefs.get(f, 0.5))))
        proximity = (1.0 - np.mean(diffs)) * 100.0

        # Gabung: ANN lebih dominan karena sudah belajar dari data
        combined = 0.65 * ann_score + 0.35 * proximity
        return float(np.clip(combined, 0.0, 100.0))

    def recommend(self, df: pd.DataFrame, user_prefs: Dict[str, float],
                  top_n: int = 5) -> pd.DataFrame:
        """Kembalikan top_n lagu dengan skor ANN tertinggi."""
        if not self._trained:
            self.train(df)

        df = df.copy()
        df["match_score"] = df.apply(
            lambda row: self.score_song(row, user_prefs), axis=1
        )
        return (df.nlargest(top_n, "match_score")
                  .reset_index(drop=True))
