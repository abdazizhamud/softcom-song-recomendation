"""
models/genetic_fuzzy.py
Model B2 — Genetic Algorithm (GA) untuk mengoptimasi parameter fungsi keanggotaan
dari Sistem Inferensi Fuzzy (GA-Tuned FIS).

═══════════════════════════════════════════════════════════════════════════════
KONSEP UTAMA
═══════════════════════════════════════════════════════════════════════════════
GA tidak membangun model baru dari nol — ia MENGOPTIMASI parameter MF
dari FIS Manual yang sudah ada. Yang dioptimasi adalah:

  Kromosom (individu) = vektor parameter MF untuk SETIAP fitur input:
  [c_low, c_mid, c_high, σ_low, σ_mid, σ_high]  × N_FEATURES

  Total gen per kromosom = 6 parameter × 5 fitur = 30 gen (float, [0,1])

PIPELINE GA:
  1. Inisialisasi populasi acak (sekitar nilai MF manual sebagai prior)
  2. Evaluasi fitness setiap individu → fitness = avg match score top-K
  3. Seleksi orang tua: Tournament Selection (robust, tidak butuh fitness positif)
  4. Crossover: Uniform Crossover (gen dipilih acak dari salah satu orang tua)
  5. Mutasi: Gaussian Mutation (pergeseran kecil acak pada gen)
  6. Elitisme: individu terbaik langsung masuk generasi berikutnya (tanpa mutasi)
  7. Ulangi hingga max_generations

MENGAPA TOURNAMENT SELECTION?
→ Tidak bergantung pada skala fitness absolut (robust terhadap outlier)
→ Kompleksitas O(k) per seleksi, k=tournament_size

MENGAPA UNIFORM CROSSOVER?
→ Setiap gen bisa berasal dari ayah/ibu dengan probabilitas sama
→ Memungkinkan eksplorasi ruang pencarian yang lebih luas dibanding one-point

MENGAPA GAUSSIAN MUTATION?
→ Mutasi kecil (σ_mut=0.05) → eksploitasi lokal
→ Dengan prob mutasi 0.15 per gen → ~4-5 gen berubah per kromosom
→ Ini menjaga diversitas tanpa merusak individu yang sudah baik
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional


# ── Fitur yang dioptimasi ────────────────────────────────────────────────────
GA_FEATURES = ["energy", "danceability", "valence", "speechiness", "acousticness"]
N_FEATURES  = len(GA_FEATURES)
N_MF        = 3   # Low, Mid, High per fitur

# Parameter per individu: center(3) + sigma(3) per fitur
GENES_PER_FEATURE = N_MF * 2   # 3 centers + 3 sigmas = 6
N_GENES           = N_FEATURES * GENES_PER_FEATURE  # 5 × 6 = 30


# ════════════════════════════════════════════════════════════════════════════
# FUNGSI KEANGGOTAAN GAUSSIAN (vectorized)
# ════════════════════════════════════════════════════════════════════════════
def _gaussmf_v(x: float, centers: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Gaussian MF untuk semua 3 MF sekaligus. Returns shape (3,)."""
    return np.exp(-((x - centers) / (np.maximum(sigmas, 0.04))) ** 2)


# ════════════════════════════════════════════════════════════════════════════
# DECODE KROMOSOM → PARAMETER MF
# ════════════════════════════════════════════════════════════════════════════
def decode_chromosome(chrom: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Decode vektor gen flat (30,) → dict terstruktur per fitur.

    Struktur output:
    {
      "energy":       {"centers": [c_low, c_mid, c_high], "sigmas": [s_low, s_mid, s_high]},
      "danceability": {...},
      ...
    }

    Constraint yang dipaksakan setelah decode:
    - Centers selalu terurut: c_low ≤ c_mid ≤ c_high (sorted)
    - Sigmas selalu positif dan ≥ 0.04 (agar MF tidak terlalu tajam)
    - Semua nilai di-clip ke [0, 1]
    """
    params = {}
    for i, feat in enumerate(GA_FEATURES):
        offset  = i * GENES_PER_FEATURE
        centers = np.clip(np.sort(chrom[offset:offset+3]), 0.0, 1.0)
        sigmas  = np.clip(np.abs(chrom[offset+3:offset+6]), 0.04, 0.5)
        params[feat] = {"centers": centers, "sigmas": sigmas}
    return params


# ════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def _compute_match_score(row: pd.Series, params: Dict, user_prefs: Dict,
                          popularity_norm: float) -> float:
    """
    Hitung skor kecocokan satu lagu menggunakan parameter MF dari kromosom GA.

    Cara kerja:
    1. Fuzzifikasi nilai setiap fitur lagu → derajat keanggotaan (Low/Mid/High)
    2. Evaluasi 4 core rules menggunakan T-norm product
    3. Hitung weighted average → fuzzy_score (0-100)
    4. Gabungkan dengan proximity ke preferensi user
    """
    # ── Fuzzifikasi: hitung derajat keanggotaan setiap fitur ─────────────────
    memberships = {}
    for feat in GA_FEATURES:
        val     = float(row.get(feat, 0.5))
        centers = params[feat]["centers"]
        sigmas  = params[feat]["sigmas"]
        memberships[feat] = _gaussmf_v(val, centers, sigmas)  # (3,) = [low, mid, high]

    # Indeks keanggotaan
    L, M, H = 0, 1, 2

    # ── Rule evaluation (5 rules core) ───────────────────────────────────────
    # Menggunakan T-norm product (lebih diskriminatif dari min)
    e  = memberships["energy"]
    d  = memberships["danceability"]
    v  = memberships["valence"]
    sp = memberships["speechiness"]
    ac = memberships["acousticness"]

    rules = [
        # (strength, crisp_output)
        (e[H] * d[H] * v[H],          95.0),   # Party / Upbeat
        (e[H] * d[H],                  82.0),   # Energik dance
        (e[M] * d[M] * v[M],           65.0),   # Balanced / Mainstream
        (e[L] * ac[H],                 55.0),   # Chill / Acoustic
        (sp[H],                        20.0),   # Heavy spoken word → penalti
        (e[L] * d[L],                  25.0),   # Low energy + low dance → rendah
        (v[H] * d[H],                  78.0),   # Mood cerah + danceable
        (e[H] * popularity_norm,       75.0),   # Energik + populer
    ]

    total_w = sum(r[0] for r in rules) + 1e-9
    fuzzy_score = sum(r[0] * r[1] for r in rules) / total_w

    # ── Proximity ke preferensi user ─────────────────────────────────────────
    diffs = [abs(float(row.get(f, 0.5)) - float(user_prefs.get(f, 0.5)))
             for f in GA_FEATURES]
    proximity = (1.0 - np.mean(diffs)) * 100.0

    combined = 0.70 * fuzzy_score + 0.30 * proximity
    return float(np.clip(combined, 0.0, 100.0))


def _fitness(chrom: np.ndarray, df: pd.DataFrame,
             user_prefs: Dict, top_k: int = 20) -> float:
    """
    Evaluasi fitness satu individu/kromosom.

    Fitness = rata-rata match score dari top-K lagu terpilih.
    Menggunakan top_k=20 (bukan seluruh dataset) agar evaluasi cepat
    sambil tetap representatif.

    Mengapa rata-rata top-K?
    → Jika kita pakai semua 1000 lagu, fitness jadi sangat mirip antar individu
      (karena ada banyak lagu dengan skor rendah yang "meratakan" hasilnya).
    → Top-K memfokuskan seleksi pada kemampuan model menemukan lagu TERBAIK.
    """
    params   = decode_chromosome(chrom)
    pop_norm = df["popularity"].values / 100.0

    scores = np.array([
        _compute_match_score(row, params, user_prefs, pop_norm[idx])
        for idx, row in df.iterrows()
    ])

    # Top-K average sebagai fitness
    top_k_scores = np.sort(scores)[-top_k:]
    return float(np.mean(top_k_scores))


# ════════════════════════════════════════════════════════════════════════════
# GA OPERATORS
# ════════════════════════════════════════════════════════════════════════════

def _tournament_select(population: np.ndarray, fitnesses: np.ndarray,
                        k: int = 4) -> np.ndarray:
    """
    Tournament Selection: pilih k individu acak, kembalikan yang terbaik.
    k=4 memberikan tekanan seleksi moderat — cukup untuk exploit tapi
    tidak terlalu agresif sehingga diversitas terjaga.
    """
    candidates = np.random.choice(len(population), k, replace=False)
    best_idx   = candidates[np.argmax(fitnesses[candidates])]
    return population[best_idx].copy()


def _uniform_crossover(parent_a: np.ndarray, parent_b: np.ndarray,
                        cx_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform Crossover: setiap gen dipilih dari parent_a atau parent_b
    dengan probabilitas cx_prob.
    Menghasilkan 2 anak (child1, child2).
    """
    mask    = np.random.random(N_GENES) < cx_prob
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)
    return child_a.copy(), child_b.copy()


def _gaussian_mutate(individual: np.ndarray, mut_prob: float = 0.15,
                      sigma_mut: float = 0.05) -> np.ndarray:
    """
    Gaussian Mutation: setiap gen berpeluang mut_prob untuk dimutasi.
    Nilai mutasi diambil dari distribusi N(0, sigma_mut).
    Hasil di-clip ke [0, 1].

    mut_prob=0.15 → rata-rata 4-5 gen berubah per kromosom (dari 30 total).
    sigma_mut=0.05 → pergeseran halus, menjaga sifat "lokal" dari mutasi.
    """
    mutant = individual.copy()
    mask   = np.random.random(N_GENES) < mut_prob
    noise  = np.random.normal(0, sigma_mut, N_GENES)
    mutant[mask] += noise[mask]
    return np.clip(mutant, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# KELAS UTAMA
# ════════════════════════════════════════════════════════════════════════════

class GeneticFuzzySystem:
    """
    GA-Tuned Fuzzy Inference System.

    Mengoptimasi parameter MF (centers + sigmas) dari FIS menggunakan
    Genetic Algorithm dengan operator:
    - Tournament Selection (k=4)
    - Uniform Crossover
    - Gaussian Mutation
    - Elitisme (top-1 dipertahankan)

    Parameter yang bisa dikonfigurasi via UI Streamlit:
    - population_size : ukuran populasi (default 40, sweet spot 30-60)
    - max_generations : jumlah generasi (default 50, cukup untuk convergence)
    - mutation_prob   : probabilitas mutasi per gen (default 0.15)
    - crossover_prob  : probabilitas crossover (default 0.8)
    """

    def __init__(
        self,
        population_size : int   = 40,
        max_generations : int   = 50,
        mutation_prob   : float = 0.15,
        crossover_prob  : float = 0.80,
        tournament_k    : int   = 4,
        elite_size      : int   = 2,
        seed            : int   = 42,
    ):
        self.pop_size   = population_size
        self.max_gen    = max_generations
        self.mut_prob   = mutation_prob
        self.cx_prob    = crossover_prob
        self.tourn_k    = tournament_k
        self.elite_size = elite_size
        self.seed       = seed

        # State setelah training
        self.best_chromosome : Optional[np.ndarray] = None
        self.best_params     : Optional[Dict]        = None
        self.best_fitness    : float                 = 0.0
        self.history         : List[Dict]            = []   # per-generasi log
        self._trained        : bool                  = False

    # ── INISIALISASI POPULASI ─────────────────────────────────────────────────
    def _init_population(self) -> np.ndarray:
        """
        Inisialisasi populasi di sekitar nilai MF manual (prior knowledge).

        Nilai awal MF manual (trapezoid center equivalents):
          Low  : center ≈ 0.15, sigma ≈ 0.20
          Mid  : center ≈ 0.50, sigma ≈ 0.15
          High : center ≈ 0.85, sigma ≈ 0.20

        Kita gunakan ini sebagai "mean" dan tambahkan noise N(0, 0.15)
        agar populasi awal bervariasi namun tidak terlalu acak.
        Strategi ini disebut "warm start" — lebih cepat konvergen daripada
        inisialisasi sepenuhnya acak.
        """
        rng = np.random.default_rng(self.seed)

        # Nilai prior MF manual: [c_low, c_mid, c_high, s_low, s_mid, s_high]
        prior_single = np.array([0.15, 0.50, 0.85, 0.20, 0.15, 0.20])
        prior        = np.tile(prior_single, N_FEATURES)   # (30,)

        # Populasi = prior + noise
        pop  = prior[np.newaxis, :] + rng.normal(0, 0.15, (self.pop_size, N_GENES))
        return np.clip(pop, 0.0, 1.0)

    # ── RUN GA ────────────────────────────────────────────────────────────────
    def evolve(
        self,
        df         : pd.DataFrame,
        user_prefs : Dict,
        progress_cb = None,   # callback(gen, max_gen, best_fit) → untuk Streamlit progress
    ) -> None:
        """
        Jalankan loop evolusi GA.

        progress_cb: callable opsional yang dipanggil setiap generasi,
                     berguna untuk update progress bar di Streamlit.
        """
        np.random.seed(self.seed)
        self.history = []

        # ── Init populasi ─────────────────────────────────────────────────────
        pop = self._init_population()

        # ── Evaluasi fitness awal ─────────────────────────────────────────────
        fitnesses = np.array([_fitness(ind, df, user_prefs) for ind in pop])
        best_idx  = int(np.argmax(fitnesses))

        self.best_chromosome = pop[best_idx].copy()
        self.best_fitness    = float(fitnesses[best_idx])

        self.history.append({
            "generation"  : 0,
            "best_fitness": self.best_fitness,
            "avg_fitness" : float(np.mean(fitnesses)),
            "std_fitness" : float(np.std(fitnesses)),
            "diversity"   : float(np.mean(np.std(pop, axis=0))),
        })

        # ── Loop generasi ─────────────────────────────────────────────────────
        for gen in range(1, self.max_gen + 1):
            new_pop = []

            # Elitisme: salin individu terbaik langsung
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            for ei in elite_indices:
                new_pop.append(pop[ei].copy())

            # Isi sisa populasi dengan offspring
            while len(new_pop) < self.pop_size:
                p1 = _tournament_select(pop, fitnesses, self.tourn_k)
                p2 = _tournament_select(pop, fitnesses, self.tourn_k)

                # Crossover
                if np.random.random() < self.cx_prob:
                    c1, c2 = _uniform_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutasi
                c1 = _gaussian_mutate(c1, self.mut_prob)
                c2 = _gaussian_mutate(c2, self.mut_prob)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop       = np.array(new_pop[:self.pop_size])
            fitnesses = np.array([_fitness(ind, df, user_prefs) for ind in pop])

            # Update best
            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > self.best_fitness:
                self.best_fitness    = float(fitnesses[gen_best_idx])
                self.best_chromosome = pop[gen_best_idx].copy()

            log = {
                "generation"  : gen,
                "best_fitness": self.best_fitness,
                "avg_fitness" : float(np.mean(fitnesses)),
                "std_fitness" : float(np.std(fitnesses)),
                "diversity"   : float(np.mean(np.std(pop, axis=0))),
            }
            self.history.append(log)

            if progress_cb is not None:
                progress_cb(gen, self.max_gen, self.best_fitness)

        # ── Simpan parameter terbaik ──────────────────────────────────────────
        self.best_params = decode_chromosome(self.best_chromosome)
        self._trained    = True

    # ── SCORING & RECOMMENDATION ──────────────────────────────────────────────
    def score_song(self, row: pd.Series, user_prefs: Dict) -> float:
        """Hitung skor satu lagu menggunakan parameter MF terbaik dari GA."""
        if not self._trained or self.best_params is None:
            raise RuntimeError("Jalankan evolve() terlebih dahulu.")
        pop_norm = float(row.get("popularity", 50)) / 100.0
        return _compute_match_score(row, self.best_params, user_prefs, pop_norm)

    def recommend(self, df: pd.DataFrame, user_prefs: Dict,
                  top_n: int = 5) -> pd.DataFrame:
        """Kembalikan top_n lagu dengan skor GA-FIS tertinggi."""
        if not self._trained:
            raise RuntimeError("Jalankan evolve() terlebih dahulu.")
        df = df.copy()
        df["match_score"] = df.apply(
            lambda row: self.score_song(row, user_prefs), axis=1
        )
        return df.nlargest(top_n, "match_score").reset_index(drop=True)

    # ── UTILITAS ──────────────────────────────────────────────────────────────
    def get_convergence_data(self) -> pd.DataFrame:
        """Kembalikan history evolusi sebagai DataFrame untuk plotting."""
        return pd.DataFrame(self.history)

    def get_mf_comparison(self) -> Dict:
        """
        Kembalikan perbandingan parameter MF manual vs GA-optimised
        untuk keperluan visualisasi.
        """
        manual = {}
        for feat in GA_FEATURES:
            manual[feat] = {
                "centers": np.array([0.15, 0.50, 0.85]),
                "sigmas" : np.array([0.20, 0.15, 0.20]),
                "type"   : "trapezoid/triangle (manual)",
            }

        optimised = {}
        if self.best_params:
            for feat in GA_FEATURES:
                optimised[feat] = {
                    "centers": self.best_params[feat]["centers"],
                    "sigmas" : self.best_params[feat]["sigmas"],
                    "type"   : "gaussian (GA-optimised)",
                }

        return {"manual": manual, "optimised": optimised}
