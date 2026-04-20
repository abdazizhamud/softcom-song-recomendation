"""
models/fuzzy_manual.py
Model A — Sistem Inferensi Fuzzy Mamdani yang dirancang oleh Pakar Manusia.

Desain didasarkan pada intuisi musikal:
- Input : energy, danceability, valence, speechiness, acousticness, popularity
- Output: match_score (0–100)

Fungsi keanggotaan menggunakan trapezoid/triangle agar interpretasi mudah
dan sesuai pendekatan Mamdani klasik.
Aturan-aturan (rules) dirumuskan seperti "jika lagu berenergi tinggi DAN
danceability tinggi MAKA skor tinggi" — meniru cara seorang DJ merekomendasikan.
"""

import numpy as np
import pandas as pd
from typing import Dict


# ════════════════════════════════════════════════════════════════════════════
# FUNGSI KEANGGOTAAN DASAR
# ════════════════════════════════════════════════════════════════════════════

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangle membership function."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid membership function."""
    if x <= a or x >= d:
        return 0.0
    if x >= b and x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-9)
    return (d - x) / (d - c + 1e-9)


# ════════════════════════════════════════════════════════════════════════════
# DERAJAT KEANGGOTAAN PER FITUR
# (Parameter dipilih berdasarkan distribusi umum dataset Spotify)
# ════════════════════════════════════════════════════════════════════════════

def _energy_low(v):    return _trapmf(v, 0.0, 0.0, 0.25, 0.45)
def _energy_med(v):    return _trimf(v,  0.30, 0.50, 0.70)
def _energy_high(v):   return _trapmf(v, 0.55, 0.75, 1.0, 1.0)

def _dance_low(v):     return _trapmf(v, 0.0, 0.0, 0.25, 0.45)
def _dance_med(v):     return _trimf(v,  0.30, 0.50, 0.70)
def _dance_high(v):    return _trapmf(v, 0.55, 0.75, 1.0, 1.0)

def _valence_low(v):   return _trapmf(v, 0.0, 0.0, 0.25, 0.45)
def _valence_med(v):   return _trimf(v,  0.30, 0.50, 0.70)
def _valence_high(v):  return _trapmf(v, 0.55, 0.75, 1.0, 1.0)

def _speech_low(v):    return _trapmf(v, 0.0, 0.0, 0.08, 0.15)
def _speech_high(v):   return _trapmf(v, 0.10, 0.20, 1.0, 1.0)

def _acoustic_low(v):  return _trapmf(v, 0.0, 0.0, 0.20, 0.40)
def _acoustic_high(v): return _trapmf(v, 0.30, 0.55, 1.0, 1.0)

def _pop_low(v):       return _trapmf(v, 0.0, 0.0, 0.20, 0.40)   # v sudah /100
def _pop_med(v):       return _trimf(v,  0.25, 0.50, 0.75)
def _pop_high(v):      return _trapmf(v, 0.60, 0.80, 1.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# SCORING RULES — Mamdani-style (setiap rule menghasilkan bobot kontribusi)
# ════════════════════════════════════════════════════════════════════════════

def _apply_rules(e, d, v, sp, ac, p) -> float:
    """
    Evaluasi 15 aturan fuzzy dan kembalikan agregat centroid (0–100).

    Notasi: E=energy, D=danceability, V=valence, Sp=speechiness,
            Ac=acousticness, P=popularity
    Penalaran: AND → min(); OR → max(); agregasi → rata-rata berbobot.

    Mengapa aturan ini?
    ─────────────────
    • Lagu "party/dance" → E tinggi + D tinggi = skor tinggi
    • Lagu "chill" → E rendah + Ac tinggi = skor sedang-tinggi
    • Lagu "hype pop" → D tinggi + V tinggi + P tinggi = skor sangat tinggi
    • Lagu "spoken word" → Sp tinggi → skor rendah (bukan musik tipikal)
    • Lagu populer mengangkat skor baseline
    """

    # Kalkulasi derajat keanggotaan
    EL, EM, EH  = _energy_low(e),   _energy_med(e),   _energy_high(e)
    DL, DM, DH  = _dance_low(d),    _dance_med(d),    _dance_high(d)
    VL, VM, VH  = _valence_low(v),  _valence_med(v),  _valence_high(v)
    SL, SH      = _speech_low(sp),  _speech_high(sp)
    AL, AH      = _acoustic_low(ac), _acoustic_high(ac)
    PL, PM, PH  = _pop_low(p),      _pop_med(p),      _pop_high(p)

    # (strength, crisp_output) — centroid per rule
    rules = [
        # R1: Lagu party sempurna (E tinggi, D tinggi, V tinggi) → 95
        (min(EH, DH, VH),               95.0),
        # R2: Energik tapi netral (E tinggi, D tinggi) → 85
        (min(EH, DH),                    85.0),
        # R3: Dance ceria (D tinggi, V tinggi) → 82
        (min(DH, VH),                    82.0),
        # R4: Hype populer (E tinggi, P tinggi) → 80
        (min(EH, PH),                    80.0),
        # R5: Mood baik tapi santai (E sedang, V tinggi, D sedang) → 75
        (min(EM, VH, DM),               75.0),
        # R6: Chill akustik (E rendah, Ac tinggi, V sedang) → 70
        (min(EL, AH, VM),               70.0),
        # R7: Sedang semua (E sedang, D sedang, V sedang) → 65
        (min(EM, DM, VM),               65.0),
        # R8: Populer saja (P tinggi) → 60
        (PH,                             60.0),
        # R9: Populer sedang, dance oke (D sedang, P sedang) → 55
        (min(DM, PM),                    55.0),
        # R10: Chill total (E rendah, Ac tinggi) → 50
        (min(EL, AH),                    50.0),
        # R11: Energi rendah + mood rendah → 40
        (min(EL, VL),                    40.0),
        # R12: Dance rendah + energi rendah → 35
        (min(DL, EL),                    35.0),
        # R13: Tidak populer + mood rendah → 30
        (min(PL, VL),                    30.0),
        # R14: Banyak spoken word → penalti (Sp tinggi) → 25
        (SH,                             25.0),
        # R15: Semua rendah → 10
        (min(EL, DL, VL, PL),           10.0),
    ]

    # Agregasi: weighted average (centroid of singleton outputs)
    total_strength = sum(r[0] for r in rules)
    if total_strength < 1e-9:
        return 50.0   # default jika tidak ada rule aktif

    score = sum(r[0] * r[1] for r in rules) / total_strength
    return float(np.clip(score, 0.0, 100.0))


# ════════════════════════════════════════════════════════════════════════════
# KELAS UTAMA
# ════════════════════════════════════════════════════════════════════════════

class ManualFuzzySystem:
    """
    Sistem Inferensi Fuzzy Mamdani yang didesain secara manual oleh pakar.
    Tidak memerlukan training — semua parameter ditentukan a priori.
    """

    def score_song(self, row: pd.Series, user_prefs: Dict[str, float]) -> float:
        """
        Hitung skor kecocokan satu lagu dengan preferensi pengguna.

        Cara kerja:
        1. Hitung jarak fitur lagu dari preferensi pengguna → similarity raw.
        2. Masukkan nilai lagu ke FIS → fuzzy score.
        3. Gabungkan keduanya (70% FIS + 30% proximity).
        """
        feats = ["energy", "danceability", "valence",
                 "speechiness", "acousticness"]

        # Proximity score: 1 - rata-rata |selisih fitur|
        diffs = []
        for f in feats:
            song_val = float(row.get(f, 0.5))
            user_val = float(user_prefs.get(f, 0.5))
            diffs.append(abs(song_val - user_val))
        proximity = (1.0 - np.mean(diffs)) * 100.0  # 0–100

        # Fuzzy score berdasarkan nilai LAGU (bukan preferensi)
        fuzzy_score = _apply_rules(
            e=float(row.get("energy", 0.5)),
            d=float(row.get("danceability", 0.5)),
            v=float(row.get("valence", 0.5)),
            sp=float(row.get("speechiness", 0.05)),
            ac=float(row.get("acousticness", 0.3)),
            p=float(row.get("popularity", 50)) / 100.0,
        )

        # Bobot: FIS memberi konteks musikal, proximity memastikan relevansi
        combined = 0.70 * fuzzy_score + 0.30 * proximity
        return float(np.clip(combined, 0.0, 100.0))

    def recommend(self, df: pd.DataFrame, user_prefs: Dict[str, float],
                  top_n: int = 5) -> pd.DataFrame:
        """Kembalikan top_n lagu dengan skor kecocokan tertinggi."""
        df = df.copy()
        df["match_score"] = df.apply(
            lambda row: self.score_song(row, user_prefs), axis=1
        )
        return (df.nlargest(top_n, "match_score")
                  .reset_index(drop=True))
