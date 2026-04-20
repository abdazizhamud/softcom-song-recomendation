"""
data/loader.py
Memuat dataset lagu dari file CSV (jika tersedia) atau membuat data sintetis
yang menyerupai dataset Spotify untuk keperluan demonstrasi.
"""

import os
import numpy as np
import pandas as pd


# ── Kolom audio yang dipakai dalam pemodelan ─────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "valence", "speechiness",
    "acousticness", "instrumentalness", "liveness", "tempo",
    "loudness", "popularity",
]


def _generate_synthetic_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Membuat dataset sintetis yang menyerupai Spotify untuk demonstrasi.
    Dipanggil hanya bila dataset.csv tidak ditemukan.
    """
    rng = np.random.default_rng(seed)

    genres = [
        "pop", "rock", "hip-hop", "jazz", "classical",
        "electronic", "r&b", "latin", "indie", "metal",
    ]
    artists_pool = [
        "Taylor Swift", "Drake", "BTS", "Ed Sheeran", "Billie Eilish",
        "The Weeknd", "Ariana Grande", "Post Malone", "Dua Lipa", "Bad Bunny",
        "Coldplay", "Eminem", "Rihanna", "Kendrick Lamar", "Harry Styles",
        "SZA", "Olivia Rodrigo", "Travis Scott", "Adele", "Justin Bieber",
    ]

    # Distribusi per genre agar lebih realistis
    genre_profiles = {
        "pop":         dict(energy=(0.6,0.2), dance=(0.65,0.15), val=(0.6,0.2), speech=(0.05,0.03)),
        "rock":        dict(energy=(0.75,0.15), dance=(0.5,0.15), val=(0.45,0.2), speech=(0.06,0.03)),
        "hip-hop":     dict(energy=(0.65,0.15), dance=(0.75,0.12), val=(0.5,0.2), speech=(0.2,0.1)),
        "jazz":        dict(energy=(0.4,0.15), dance=(0.45,0.15), val=(0.55,0.2), speech=(0.05,0.02)),
        "classical":   dict(energy=(0.25,0.15), dance=(0.3,0.1), val=(0.4,0.2), speech=(0.03,0.01)),
        "electronic":  dict(energy=(0.8,0.1), dance=(0.7,0.12), val=(0.5,0.2), speech=(0.06,0.03)),
        "r&b":         dict(energy=(0.55,0.15), dance=(0.7,0.12), val=(0.6,0.2), speech=(0.08,0.04)),
        "latin":       dict(energy=(0.7,0.15), dance=(0.8,0.1), val=(0.7,0.15), speech=(0.07,0.03)),
        "indie":       dict(energy=(0.5,0.2), dance=(0.5,0.18), val=(0.5,0.22), speech=(0.05,0.03)),
        "metal":       dict(energy=(0.9,0.08), dance=(0.4,0.15), val=(0.3,0.15), speech=(0.07,0.04)),
    }

    rows = []
    for i in range(n):
        genre = rng.choice(genres)
        p = genre_profiles[genre]
        clip = lambda x: float(np.clip(x, 0.0, 1.0))

        energy       = clip(rng.normal(p["energy"][0],  p["energy"][1]))
        danceability = clip(rng.normal(p["dance"][0],   p["dance"][1]))
        valence      = clip(rng.normal(p["val"][0],     p["val"][1]))
        speechiness  = clip(rng.normal(p["speech"][0],  p["speech"][1]))
        acousticness = clip(rng.normal(0.35, 0.25))
        instrumentalness = clip(rng.exponential(0.1))
        liveness     = clip(rng.normal(0.18, 0.1))
        tempo        = float(np.clip(rng.normal(118, 28), 60, 210))
        loudness     = float(np.clip(rng.normal(-8, 5), -35, 0))
        popularity   = int(np.clip(rng.normal(52, 22), 0, 100))

        rows.append({
            "track_id":        f"T{i:05d}",
            "artists":         rng.choice(artists_pool),
            "album_name":      f"Album_{rng.integers(1, 300)}",
            "track_name":      f"Song_{i:04d}_{genre.title()}",
            "popularity":      popularity,
            "duration_ms":     int(rng.normal(210_000, 50_000)),
            "explicit":        bool(rng.choice([True, False], p=[0.2, 0.8])),
            "danceability":    danceability,
            "energy":          energy,
            "key":             int(rng.integers(0, 12)),
            "loudness":        loudness,
            "mode":            int(rng.choice([0, 1])),
            "speechiness":     speechiness,
            "acousticness":    acousticness,
            "instrumentalness": instrumentalness,
            "liveness":        liveness,
            "valence":         valence,
            "tempo":           tempo,
            "time_signature":  int(rng.choice([3, 4], p=[0.15, 0.85])),
            "track_genre":     genre,
        })

    return pd.DataFrame(rows)


def load_dataset(csv_path: str = "dataset.csv", sample_n: int = 1000) -> pd.DataFrame:
    """
    Memuat dan memproses dataset lagu.

    Langkah-langkah:
    1. Jika dataset.csv ada, muat dan ambil sampel acak 1000 baris.
    2. Jika tidak ada, buat dataset sintetis.
    3. Tangani missing values dengan median/modus.
    4. Normalkan kolom tempo dan loudness ke [0,1].
    5. Kembalikan DataFrame yang siap dipakai.
    """
    # ── Step 1: Load atau generate ────────────────────────────────────────────
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ambil sampel acak agar komputasi cepat
        if len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    else:
        print(f"[loader] '{csv_path}' not found — using synthetic dataset ({sample_n} rows).")
        df = _generate_synthetic_dataset(n=sample_n)

    # ── Step 2: Drop kolom string non-esensial (tidak dipakai untuk model) ───
    drop_cols = [c for c in ["track_id", "album_name", "explicit",
                              "key", "mode", "time_signature", "duration_ms"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # ── Step 3: Tangani missing values ───────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Step 4: Normalisasi tempo → [0,1] dan loudness → [0,1] ──────────────
    if "tempo" in df.columns:
        df["tempo"] = (df["tempo"] - df["tempo"].min()) / (
            df["tempo"].max() - df["tempo"].min() + 1e-9)

    if "loudness" in df.columns:
        df["loudness"] = (df["loudness"] - df["loudness"].min()) / (
            df["loudness"].max() - df["loudness"].min() + 1e-9)

    # ── Step 5: Pastikan kolom penting ada ───────────────────────────────────
    required = ["track_name", "artists", "danceability", "energy",
                "valence", "speechiness", "acousticness", "popularity"]
    for col in required:
        if col not in df.columns:
            df[col] = 0.5 if col not in ["track_name", "artists"] else "Unknown"

    return df.reset_index(drop=True)
