# 🎵 THE INTELLIGENCE BATTLE — Panduan Penggunaan Aplikasi

**Proyek UTS Soft Computing**

**Judul:** The Intelligence Battle: Human Expert vs. GA-Tuned FIS vs. Neuro-Fuzzy ANN

**Mahasiswa:** 

**Program Studi:** Teknik Informatika — Universitas Padjadjaran

---

## 📋 Persyaratan Sistem

### Perangkat Keras (Minimum)

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Storage | 200 MB (free) | 500 MB |
| CPU | Dual-core | Quad-core |

### Perangkat Lunak

- **Python** versi 3.9 atau lebih baru
- **pip** (Python package manager) — biasanya sudah terinstal bersama Python

### Library Python yang Dibutuhkan

```
streamlit>=1.32.0       # Framework GUI web app
pandas>=2.0.0           # Manipulasi dataset
numpy>=1.26.0           # Komputasi numerik (WAJIB untuk model ANN)
matplotlib>=3.8.0       # Plotting dasar
plotly>=5.20.0          # Visualisasi interaktif (radar chart, bar chart, heatmap)
scikit-learn>=1.4.0     # Utilitas preprocessing
deap>=1.4.1             # Framework Genetic Algorithm
```

> **Catatan:** 
> - Sistem Fuzzy Mamdani diimplementasi dari nol menggunakan NumPy (no scikit-fuzzy)
> - DEAP library digunakan untuk optimasi GA (Distributed Evolutionary Algorithms in Python)

---

## ⚙️ Langkah Instalasi & Akses Aplikasi

### Langkah 1 — Siapkan Struktur Folder

Pastikan folder proyek Anda memiliki struktur berikut:

```
song-recomendation-softcom/
├── app.py                  ← File utama Streamlit
├── requirements.txt        ← Daftar dependensi
├── dataset.csv             ← (Opsional) Dataset lagu Spotify Anda
├── POSTER.txt              ← Konten poster digital
├── README.md               ← File ini
├── data/
│   ├── __init__.py
│   └── loader.py           ← Modul pemuat dataset
└── models/
    ├── __init__.py
    ├── fuzzy_manual.py     ← Model A: Mamdani FIS (Manual)
    ├── genetic_fuzzy.py    ← Model B: GA-Tuned FIS (Genetic Algorithm Optimization)
    └── neuro_fuzzy.py      ← Model C: Neuro-Fuzzy ANN (Adaptive Neural)
```

### Langkah 2 — (Opsional) Siapkan Dataset

Jika Anda memiliki file `dataset.csv` (dataset Spotify), letakkan di folder
utama (sejajar dengan `app.py`). Kolom yang dibutuhkan:

```
track_name, artists, popularity, danceability, energy,
speechiness, acousticness, instrumentalness, liveness,
valence, tempo, loudness, track_genre
```

> Jika `dataset.csv` tidak ada, aplikasi otomatis menggunakan **1.000 data sintetis**
> yang menyerupai distribusi dataset Spotify. Aplikasi tetap berfungsi penuh.

### Langkah 3 — Buat Virtual Environment (Disarankan)

```bash
# Buat environment baru bernama "venv"
python -m venv venv

# Aktifkan environment (Windows)
venv\Scripts\activate

# Aktifkan environment (macOS / Linux)
source venv/bin/activate
```

### Langkah 4 — Instal Dependensi

```bash
pip install -r requirements.txt
```

Proses instalasi membutuhkan koneksi internet dan sekitar 1–3 menit.

### Langkah 5 — Jalankan Aplikasi

```bash
streamlit run app.py
```

Setelah perintah dijalankan, terminal akan menampilkan:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Buka browser dan akses **http://localhost:8501** untuk melihat aplikasi.

> **Tips:** Pertama kali dijalankan, model Neuro-Fuzzy perlu melatih diri (sekitar
> 5–15 detik tergantung kecepatan CPU). Setelah itu, model akan di-cache dan
> inferensi berikutnya sangat cepat (< 1 detik).

---

## 🖥️ Fungsi Utama & Cara Penggunaan

### 1. Panel Preferensi (Sidebar — kiri layar)

Atur preferensi musik Anda menggunakan **6 slider**:

| Slider | Rentang | Keterangan |
|--------|---------|------------|
| ⚡ **Energy** | 0.0 – 1.0 | Seberapa intens/energik lagu yang Anda inginkan. 1.0 = sangat energik (metal, EDM); 0.0 = tenang (ambient, classical) |
| 💃 **Danceability** | 0.0 – 1.0 | Seberapa cocok lagu untuk berdansa. 1.0 = sangat bisa diajak dance; 0.0 = sulit di-dance |
| 😊 **Valence (Mood)** | 0.0 – 1.0 | Nada emosional lagu. 1.0 = sangat positif/ceria; 0.0 = melankolis/sedih |
| 🎤 **Speechiness** | 0.0 – 1.0 | Kandungan kata-kata/spoken word. 0.0 = murni musik; 1.0 = podcast/rap |
| 🎸 **Acousticness** | 0.0 – 1.0 | Seberapa akustik (vs. elektronik) lagu tersebut. 1.0 = murni akustik |
| 🌟 **Popularity** | 0 – 100 | Preferensi popularitas lagu. 100 = lagu hits terkenal |

**Contoh Preset:**

| Preset | Energy | Dance | Valence | Speech | Acoustic | Pop |
|--------|--------|-------|---------|--------|----------|-----|
| 🎉 Party/Dance | 0.85 | 0.85 | 0.7 | 0.05 | 0.1 | 80 |
| 😌 Chill/Relax | 0.25 | 0.35 | 0.6 | 0.05 | 0.7 | 50 |
| 🎸 Rock | 0.8 | 0.45 | 0.4 | 0.07 | 0.15 | 65 |
| 🎵 Jazz | 0.35 | 0.5 | 0.55 | 0.05 | 0.65 | 40 |

### 2. Konfigurasi Genetic Algorithm (Opsional)

Sebelum generate, Anda bisa menyesuaikan parameter GA di bagian **⚙️ Genetic Algorithm Settings**:

| Parameter | Rentang | Default | Penjelasan |
|-----------|---------|---------|------------|
| **Population Size** | 10–80 | 40 | Jumlah individu (MF sets) dalam satu generasi |
| **Max Generations** | 10–120 | 50 | Jumlah iterasi evolusi |
| **Mutation Rate** | 0.05–0.40 | 0.15 | Probabilitas perubahan gen (centers/sigmas) |
| **Crossover Rate** | 0.50–1.00 | 0.80 | Probabilitas crossover (uniform crossover) |
| **Random Seed** | 0–9999 | 42 | Untuk reproducibilitas hasil |

> **Tips:** 
> - Pop Size 30–60 = sweet spot (akurasi tinggi, waktu wajar)
> - Gen 40–80 = GA biasanya konvergen di sini
> - Mutation 0.10–0.20 = balance antara eksplorasi & eksploitasi

### 3. Tombol "Generate Recommendations" 🚀

Klik tombol di bagian bawah sidebar untuk memulai proses rekomendasi.
Ketiga model (FIS Manual, GA-Tuned FIS, dan Neuro-Fuzzy ANN) akan berjalan
**secara bersamaan** dan hasilnya ditampilkan berdampingan.

**Durasi Eksekusi:**
- Model A (FIS Manual): ~100–200 ms (instant)
- Model B (GA-Tuned FIS): ~30–80 s (tergantung pop size & gen)
- Model C (Neuro-Fuzzy ANN): ~200–500 ms (setelah training diload dari cache)

### 4. Membaca Hasil Rekomendasi

Hasil ditampilkan dalam **3 kolom**:

**Kolom Kiri — 👤 Model A: FIS Manual**

- Tabel berisi 5 lagu terbaik menurut sistem fuzzy manual
- Badge biru menunjukkan waktu eksekusi (biasanya 100–200 ms)
- Sumber: Rule-based Mamdani FIS

**Kolom Tengah — 🧬 Model B: GA-Tuned FIS**

- Tabel berisi 5 lagu terbaik menurut GA-optimized FIS
- Badge hijau menunjukkan durasi GA evolution & best fitness
- Sumber: Membership functions yang sudah di-evolusi GA

**Kolom Kanan — 🤖 Model C: Neuro-Fuzzy ANN**

- Tabel berisi 5 lagu terbaik menurut model ANN
- Badge pink menunjukkan waktu eksekusi (< 500 ms)
- Sumber: Trained neural network

**Banner Pemenang 🏆**

Tepat di bawah ketiga tabel, banner emas menampilkan model mana yang
menghasilkan rata-rata skor TERTINGGI untuk preferensi saat ini, beserta
perbandingan skor FIS vs GA vs ANN.

### 5. Membaca Visualisasi & Analisis

Setelah generate, ada 5 tab untuk analisis mendalam:

**Tab 🕸️ Radar Chart**

- Menampilkan 4 poligon: FIS (biru), GA (hijau), ANN (pink), Preferensi Anda (kuning)
- Semakin tumpang tindih poligon model dengan preferensi → semakin relevan
- Sumbu: danceability, energy, valence, speechiness, acousticness
- **Insight:** Poligon GA biasanya lebih "sharp" ke preferensi Anda vs FIS manual

**Tab 📊 Score Bars**

- Tiga bar chart horizontal berdampingan (FIS, GA, ANN)
- Setiap bar = satu lagu, panjang = skor kecocokan (0–100)
- Warna gradient menunjukkan intensitas skor
- **Insight:** Lihat mana model yang scoring-nya paling sesuai ekspektasi Anda

**Tab 🌡️ Feature Heatmap**

- Grid 15×6: baris = 15 lagu (5 FIS + 5 GA + 5 ANN), kolom = 6 fitur audio
- Warna plasma: terang (nilai tinggi) — gelap (nilai rendah)
- **Insight:** Lihat karakteristik lagu apa yang dipilih masing-masing model

**Tab 🧬 GA Convergence**

- Kurva fitness vs generation untuk GA yang baru saja dijalankan
- Garis hijau = best fitness per generasi
- Garis kuning putus-putus = average fitness
- Shaded area = standar deviasi band
- **Insight:** Lihat apakah GA konvergen di generasi berapa, ada plateau/improvement?
- Metrics: Final Best Fitness, Convergence Gen, Pop Size, Total Gen

**Tab 📐 MF Shift**

- Perbandingan visual membership functions sebelum (Manual) vs sesudah (GA-Optimized)
- Dua subplot: Manual FIS (kiri) vs GA-Optimized (kanan)
- Tiga kurva per subplot: Low (biru), Mid (kuning), High (pink)
- Tabel numerik: Delta centers & sigmas untuk melihat pergeseran kuantitatif
- **Insight:** Apakah GA menggeser MF dengan masuk akal? Atau terjadi overfitting?

### 6. Analisis Overlap

Di bagian bawah halaman, **🔍 Overlap Analysis** menunjukkan:

| Metrik | Penjelasan |
|--------|------------|
| **FIS Only** | Lagu yang hanya FIS Manual yang rekomendasikan |
| **GA Only** | Lagu yang hanya GA-Tuned FIS yang rekomendasikan |
| **ANN Only** | Lagu yang hanya Neuro-Fuzzy ANN yang rekomendasikan |
| **FIS & GA** | Lagu yang disepakati FIS Manual & GA |
| **All 3 Agree** | Lagu yang disepakati KETIGA model (consensus) |

**Interpretasi:**
- Overlap tinggi (3–5 lagu) → ketiga model punya "opinion" serupa
- Overlap rendah (0–1 lagu) → tiap model punya preferensi unik
- All 3 Agree = high confidence lagu → rekomendasi paling andal

---

## 🔬 Penjelasan Teknis Singkat

### Tiga Paradigma Soft Computing

| Aspek | Model A: FIS Manual | Model B: GA-Tuned FIS | Model C: Neuro-Fuzzy ANN |
|-------|--------------------|-----------------------|--------------------------|
| **Paradigma** | Expert System | Evolutionary Algorithm | Neural Network |
| **Sumber Pengetahuan** | Intuisi manusia/pakar | Rule + Data (fitness function) | Data (supervised learning) |
| **Parameter Fuzzy** | Hardcoded (manual) | Auto-optimized oleh GA | Learned oleh backprop |
| **Fleksibilitas** | Tetap (static) | Adaptif (evolved) | Sangat adaptif (learned) |
| **Interpretabilitas** | ✅ Sangat tinggi | ⚠️ Sedang (harus lihat MF) | ❌ Black box |
| **Waktu Setup** | ⚡ Instan | 🕐 30–80 s (GA evolution) | 🕐 5–15 s (training) |
| **Bias** | 👤 Bias pakar | 🧬 Bias fitness function | 📊 Bias training data |
| **Kapan Pakai?** | Expert available | Auto-tuning diinginkan | Data melimpah |

### Mengapa Ketiga Model Bisa Memberikan Hasil Berbeda?

**FIS Manual** → Rule-based, hasil konsisten tapi terbatas pada rule yang dibuat pakar

**GA-Tuned FIS** → Rule sama, tapi membership functions di-optimize untuk **fit maksimal dengan preferensi user saat ini**. Bisa overfitting jika pop size/gen terlalu kecil.

**Neuro-Fuzzy ANN** → Learns dari training data, bukan "custom" per user. Lebih general tapi kurang personalisasi.

### Cara Kerja Skor (0–100)

Ketiga model menggunakan strategi **fuzzy inference** + **proximity matching**:

```
Final Score = w₁ × Fuzzy_Output + w₂ × Proximity_Score
              w₁ = 65–70%              w₂ = 30–35%
```

**Fuzzy Output:**
- Inference: Berapa derajat membership lagu pada output fuzzy sets (Low/Medium/High Recommendation)
- Range: [0, 1] di-scale ke [0, 100]

**Proximity Score:**
- Euclidean distance antara fitur lagu & preferensi user dalam 6D space
- Closer = higher score

**Perbedaan antar model:**
- **FIS Manual:** Fuzzy rules hardcoded
- **GA-Tuned FIS:** Fuzzy rules sama, tapi MF parameters evolved → fuzzy output bisa sangat berbeda
- **Neuro-Fuzzy:** Fuzzy rules & MF parameters learned from data → completely different scoring

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: streamlit` | Jalankan `pip install -r requirements.txt` |
| Aplikasi loading lama | Normal untuk pertama kali — model ANN sedang training |
| `dataset.csv not found` | Tidak masalah — data sintetis digunakan otomatis |
| Port 8501 sudah digunakan | Jalankan `streamlit run app.py --server.port 8502` |
| Tampilan berantakan | Coba zoom out browser ke 80-90% |

---

## 📦 Menghentikan Aplikasi

Tekan `Ctrl + C` di terminal untuk menghentikan server Streamlit.