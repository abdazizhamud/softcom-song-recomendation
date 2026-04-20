# 🎵 SONIC MIND — Panduan Penggunaan Aplikasi

**Proyek UTS Soft Computing**  
**Judul:** The Intelligence Battle: Human Expert vs. Evolutionary Tuning & Neuro-Fuzzy  
**Mahasiswa:** Hamud Abdul Aziz (NPM: 10020230042)  
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
scikit-learn>=1.4.0     # Utilitas preprocessing (opsional)
```

> **Catatan:** Library `scikit-fuzzy` TIDAK digunakan. Sistem Fuzzy Mamdani
> diimplementasi dari nol menggunakan NumPy agar lebih transparan dan edukatif.

---

## ⚙️ Langkah Instalasi & Akses Aplikasi

### Langkah 1 — Siapkan Struktur Folder

Pastikan folder proyek Anda memiliki struktur berikut:

```
sonic-mind/
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
    ├── fuzzy_manual.py     ← Model A: Mamdani FIS
    └── neuro_fuzzy.py      ← Model B: Neuro-Fuzzy ANN
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
- 🎉 *Party/Dance*: Energy=0.85, Dance=0.85, Valence=0.7, Speech=0.05, Acoustic=0.1, Pop=80
- 😌 *Chill/Relax*: Energy=0.25, Dance=0.35, Valence=0.6, Speech=0.05, Acoustic=0.7, Pop=50
- 🎸 *Rock*: Energy=0.8, Dance=0.45, Valence=0.4, Speech=0.07, Acoustic=0.15, Pop=65
- 🎵 *Jazz*: Energy=0.35, Dance=0.5, Valence=0.55, Speech=0.05, Acoustic=0.65, Pop=40

### 2. Tombol "Generate Recommendations" 🚀

Klik tombol di bagian bawah sidebar untuk memulai proses rekomendasi.
Kedua model (FIS dan Neuro-Fuzzy) akan berjalan **secara bersamaan** dan
hasilnya ditampilkan berdampingan.

### 3. Membaca Hasil Rekomendasi

Hasil ditampilkan dalam **2 kolom**:

**Kolom Kiri — 👤 Model A: Human Expert FIS**
- Tabel berisi 5 lagu terbaik menurut sistem fuzzy manual
- Badge biru menunjukkan waktu eksekusi (biasanya < 100 ms)
- Kolom: Track Name | Artist | Match Score | Energy | Dance | Valence

**Kolom Kanan — 🤖 Model B: Neuro-Fuzzy (ANN)**
- Tabel berisi 5 lagu terbaik menurut model ANN
- Badge pink menunjukkan waktu eksekusi (biasanya < 50 ms setelah training)
- Format tabel sama dengan kolom kiri

**Banner Pemenang 🏆**  
Tepat di bawah kedua tabel, banner emas menampilkan model mana yang
menghasilkan rata-rata skor lebih tinggi untuk preferensi saat ini.

### 4. Membaca Visualisasi

Klik tab-tab berikut di bagian bawah halaman:

**Tab 🕸️ Radar Chart**
- Menampilkan 3 poligon: FIS (biru), ANN (pink), Preferensi Anda (kuning)
- Semakin tumpang tindih poligon model dengan preferensi Anda → semakin relevan rekomendasinya
- Sumbu: danceability, energy, valence, speechiness, acousticness

**Tab 📊 Score Bars**
- Dua bar chart horizontal berdampingan (FIS vs ANN)
- Setiap bar = satu lagu rekomendasi, panjang bar = skor kecocokan (0–100)
- Warna lebih terang = skor lebih tinggi

**Tab 🌡️ Feature Heatmap**
- Grid 10×6: baris = 10 lagu (5 FIS + 5 ANN), kolom = 6 fitur audio
- Warna terang = nilai fitur tinggi; gelap = rendah
- Berguna untuk melihat perbedaan karakteristik lagu yang dipilih kedua model

### 5. Analisis Overlap

Di bagian bawah halaman, **Overlap Analysis** menunjukkan:
- Berapa lagu yang hanya direkomendasikan FIS
- Berapa lagu yang direkomendasikan oleh KEDUA model (sepakat)
- Berapa lagu yang hanya direkomendasikan ANN

Semakin besar overlap → kedua model memiliki "pendapat" yang sama tentang
lagu terbaik untuk preferensi tersebut.

---

## 🔬 Penjelasan Teknis Singkat

### Mengapa Dua Model Bisa Berbeda?

| Aspek | FIS Manual (Model A) | Neuro-Fuzzy (Model B) |
|-------|---------------------|----------------------|
| **Sumber Pengetahuan** | Intuisi manusia/pakar | Data (pseudo-labels) |
| **Fleksibilitas** | Tetap (hardcoded rules) | Adaptif (learned weights) |
| **Interpretabilitas** | Sangat tinggi | Sedang |
| **Waktu Setup** | Instan | Perlu training |
| **Bias** | Bias pakar | Bias data |

### Cara Kerja Skor (0–100)

Kedua model menggunakan strategi gabungan:
- **Skor Model** (70% FIS / 65% ANN): Seberapa "baik" lagu tersebut secara absolut
- **Skor Proximity** (30% / 35%): Seberapa dekat fitur lagu dengan preferensi Anda

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

---
#   s o f t c o m - s o n g - r e c o m e n d a t i o n  
 