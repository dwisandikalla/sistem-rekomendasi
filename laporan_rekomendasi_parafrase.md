# Laporan Proyek Machine Learning - Sistem Rekomendasi Anime

## Project Overview

Seiring meningkatnya minat masyarakat terhadap anime, volume konten yang tersedia turut melonjak pesat setiap tahunnya. Kondisi ini menciptakan tantangan berupa "information overload" di mana pengguna kerap kesulitan dalam memilih tontonan yang sesuai dengan preferensi mereka. Sistem rekomendasi hadir sebagai solusi untuk mengurangi beban pencarian dengan memberikan saran tayangan yang dipersonalisasi berdasarkan histori interaksi pengguna.

Dalam proyek ini, dikembangkan sistem rekomendasi anime berbasis machine learning dengan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering. Sistem ini dibangun menggunakan dataset dari Kaggle yang berisi metadata anime dan rating pengguna. Dengan memanfaatkan teknik seperti TF-IDF dan neural collaborative filtering, sistem ini dirancang untuk menyarankan anime yang relevan dan personal.

## Business Understanding

### Problem Statements

1. Banyaknya pilihan anime menyulitkan pengguna dalam menemukan tontonan sesuai selera.
2. Rekomendasi berbasis popularitas tidak selalu mencerminkan preferensi individual pengguna.
3. Belum optimalnya pemanfaatan histori interaksi pengguna dalam memberikan rekomendasi.

### Goals

1. Mengembangkan sistem yang dapat memberikan rekomendasi berdasarkan riwayat rating pengguna.
2. Menerapkan dan membandingkan dua metode rekomendasi: Content-Based dan Collaborative Filtering.
3. Mengevaluasi sistem menggunakan metrik yang sesuai untuk mengukur akurasi rekomendasi.

### Solution Statements

- **Content-Based Filtering:** Menggunakan fitur genre dari anime dan teknik TF-IDF untuk menghitung kemiripan antar konten, menghasilkan rekomendasi berdasarkan anime yang disukai sebelumnya.
- **Collaborative Filtering:** Menggunakan embedding dan neural network untuk mempelajari pola rating antar pengguna, lalu memprediksi rating potensial untuk anime yang belum ditonton.

## Data Understanding

Dataset diambil dari Kaggle: [Anime Recommendation Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Dataset ini terdiri dari dua file:

- `anime.csv` (12.294 baris, 7 kolom) berisi informasi anime: ID, nama, genre, jenis, jumlah episode, rating rata-rata, dan jumlah pengguna.
- `rating.csv` (7.813.737 baris, 3 kolom) berisi data interaksi pengguna dengan anime berupa user_id, anime_id, dan rating.

### Fitur Data

- **anime.csv:**
  - `anime_id`: ID unik.
  - `name`: Judul anime.
  - `genre`: Daftar genre.
  - `type`: Tipe (TV, Movie, dll).
  - `episodes`: Jumlah episode.
  - `rating`: Nilai rata-rata rating komunitas.
  - `members`: Jumlah pengguna yang menambahkan ke daftar mereka.

- **rating.csv:**
  - `user_id`: ID pengguna.
  - `anime_id`: ID anime.
  - `rating`: Rating dari -1 hingga 10 (-1 berarti tidak memberikan rating).

### EDA dan Temuan

- Missing value ditemukan di genre, type, episodes, dan rating.
- Rating -1 di rating.csv menandakan pengguna tidak menilai.
- Genre anime sangat beragam dan berperan penting dalam rekomendasi berbasis konten.
- Visualisasi menunjukkan bias positif: sebagian besar pengguna memberi rating tinggi (8–10).
- TV Series adalah tipe anime yang paling banyak muncul.

## Data Preparation

- Menghapus baris duplikat dan nilai kosong dari anime.csv.
- Menghilangkan rating bernilai -1 dari rating.csv.
- Menyaring data rating agar hanya mencakup anime yang tersedia di anime.csv.
- Sampling maksimum 100 rating per anime.
- Menggabungkan kedua dataset berdasarkan `anime_id` menjadi dataset `anime_rating`.
- Encoding user dan anime ke indeks numerik.
- Normalisasi rating ke skala 0–1 menggunakan min-max scaling.
- Membagi data menjadi training dan validation (80:20).

## Modeling

### Content-Based Filtering

- Menggunakan TF-IDF Vectorizer pada kolom genre.
- Menghitung cosine similarity antar anime.
- Fungsi rekomendasi akan mengambil anime yang mirip dengan yang disukai sebelumnya.

**Kelebihan:**
- Tidak memerlukan data pengguna lain.
- Cocok untuk item baru (cold start).

**Kekurangan:**
- Hanya melihat kemiripan konten, bukan preferensi pengguna secara keseluruhan.

### Collaborative Filtering

- Menerapkan model neural network (RecommenderNet) dengan TensorFlow.
- Menggunakan embedding layer untuk user dan anime.
- Mengombinasikan dot product dan dense layer untuk prediksi rating.
- Training menggunakan MSE, evaluasi menggunakan RMSE.

**Kelebihan:**
- Menangkap preferensi pengguna yang kompleks.
- Memberikan rekomendasi yang lebih personal.

**Kekurangan:**
- Tidak optimal untuk pengguna baru (cold start).
- Memerlukan proses training yang lebih lama.

## Evaluation

### Content-Based Filtering

Metrik evaluasi yang digunakan: **Precision@K**

\[ \text{Precision@K} = \frac{\text{Jumlah item relevan dalam K rekomendasi}}{K} \]

- 100 pengguna diambil secara acak.
- Rekomendasi diberikan berdasarkan 1 anime favorit per user.
- Evaluasi menghasilkan rata-rata **Precision@5 = 0.056** (5.6%).

### Collaborative Filtering

Metrik evaluasi yang digunakan: **RMSE**

\[ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } \]

- Model dilatih dengan validasi.
- Nilai RMSE cukup rendah, menunjukkan kemampuan prediksi yang baik.

### Kesimpulan Evaluasi

- Content-based memberikan hasil awal yang cukup baik namun terbatas.
- Collaborative filtering lebih efektif dan akurat dalam merekomendasikan anime.

## Kesimpulan

Sistem rekomendasi yang dibangun telah menjawab tantangan informasi berlebih pada platform anime. Pendekatan content-based mampu memberikan rekomendasi berdasarkan genre, sementara collaborative filtering lebih unggul dalam memberikan rekomendasi yang personal dan akurat. Model ini dapat digunakan sebagai dasar pengembangan lebih lanjut seperti hybrid recommendation atau penggunaan metadata lainnya (sinopsis, studio, dll).

## Referensi

- He, X. et al. (2017). Neural collaborative filtering. *WWW '17 Proceedings of the 26th International Conference on World Wide Web*.
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
- Schäfer, J. B. et al. (2007). Collaborative Filtering Recommender Systems. *The Adaptive Web*.
- Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. *Advances in Artificial Intelligence*.

