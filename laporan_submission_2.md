# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Pertumbuhan industri hiburan digital khususnya dalam bidang anime telah berkembang pesat dalam dekade terakhir. Berdasarkan laporan dari Anime News Network, setiap tahunnya terdapat lebih dari 200 judul anime baru yang dirilis hanya di Jepang. Situasi ini menyebabkan information overload, yaitu kondisi di mana pengguna menghadapi terlalu banyak pilihan sehingga kesulitan dalam memilih konten yang sesuai dengan preferensi mereka (Ricci, Rokach, & Shapira, 2015).

Salah satu solusi efektif dalam permasalahan ini adalah penggunaan sistem rekomendasi. Sistem rekomendasi mampu memberikan saran personal berdasarkan pola interaksi pengguna dengan konten sebelumnya, dan telah banyak digunakan pada platform seperti Netflix, YouTube, dan Spotify (Su & Khoshgoftaar, 2009). Di ranah anime, sistem rekomendasi tidak hanya meningkatkan pengalaman pengguna, tetapi juga membantu meningkatkan retensi dan engagement pengguna terhadap platform distribusi anime seperti MyAnimeList atau AniList.

Sistem rekomendasi tradisional seperti popularity-based atau content-based filtering cenderung kurang akurat karena tidak memperhatikan perilaku kolektif pengguna (Schäfer et al., 2007). Sebaliknya, pendekatan Collaborative Filtering (CF) menawarkan akurasi yang lebih tinggi dengan cara memanfaatkan informasi dari pengguna lain yang memiliki preferensi serupa.

Namun, pendekatan CF konvensional seperti matrix factorization memiliki keterbatasan dalam menangkap non-linearitas dan kompleksitas hubungan antar pengguna dan item. Oleh karena itu, pendekatan modern seperti Neural Collaborative Filtering (NCF), yang menggabungkan teknik deep learning dengan collaborative filtering, menjadi solusi yang lebih powerful dalam mengatasi kelemahan tersebut (He et al., 2017).

Dalam proyek ini, dikembangkan sebuah sistem rekomendasi anime berbasis Neural Collaborative Filtering dengan memanfaatkan data interaksi user-anime berupa rating. Model dirancang menggunakan framework TensorFlow dan memanfaatkan embedding layers untuk merepresentasikan masing-masing pengguna dan anime dalam ruang vektor berdimensi rendah. Prediksi dilakukan dengan menghitung dot product dari vektor pengguna dan anime, disertai bias, untuk menghasilkan skor rating prediktif.

Sistem ini diharapkan mampu memberikan saran yang lebih personal dan relevan kepada pengguna berdasarkan preferensi mereka sebelumnya. Model juga dievaluasi menggunakan metrik seperti Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi rating.

## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

## Referensu
> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. Proceedings of the 26th International Conference on World Wide Web, 173–182. https://doi.org/10.1145/3038912.3052569
> 
> Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6
> 
> Schäfer, J. B., Frankowski, D., Herlocker, J. L., & Sen, S. (2007). Collaborative Filtering Recommender Systems. In The Adaptive Web (pp. 291–324). Springer. https://doi.org/10.1007/978-3-540-72079-9_9
> 
> Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009, Article ID 421425. https://doi.org/10.1155/2009/421425

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
