# Laporan Proyek Machine Learning - Dwi Sandi Kalla

## Project Overview

Pertumbuhan industri hiburan digital khususnya dalam bidang anime telah berkembang pesat dalam dekade terakhir. Berdasarkan laporan dari Anime News Network, setiap tahunnya terdapat lebih dari 200 judul anime baru yang dirilis hanya di Jepang. Situasi ini menyebabkan information overload, yaitu kondisi di mana pengguna menghadapi terlalu banyak pilihan sehingga kesulitan dalam memilih konten yang sesuai dengan preferensi mereka (Ricci, Rokach, & Shapira, 2015).

Salah satu solusi efektif dalam permasalahan ini adalah penggunaan sistem rekomendasi. Sistem rekomendasi mampu memberikan saran personal berdasarkan pola interaksi pengguna dengan konten sebelumnya, dan telah banyak digunakan pada platform seperti Netflix, YouTube, dan Spotify (Su & Khoshgoftaar, 2009). Di ranah anime, sistem rekomendasi tidak hanya meningkatkan pengalaman pengguna, tetapi juga membantu meningkatkan retensi dan engagement pengguna terhadap platform distribusi anime seperti MyAnimeList atau AniList.

Sistem rekomendasi tradisional seperti popularity-based atau content-based filtering cenderung kurang akurat karena tidak memperhatikan perilaku kolektif pengguna (Schäfer et al., 2007). Sebaliknya, pendekatan Collaborative Filtering (CF) menawarkan akurasi yang lebih tinggi dengan cara memanfaatkan informasi dari pengguna lain yang memiliki preferensi serupa.

Namun, pendekatan CF konvensional seperti matrix factorization memiliki keterbatasan dalam menangkap non-linearitas dan kompleksitas hubungan antar pengguna dan item. Oleh karena itu, pendekatan modern seperti Neural Collaborative Filtering (NCF), yang menggabungkan teknik deep learning dengan collaborative filtering, menjadi solusi yang lebih powerful dalam mengatasi kelemahan tersebut (He et al., 2017).

Dalam proyek ini, dikembangkan sebuah sistem rekomendasi anime berbasis Neural Collaborative Filtering dengan memanfaatkan data interaksi user-anime berupa rating. Model dirancang menggunakan framework TensorFlow dan memanfaatkan embedding layers untuk merepresentasikan masing-masing pengguna dan anime dalam ruang vektor berdimensi rendah. Prediksi dilakukan dengan menghitung dot product dari vektor pengguna dan anime, disertai bias, untuk menghasilkan skor rating prediktif.

Sistem ini diharapkan mampu memberikan saran yang lebih personal dan relevan kepada pengguna berdasarkan preferensi mereka sebelumnya. Model juga dievaluasi menggunakan metrik seperti Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi rating.

## Business Understanding

Seiring pertumbuhan industri hiburan Jepang, jumlah anime yang tersedia meningkat secara signifikan setiap tahunnya. Banyaknya pilihan ini membuat pengguna mengalami kesulitan dalam menentukan anime yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem yang mampu memberikan rekomendasi anime secara personal.

Dalam proyek ini, dibangun sebuah sistem rekomendasi anime berbasis pembelajaran mesin yang bertujuan untuk membantu pengguna menemukan anime yang relevan dengan minat dan histori penontonan mereka. Sistem rekomendasi ini berfokus pada dua pendekatan utama, yaitu Collaborative Filtering dan Content-Based Filtering.

### Problem Statements

1. Terlalu banyak pilihan anime membuat pengguna kesulitan menemukan tontonan yang sesuai dengan preferensi mereka.

2. Rekomendasi umum (berdasarkan popularitas) belum tentu sesuai dengan selera individual pengguna.

3. Kurangnya pemanfaatan data histori pengguna secara efektif dalam memberikan rekomendasi yang personal.

### Goals

1. Membangun sistem rekomendasi anime yang dapat menyarankan anime berdasarkan histori penilaian pengguna terhadap anime sebelumnya.

2. Mengimplementasikan dua pendekatan utama sistem rekomendasi, yaitu Collaborative Filtering dan Content-Based Filtering, untuk membandingkan performa keduanya dalam menghasilkan rekomendasi.

3. Melakukan pengujian sistem rekomendasi dengan mengevaluasi akurasi hasil rekomendasi berdasarkan data rating pengguna.

### Solution Statements

Untuk mencapai tujuan di atas, berikut adalah dua pendekatan solusi yang digunakan dalam proyek ini:
1. Collaborative Filtering : Pendekatan ini menggunakan interaksi pengguna dengan anime dalam bentuk rating. Sistem mempelajari pola dari rating antar pengguna dan mencoba merekomendasikan anime yang disukai oleh pengguna lain dengan selera serupa. Dalam implementasinya, model dibangun menggunakan teknik embedding terhadap userID dan animeID, dilanjutkan dengan beberapa lapisan dense neural network untuk memprediksi rating.

2. Content-Based Filtering : Pendekatan ini memberikan rekomendasi berdasarkan kemiripan konten, dalam hal ini adalah genre anime. Sistem menyarankan anime lain yang memiliki genre serupa dengan anime yang sebelumnya disukai atau diberi rating tinggi oleh pengguna.


## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan gabungan dari dua sumber utama, yaitu:

* Anime Dataset: berisi informasi mengenai daftar anime beserta genre-nya.

* Rating Dataset: berisi data penilaian atau rating yang diberikan oleh pengguna terhadap anime tertentu.

Dataset ini diambil dari [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data). Terdapat dua file berformat csv pada dataset, yaitu anime.csv dan rating.csv. Beberapa rincian penjelasan dataset ini sebagai berikut:

### Statistika Dataset

* Anime Dataset (anime.csv)
  * Jumlah entri: 12294 baris
  * Jumlah kolom: 7 fitur
* Rating Dataset (rating.csv)
  * Jumlah entri: 7813737 baris
  * Jumlah kolom: 3 fitur

### Deskripsi Fitur

1. Dataset Anime (anime.csv)
   * **anime_id** (_int64_): ID unik untuk setiap anime.
   * **name** (_object_): Nama atau judul dari anime.
   * **genre** (_object_): Daftar genre yang terkait dengan anime (contoh: Action, Comedy, dll).
   * **type** (_object_): Tipe anime seperti TV, Movie, OVA, dll.
   * **episodes** (_object_): Jumlah episode anime (beberapa nilai tidak diketahui, ditandai dengan 'Unknown').
   * **rating** (_float64_): Rata-rata rating yang diberikan pengguna MyAnimeList terhadap anime (bukan individual rating).
   * **members** (_int64_): Jumlah pengguna MyAnimeList yang menambahkan anime ke daftar mereka.

2. Dataset Rating (rating.csv)
   * **user_id** (_int64_): ID unik pengguna.
   * **anime_id** (_int64_): ID anime yang diberi rating oleh pengguna.
   * **rating** (_int64_): Nilai rating yang diberikan oleh pengguna terhadap anime. Nilai -1 berarti pengguna telah menonton tetapi tidak memberikan rating eksplisit.

### Exploratory Data Analysis (EDA)

1. Informasi Struktur Data
   * ```anime.info()``` dan ```rating.info()``` digunakan untuk mengecek jumlah entri (baris), tipe data setiap kolom, dan apakah ada nilai yang hilang (null).
   * Terdapat 12.294 anime dan 7.813.737 penilaian dari pengguna terhadap anime.
   * Data anime memiliki kolom bertipe objek dan numerik, sedangkan rating terdiri dari user_id, anime_id, dan rating bertipe integer.
   
2. Nilai Unik dan Missing Value
   * ```anime.nunique()``` digunakan untuk mengetahui jumlah nilai unik di setiap kolom.
   * Ditemukan missing value pada kolom genre, type, episodes, dan rating pada data anime.
   * Data rating tidak memiliki nilai yang hilang secara eksplisit (NaN), tetapi memiliki nilai -1 yang menandakan pengguna menonton anime namun tidak memberikan rating (ini umumnya dihapus dalam tahap preprocessing).
   
3. Eksplorasi Genre dan Type Anime
   * Genre dalam dataset ditulis dalam bentuk string yang dipisahkan koma.
   * Jumlah genre berbeda-beda dan kombinasi genre sangat beragam. Ini akan berguna untuk pendekatan Content-Based Filtering.
   * Jenis type anime meliputi TV, Movie, OVA, Special, ONA, dan beberapa lainnya.

4. Statistika Deskriptif
   Menggunakan ```anime.describe(include='all')``` dan ```rating.describe(include='all')``` untuk mengetahui:
   * Rata-rata, standar deviasi, nilai minimum dan maksimum dari anime dan rating.
   * Jumlah anime yang ditambahkan oleh pengguna (members) sangat bervariasi, menunjukkan ketimpangan popularitas anime.

5. Visualisasi Distribusi Rating oleh Pengguna
   * Visualisasi barplot dari rating['rating'].value_counts() menunjukkan bahwa pengguna cenderung memberi rating tinggi (8–10).
   * Ini mencerminkan adanya positivity bias di kalangan pengguna (umum dalam sistem rating online).
   * Terdapat rating -1 yang seharusnya rating bernilai > 1.
     ![rating user](https://github.com/user-attachments/assets/f4ee9c22-c325-47de-8c1b-c19508afd3fd)

6. Visualisasi Distribusi Rating per Anime
   * Distribusi rating rata-rata per anime divisualisasikan menggunakan histogram dan KDE plot (```sns.histplot```).
   * Sebagian besar anime memiliki rating rata-rata antara 6 dan 8, yang berarti tidak banyak anime yang dianggap sangat buruk atau sangat bagus oleh komunitas secara umum.
     ![anime rat](https://github.com/user-attachments/assets/79a7323e-15aa-43c0-8834-c5ecc2c3fee8)

7. Genre Terbanyak
   * Menggunakan ```Counter``` dan ```pd.Series```, genre diekstrak dan dihitung kemunculannya.
   * Genre paling diminati adalah Comedy, Action, Adventure, Fantasy, dan lain-lain.
   * Genre ini nantinya menjadi fitur penting dalam sistem rekomendasi berbasis konten.
     ![genre](https://github.com/user-attachments/assets/3f10c6e0-9f82-46ee-a181-6141c8a0c9de)

8. Distribusi Tipe Anime
   * Visualisasi distribusi jenis type menunjukkan bahwa TV Series mendominasi dataset, disusul oleh Movie, OVA, dan lainnya.
   * Ini memberi wawasan penting: sistem rekomendasi dapat menyesuaikan preferensi pengguna terhadap tipe anime.
     ![type](https://github.com/user-attachments/assets/08600727-fa24-4a4b-8018-05244793e08d)

## Data Preparation
1. Menghapus Duplikasi pada Dataset Anime
   ```python
   anime.drop_duplicates(inplace=True)
   ```
   Menghapus duplikasi baris pada dataset anime diperlukan untuk menghindari bias dalam sistem rekomendasi. Duplikasi dapat menyebabkan item tertentu muncul lebih dari sekali, yang akan memengaruhi distribusi rekomendasi dan hasil evaluasi model.

2. Menghapus Nilai Kosong (_Missing Value_)
   ```python
   anime = anime.dropna()
   anime.reset_index(drop=True, inplace=True)
   ```
   Baris dengan nilai kosong dihapus untuk menghindari error atau hasil yang tidak akurat selama pemodelan. Fitur seperti genre, type, rating, dan members sangat penting untuk pemrosesan lebih lanjut (terutama pada Content-Based Filtering), sehingga missing value perlu dihilangkan.

3. Menghapus Rating tidak Valid
   ```python
   rating = rating[rating['rating'] != -1]
   ```
   Dalam dataset rating, nilai -1 menandakan bahwa pengguna belum memberikan penilaian terhadap anime tersebut. Data ini tidak relevan dalam sistem rekomendasi berbasis rating karena tidak mencerminkan preferensi pengguna.

4. Menghapus Duplikasi pada Dataset Rating
   ```python
   rating.drop_duplicates(inplace=True)
   rating.reset_index(drop=True, inplace=True)
   ```
   Duplikasi dalam data rating dapat menyebabkan bias pada rekomendasi, terutama dalam algoritma Collaborative Filtering. Setiap pengguna seharusnya memberikan satu rating untuk satu anime.

5. Menyaring Anime yang Tersedia
   ```python
   rating = rating[rating['anime_id'].isin(anime['anime_id'])]
   ```
   Langkah ini memastikan bahwa hanya anime yang tersedia di dataset anime yang digunakan dalam dataset rating. Hal ini menjaga konsistensi dan integritas data yang akan digunakan dalam sistem rekomendasi.

6. Sampling Rating
   ```python
   rating_per_anime = rating.groupby('anime_id').apply(lambda x: x.sample(min(len(x), 100), random_state=42)).reset_index(drop=True)
   ```
   Untuk mengurangi ukuran data dan mempercepat proses training model, dilakukan sampling maksimal 100 rating per anime. Sampling ini mempertahankan keberagaman data namun tetap efisien secara komputasi.

7. Penggabungan Dataset Anime dan Rating
   ```python
   anime_rating = pd.merge(rating_per_anime, anime, on='anime_id')
   ```
   Dataset anime_rating merupakan gabungan dari dataset anime dan rating berdasarkan kolom anime_id. Dataset ini berisi informasi lengkap yang dibutuhkan untuk model Collaborative Filtering (berbasis user-anime-rating) dan Content-Based Filtering (berbasis fitur seperti genre dan type).

8. Final Check
   ```python
   anime_rating.info()
   anime_rating.isnull().sum()
   anime_rating.head()
   ```
   Langkah ini dilakukan untuk memastikan tidak ada missing value yang tersisa dan struktur data sudah siap digunakan pada tahap modeling.

### Content Based Filtering
1. Menghapus Duplikat dan mengisi dengan string
   ```python
   data = anime_rating[['anime_id', 'name', 'genre']].drop_duplicates()
   data['genre'] = data['genre'].fillna('')
   ```
   Kode tersebut bertujuan untuk mempersiapkan data anime dengan memilih kolom anime_id, name, dan genre dari dataframe anime_rating, kemudian menghapus baris duplikat berdasarkan kombinasi nilai di ketiga kolom tersebut. Selanjutnya, nilai yang hilang atau kosong pada kolom genre akan diisi dengan string kosong (''), memastikan tidak ada nilai NaN yang mengganggu analisis lebih lanjut pada kolom genre.
   
2. TF-IDF
   ```python
   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(data['genre'])
   ```
   Memproses fitur teks pada kolom genre diubah menjadi representasi numerik menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency). TF-IDF ini membantu mengekstrak informasi penting dari deskripsi genre dengan mengurangi pengaruh kata-kata umum (stop words) dalam bahasa Inggris.

### Colaborative Filtering
1. Encoding
    ```python
    user_ids = df['userID'].unique().tolist()
    anime_ids = df['animeID'].unique().tolist()
    
    user_to_encoded = {x: i for i, x in enumerate(user_ids)}
    anime_to_encoded = {x: i for i, x in enumerate(anime_ids)}
    
    df['user'] = df['userID'].map(user_to_encoded)
    df['anime'] = df['animeID'].map(anime_to_encoded)
    ```
    Kode ini bertujuan mengubah ID asli pengguna (userID) dan anime (animeID) menjadi indeks numerik yang berurutan mulai dari 0. Proses ini penting untuk collaborative filtering karena algoritma tersebut biasanya membutuhkan input dalam bentuk indeks numerik agar dapat mengelola data dengan efisien, seperti membangun matriks interaksi pengguna-anime. Dengan cara ini, data menjadi lebih mudah diolah dalam model rekomendasi.

2. Normalisasi Data
    ```python
    df['rating_x'] = df['rating_x'].astype('float32')
    min_rating, max_rating = df['rating_x'].min(), df['rating_x'].max()
    df['norm_rating'] = df['rating_x'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
    ```
    Proses ini melakukan normalisasi nilai rating agar berada dalam rentang 0 hingga 1. Pertama, tipe data rating diubah ke float32 untuk efisiensi memori dan konsistensi tipe data. Kemudian, nilai rating asli diubah menggunakan rumus normalisasi min-max, yaitu menggeser dan menskalakan semua nilai supaya proporsional antara nilai minimum dan maksimum. Normalisasi ini penting dalam collaborative filtering atau model machine learning lainnya agar skala nilai rating seragam, sehingga model dapat belajar dengan lebih stabil dan menghasilkan rekomendasi yang lebih akurat.

3. Pengacakan Data
    ```python
    df = df.sample(frac=1, random_state=42)
    ```
    Kode ini melakukan pengacakan (shuffling) pada seluruh baris data secara acak menggunakan sample dengan parameter frac=1, yang berarti mengambil 100% data dalam urutan acak. random_state=42 digunakan agar hasil pengacakan dapat direproduksi secara konsisten. Pengacakan data penting untuk memastikan model machine learning atau collaborative filtering tidak terpengaruh oleh urutan data asli, sehingga proses pelatihan dan evaluasi model menjadi lebih adil dan hasilnya lebih general.

4. Splitting Data
    ```python
    x = df[['user', 'anime']].values
    y = df['norm_rating'].values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    ```
    Kode ini memisahkan data fitur (user, anime) dan target (norm_rating), lalu membaginya menjadi data latih dan validasi dengan rasio 80:20. Pembagian ini penting agar model dapat dilatih dan diuji secara adil menggunakan data yang berbeda, serta memastikan hasil evaluasi lebih akurat.

## Modeling
Dalam proyek ini, saya membangun dua pendekatan berbeda untuk sistem rekomendasi anime, yaitu Content-Based Filtering dan Collaborative Filtering. Masing-masing pendekatan dirancang untuk menjawab permasalahan dalam merekomendasikan anime kepada pengguna secara personal dan relevan.

1. Content-Based Filtering
   Content-Based Filtering merekomendasikan anime yang memiliki kemiripan genre dengan anime yang disukai oleh pengguna. Sistem ini hanya memerlukan informasi konten dari item (dalam hal ini: genre dari anime) dan tidak tergantung pada data pengguna lain. Berikut adalah langkah-langkahnya:
   * Mengambil data anime_id, name, dan genre.
   * Mengubah teks genre menjadi representasi numerik menggunakan TF-IDF Vectorizer.
   * Menghitung kemiripan antar anime menggunakan Cosine Similarity.
   * Mengembalikan anime yang paling mirip dengan input yang diberikan.

   * Kelebihan :
    * Tidak membutuhkan data pengguna.
    * Cocok untuk item-item baru yang belum memiliki rating (cold start untuk item).
    * Hasil rekomendasi dapat dijelaskan karena didasarkan pada fitur konten (misalnya genre mirip).

   * Kekurangan :
    * Terbatas hanya pada kemiripan konten — tidak bisa menangkap preferensi pengguna yang kompleks.
    * Tidak bisa merekomendasikan anime di luar genre yang sudah disukai.
  
   * Cara Kerja Cosine Similarity:
     Cosine similarity mengukur tingkat kemiripan antara dua vektor dengan menghitung cosinus sudut di antara keduanya. Rumusnya adalah sebagai berikut:

    $$
 \text{cosine similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \cdot \|\vec{B}\|}
   $$

   Contoh Penerapan:
   ```python
   anime_recommendations('Naruto')
   ```
   Output:
   1. Boruto: Naruto the Movie - Naruto ga Hokage ni...
   2. Naruto Shippuuden: Sunny Side Battle
   3. Naruto x UT
   4. Naruto: Shippuuden Movie 4 - The Lost Tower
   5. Boruto: Naruto the Movie

3. Collaborative Filtering
   Collaborative Filtering mempelajari hubungan antara pengguna dan anime berdasarkan rating yang diberikan, tanpa melihat isi konten anime itu sendiri. Model ini dibuat menggunakan pendekatan Matrix Factorization dengan TensorFlow.

   * Langkah-langkah:
     * Encoding user dan anime ke dalam ID numerik.
     * Menormalisasi rating dari skala aslinya ke [0, 1].
     * Melatih model neural network dengan embedding untuk pengguna dan anime.
     * Menyimpan bobot embedding sebagai representasi fitur laten.

   * Arsitektur Model:
     * Embedding pengguna dan anime.
     * Penjumlahan bias.
     * Dense Layer + Dropout.
     * Output layer dengan aktivasi sigmoid.

   * Kelebihan:
     * Mampu mempelajari pola kompleks dari perilaku pengguna.
     * Cocok untuk menangkap preferensi pengguna secara personal.
     * Dapat merekomendasikan anime dari genre yang belum pernah ditonton sebelumnya (serendipity).
  
   * Kekurangan:
     * Tidak bekerja dengan baik jika data rating sangat sedikit (cold start untuk user).
     * Membutuhkan proses training yang lebih kompleks dan waktu komputasi lebih lama.
       Rekomendasi Anime untuk User ID: 6113 (Collaborative Filtering)

   * Cara Kerja RecommenderNet
      * Input Data
        Data masukan terdiri dari pasangan (user, anime) yang sudah di-encode sebagai indeks numerik. Targetnya adalah rating yang telah dinormalisasi ke rentang 0–1.
      * Embedding Layer
        Model memiliki dua embedding layer: satu untuk pengguna dan satu untuk anime. Masing-masing mengubah ID menjadi vektor berdimensi 50 yang merepresentasikan karakteristik laten (latent features).
      * Bias Embedding
        Disertakan pula bias embedding untuk pengguna dan anime untuk menangkap kecenderungan rating spesifik dari masing-masing entitas. 
      * Kombinasi Vektor (Dot Product)
        Vektor user dan anime dikalikan elemen-per-elemen (element-wise product) dan dijumlahkan (dot product), lalu ditambahkan dengan bias masing-masing. 
      * Layer Tambahan
        Hasil dot product + bias kemudian masuk ke layer dense (ukuran 64, aktivasi ReLU), lalu dropout untuk mencegah overfitting, dan akhirnya layer output dengan aktivasi sigmoid (karena rating sudah dinormalisasi ke [0, 1]).
      * Pelatihan Model
        Model dilatih menggunakan loss function Mean Squared Error (MSE) dan optimizer Adam, dengan evaluasi menggunakan metrik Root Mean Squared Error (RMSE).
      Penggunaan EarlyStopping dan ReduceLROnPlateau membantu menghentikan pelatihan dini saat performa stagnan dan menyesuaikan laju pembelajaran otomatis.
      * Prediksi dan Rekomendasi
        Setelah pelatihan, model dapat memprediksi tingkat kesukaan pengguna terhadap anime tertentu dan memberikan rekomendasi berdasarkan rating tertinggi.

    Anime dengan Rating Tinggi oleh Pengguna
    *  **Macross: Do You Remember Love?**  
       *Genre:* Action, Mecha, Military, Music, Romance, Sci-Fi, Space  
    *  **Change!! Getter Robo: Sekai Saigo no Hi**  
       *Genre:* Action, Adventure, Horror, Mecha, Psychological, Sci-Fi, Shounen  
    *  **Gaiking: Legend of Daiku-Maryu**  
       *Genre:* Action, Mecha, Sci-Fi  
    *  **Kikou Senki Dragonar**  
       *Genre:* Adventure, Mecha, Sci-Fi, Shounen, Space  
    *  **Choujuushin Gravion**  
       *Genre:* Action, Comedy, Mecha, Sci-Fi, Shounen  
    ============================================================================
    Rekomendasi Anime Teratas
    *  **Kimi no Na wa.**  
       *Genre:* Drama, Romance, School, Supernatural  
    *  **Gintama°**  
       *Genre:* Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen  
    *  **Steins;Gate**  
       *Genre:* Sci-Fi, Thriller  
    *  **Ginga Eiyuu Densetsu**  
       *Genre:* Drama, Military, Sci-Fi, Space  
    *  **Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare**  
       *Genre:* Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen  
    *  **Gintama': Enchousen**  
       *Genre:* Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen  
    *  **Clannad: After Story**  
       *Genre:* Drama, Fantasy, Romance, Slice of Life, Supernatural  
    *  **Koe no Katachi**  
       *Genre:* Drama, School, Shounen  
    *  **Gintama**  
       *Genre:* Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen
    *  **Code Geass: Hangyaku no Lelouch R2**  
        *Genre:* Action, Drama, Mecha, Military, Sci-Fi, Super Power
   
## Evaluation

### Content Based Filtering
Dalam content based filtering digunakan metrik evaluasi Precision@K adalah metrik evaluasi dalam sistem rekomendasi yang mengukur seberapa relevan item yang direkomendasikan oleh model dalam daftar K teratas.

* Rumus untuk Precision@K

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan dalam K rekomendasi}}{\text{K}}
$$

* Cara Kerja: <br>
  * Diambil sampel 100 pengguna secara acak dari data.
  * Untuk setiap pengguna, dipilih satu anime yang pernah mereka beri rating tinggi (disukai) sebagai query anime.
  * Dari query tersebut, sistem menghasilkan 5 anime teratas yang direkomendasikan menggunakan metode content-based filtering.
  * Precision dihitung dengan membandingkan daftar 5 rekomendasi terhadap daftar anime yang benar-benar disukai oleh pengguna tersebut (ground truth).
  * Proses ini diulang untuk seluruh 100 pengguna, dan nilai Precision@5 dirata-rata untuk mendapatkan skor keseluruhan (Average Precision@5).

* Hasil Evaluasi
  Berdasarkan evaluasi terhadap 100 pengguna, diperoleh hasil:

$$
\text{Average Precision@K} = 0.0560
$$
  
  Artinya, secara rata-rata hanya 5,6% dari 5 rekomendasi teratas yang benar-benar sesuai dengan preferensi pengguna. Ini menunjukkan bahwa sistem mulai mampu menangkap preferensi pengguna, namun akurasinya masih perlu ditingkatkan.

### Collaborative Filtering
Dalam proyek ini digunakan Root Mean Squared Error (RMSE) sebagai metrik evaluasi utama. RMSE digunakan karena model ini berfokus pada prediksi nilai rating user terhadap anime, yang merupakan data kontinu (bukan klasifikasi).

* Rumus RMSE:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

- $y_i$ = rating sebenarnya
- $\hat{y}_i$ = rating yang diprediksi oleh model
- $(n)$ = jumlah data

* Hasil RMSE pada data training dan validasi:

  ![download](https://github.com/user-attachments/assets/f877aef8-6d68-4428-8465-b973d26033f6)

* Cara Kerja RMSE
  RMSE menghitung selisih antara rating prediksi dan aktual, kemudian mengkuadratkan selisih tersebut agar tidak saling meniadakan (positif-negatif), menjumlahkannya, lalu diambil akar rata-rata kuadratnya. Dengan demikian:
  * Semakin kecil nilai RMSE, semakin kecil rata-rata kesalahan prediksi model.
  * Karena nilai rating dinormalisasi dalam rentang 0–1, maka RMSE juga berada di rentang 0–1.

* Alasan Memilih RMSE
  * Cocok untuk data regresi seperti prediksi rating.
  * Peka terhadap error besar, sehingga model didorong untuk meminimalkan prediksi yang jauh meleset.
  * Memudahkan interpretasi karena memiliki satuan yang sama dengan skala rating.

### Analisis Terhadap Bussines Understanding
1. Apakah Model Menjawab Setiap Problem Statement?
   * Melalui sistem rekomendasi berbasis Content-Based dan Collaborative Filtering, pengguna kini diberikan rekomendasi yang lebih terfokus, sehingga dapat menyaring pilihan dari ribuan anime menjadi beberapa opsi yang paling relevan. Ini mengurangi beban pencarian secara manual.
   * Model Collaborative Filtering berhasil mempelajari pola rating pengguna dan memberikan rekomendasi berdasarkan preferensi pengguna lain yang serupa. Hal ini menghasilkan rekomendasi yang lebih personal dibandingkan sekadar mengandalkan daftar anime populer.
   * Melalui teknik embedding userID dan animeID, model Collaborative Filtering berhasil memanfaatkan histori rating pengguna dalam prediksi. Evaluasi menggunakan RMSE menunjukkan bahwa model mampu memprediksi rating dengan tingkat kesalahan yang relatif rendah, menandakan adanya pemanfaatan data histori yang efektif.

2. Apakah Setiap Goals Berhasil Dicapai?
   * Model Collaborative Filtering telah dibangun dan diuji menggunakan data histori rating, serta menunjukkan performa yang baik melalui metrik RMSE.
   * Kedua pendekatan telah berhasil diimplementasikan dan dievaluasi. Content-Based Filtering dievaluasi menggunakan Precision@5, sementara Collaborative Filtering menggunakan RMSE. Hasil evaluasi menunjukkan bahwa keduanya memberikan hasil yang berbeda, yang bisa digunakan untuk eksplorasi hibridisasi model di masa depan.
   * Evaluasi telah dilakukan secara kuantitatif:
     * Precision@5 = 0.056 untuk Content-Based Filtering.
     * RMSE pada Collaborative Filtering menunjukkan nilai kesalahan prediksi yang cukup rendah.

3. Apakah Solusi yang Direncanakan Memberikan Dampak?
   * Model ini terbukti mampu memberikan rekomendasi berbasis pola rating antar pengguna. RMSE yang rendah menunjukkan bahwa model dapat memprediksi rating dengan cukup baik, sehingga membantu pengguna menemukan anime baru yang sesuai dengan preferensinya. Ini sangat relevan dengan kebutuhan sistem rekomendasi yang personal.
   * Meskipun nilai Precision@5 masih rendah (5,6%), sistem ini menunjukkan kemampuan awal dalam menangkap kemiripan konten, terutama berdasarkan genre. Ini bisa menjadi pondasi awal untuk pengembangan sistem rekomendasi berbasis konten yang lebih kaya, seperti menggunakan sinopsis, studio, atau bahkan data audio-visual di masa depan.

### Kesimpulan
Secara keseluruhan, model yang dibangun telah berhasil menjawab seluruh problem statements, mencapai goals yang ditetapkan, dan solusi yang dirancang terbukti memberikan dampak yang nyata, meskipun dengan tingkat akurasi yang bervariasi. Collaborative Filtering menunjukkan performa yang lebih unggul dan efektif dalam memberikan rekomendasi yang personal, sedangkan Content-Based Filtering memberikan pendekatan alternatif yang masih memiliki ruang untuk ditingkatkan. Dengan demikian, proyek ini telah menunjukkan kontribusi nyata dalam meningkatkan pengalaman pengguna dalam menemukan anime yang sesuai dengan preferensi mereka.

## Referensi
> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. Proceedings of the 26th International Conference on World Wide Web, 173–182. https://doi.org/10.1145/3038912.3052569
> 
> Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6
> 
> Schäfer, J. B., Frankowski, D., Herlocker, J. L., & Sen, S. (2007). Collaborative Filtering Recommender Systems. In The Adaptive Web (pp. 291–324). Springer. https://doi.org/10.1007/978-3-540-72079-9_9
> 
> Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009, Article ID 421425. https://doi.org/10.1155/2009/421425
