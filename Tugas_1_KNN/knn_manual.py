import numpy as np
from collections import Counter

class KNNManual:
    def __init__(self, k=3):
        """
        Inisialisasi algoritma KNN.
        k: jumlah tetangga terdekat (default = 3)
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Menyimpan data training. 
        KNN adalah 'lazy learner', jadi tidak ada proses training formal, 
        hanya menyimpan data untuk pembanding nanti.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        """
        Menghitung jarak Euclidean antara dua titik data.
        Rumus: sqrt(sum((x1 - x2)^2))
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """
        Menerima sekumpulan data uji dan mengembalikan prediksinya.
        """
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x_test_single):
        """
        Fungsi internal untuk memprediksi SATU sampel data uji.
        """
        # 1. Hitung jarak dari x_test_single ke SEMUA titik di data training
        distances = [self._euclidean_distance(x_test_single, x_train) for x_train in self.X_train]
        
        # 2. Ambil indeks dari K-tetangga dengan jarak paling dekat (terkecil)
        # np.argsort mengurutkan indeks dari nilai terkecil ke terbesar
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Ambil label/kelas dari K-tetangga tersebut
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Voting terbanyak: cari label mana yang paling sering muncul
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]