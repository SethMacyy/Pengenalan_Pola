# Proyek Praktikum Pengenalan Pola (Pattern Recognition)

Repositori ini berisi kumpulan tugas praktikum mata kuliah Pengenalan Pola yang mencakup implementasi algoritma klasifikasi dasar, ekstraksi fitur citra tekstur, hingga pemanfaatan Deep Learning (*Transfer Learning*).

---

## 📂 Struktur Repositori

```text
└── PENGENALAN_POLA/
    ├── Tugas_1_KNN/
    │   ├── dataset/
    │   ├── knn_manual.py         # Implementasi KNN dari nol (Matematika dasar)
    │   └── evaluasi_kkn.ipynb    # Komparasi KNN Manual vs Scikit-Learn (5-Fold CV)
    ├── Tugas_2_Tekstur/
    │   ├── feature_extraction.py # Ekstraksi Fitur LBP, HOG, dan GLCM
    │   └── classification.ipynb  # Klasifikasi Tekstur dengan SVM & Random Forest
    └── Tugas_3_CNN/
        └── transfer_learning.ipynb # Komparasi Custom CNN vs Transfer Learning (MobileNetV2)
