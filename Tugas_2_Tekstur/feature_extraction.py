import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

def extract_lbp(image_gray):
    # Parameter standar LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
    # Mengubah matriks LBP menjadi histogram sebagai representasi fitur
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_hog(image_gray):
    # Mengubah ukuran gambar ke ukuran standar (misal 64x64) agar panjang fitur HOG sama semua
    img_resized = cv2.resize(image_gray, (64, 64))
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False)
    return features

def extract_glcm(image_gray):
    # Menghitung matriks ko-okurensi aras keabuan
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    # Mengekstrak properti tekstur dari GLCM
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    
    # Menggabungkan semua properti menjadi satu vektor fitur
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])