# Deteksi dan Klasifikasi Tumor Otak Menggunakan Citra MRI

## Anggota Tim
- **SINDI SUTRISNO EKA** : F1D02310026
- **KANDA RIFQI ALFAZ** : F1D02310064
- **APTA MAHOGRA BHAMAKERTI** : F1D022035
- **I MADE INDRA DWI PAYANA** : F1D02310056
- **IDA BAGUS AMANTA PRADIPA KRISHNA** : F1D02310059

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan eksperimen klasifikasi tumor otak berdasarkan citra MRI. Fokus utama adalah pada penerapan teknik Pengolahan Citra Digital (PCD) dan Machine Learning, dengan penekanan pada pemilihan teknik preprocessing dan ekstraksi fitur yang tepat.

## Tujuan Proyek
1. **Menguji Kemampuan Implementasi PCD**  
   Menguji kemampuan dalam mengimplementasikan teknik Pengolahan Citra Digital (PCD) untuk deteksi dan klasifikasi tumor otak.
2. **Pemilihan Preprocessing yang Tepat**  
   Memilih dan menyesuaikan tahapan preprocessing yang paling sesuai dengan karakteristik data citra MRI yang digunakan.
3. **Evaluasi Efektivitas Kombinasi Teknik**  
   Mengevaluasi efektivitas berbagai kombinasi teknik preprocessing, ekstraksi fitur, dan model klasifikasi yang diterapkan.

## Metodologi dan Fokus Eksperimen
Proyek ini fokus pada ketepatan pemilihan teknik preprocessing dan proses ekstraksi fitur. Pemahaman dalam analisis dan proses lebih diutamakan daripada hasil akurasi akhir. Eksperimen dilakukan melalui tahapan berikut:
1. Pemilihan teknik preprocessing berdasarkan modul 1-5
2. Ekstraksi fitur setelah preprocessing
3. Pembuatan model klasifikasi menggunakan data hasil ekstraksi fitur
4. Pengulangan eksperimen (3 kali) dengan improvement teknik preprocessing
5. Analisis perbedaan akurasi antar model klasifikasi (Random Forest, SVM, KNN)

## Library yang Digunakan
Proyek ini menggunakan berbagai library Python utama:
- **os**: Interaksi dengan sistem operasi (memuat dataset)
- **cv2 (OpenCV)**: Pemrosesan citra (membaca, resize, konversi warna)
- **numpy**: Operasi numerik pada array citra
- **pandas**: Manipulasi data
- **matplotlib.pyplot**: Visualisasi dasar (menampilkan citra dan plot)
- **seaborn**: Visualisasi statistik
- **scikit-learn**: Tugas machine learning termasuk:
  - `model_selection.train_test_split`: Membagi dataset
  - `metrics`: Evaluasi model (accuracy_score, classification_report, confusion_matrix)
  - `ensemble.RandomForestClassifier`: Model Random Forest
  - `svm.SVC`: Model Support Vector Machine
  - `neighbors.KNeighborsClassifier`: Model K-Nearest Neighbors
- **skimage.feature**: Ekstraksi fitur tekstur GLCM
- **scipy.stats**: Menghitung entropi

---

## Cuplikan Kode

### Import Library
```python
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
```

## Data Loading
Dataset citra MRI dimuat dengan struktur folder:

dataset/
├── normal/
│   ├── normal_1.jpg
│   └── ...
└── tumor/
    ├── tumor_1.jpg
    └── ...

## Cuplikan Kode
```
data = []
labels = []

for sub_folder in os.listdir("dataset/"):
    sub_folder_path = os.path.join("dataset/", sub_folder)
    if os.path.isdir(sub_folder_path):
        for filename in os.listdir(sub_folder_path):
            img_path = os.path.join(sub_folder_path, filename)
            img = cv.imread(img_path)
            if img is not None:
                img = img.astype(np.uint8)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.resize(img, (256, 256))
                
                data.append(img)
                labels.append(sub_folder)

data = np.array(data)
labels = np.array(labels)
```

## Preprocessing dan Ekstraksi Fitur
Contoh fitur yang diekstraksi:
- Fitur Tekstur GLCM: Dissimilarity, Correlation, Homogeneity, Energy, Contrast, ASM
- Entropi
Cuplikan kode ekstraksi fitur:
```
def feature_extraction(data):
    features = []
    for img in data:
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        entr = entropy(img.flatten())
        
        features.append([dissimilarity, correlation, homogeneity, energy, contrast, asm, entr])
    return np.array(features)
```

## Modeling dan Evaluasi
Pemisahan data:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Model klasifikasi:
```
rf = RandomForestClassifier(n_estimators=5, random_state=42)
svm = SVC(kernel='rbf', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
```

Fungsi evaluasi:
```
def generateClassificationReport(y_true, y_pred, model_name=""):
    print(f"Laporan Klasifikasi untuk {model_name}:")
    print(classification_report(y_true, y_pred))
    print(f"Matriks Kebingungan untuk {model_name}:")
    print(confusion_matrix(y_true, y_pred))
    print(f'Akurasi untuk {model_name}: {accuracy_score(y_true, y_pred):.4f}\n')
```

## Hasil Eksperimen
Hasil dari setiap model klasifikasi (Random Forest, SVM, KNN) dianalisis berdasarkan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score. Perbandingan antar model dilakukan untuk menentukan model yang paling efektif.
