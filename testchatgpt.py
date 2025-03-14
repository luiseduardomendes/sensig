import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import spectrogram
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin

# Parâmetros
dataset_path = "dataset"
labels_map = {"thumbs_up": 0, "thumbs_down": 1, "thumbs_left": 2, "thumbs_right": 3, "fist_closed": 4}
n_neighbors = 5
n_components = 10
n_splits = 5  # Número de divisões para o K-Fold

# Função para preprocessar imagem
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

# FFT Feature Extraction
def extract_fft_features(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    return magnitude_spectrum.flatten()

# Spectrogram Feature Extraction
def extract_spectrogram_features(image):
    f, t, Sxx = spectrogram(image.flatten())
    return np.log(Sxx + 1).flatten()

# PCA Feature Extraction
def extract_pca_features(image, pca):
    image_flat = image.flatten().reshape(1, -1)
    return pca.transform(image_flat).flatten()

# LDA Feature Extraction
def extract_lda_features(image, lda):
    image_flat = image.flatten().reshape(1, -1)
    return lda.transform(image_flat).flatten()

# Compressão usando PCA
def compress_features_pca(features):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

# Compressão usando LDA
def compress_features_lda(features, labels):
    n_classes = len(np.unique(labels))  # Número de classes
    n_components_lda = min(n_components, n_classes - 1)  # Número máximo de componentes para LDA
    lda = LDA(n_components=n_components_lda)
    return lda.fit_transform(features, labels)

# Implementação do BAC
class BalancedCenterClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.class_centers_ = {}
        for label in np.unique(y):
            self.class_centers_[label] = np.mean(X[y == label], axis=0)
        return self
    
    def predict(self, X):
        predictions = []
        for sample in X:
            distances = {label: np.linalg.norm(sample - center) for label, center in self.class_centers_.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)

# Carregar e processar dados
data_fft, data_spec, labels = [], [], []
for label_name, label in labels_map.items():
    folder_path = os.path.join(dataset_path, label_name)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = preprocess_image(file_path)
        
        data_fft.append(extract_fft_features(image))
        data_spec.append(extract_spectrogram_features(image))
        labels.append(label)

# Converter para numpy array
data_fft = np.array(data_fft)
data_spec = np.array(data_spec)
labels = np.array(labels)

# Aplicar compressão PCA e LDA nos espectrogramas
data_spec_pca = compress_features_pca(data_spec)
data_spec_lda = compress_features_lda(data_spec, labels)

# Treinar e avaliar modelos com K-Fold
def train_and_evaluate(X, y, model_name):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    svm = SVC(kernel='linear')
    bac = BalancedCenterClassifier()
    
    knn_scores = cross_val_score(knn, X, y, cv=kf)
    svm_scores = cross_val_score(svm, X, y, cv=kf)
    bac_scores = cross_val_score(bac, X, y, cv=kf)
    
    knn_mean, knn_std = np.mean(knn_scores), np.std(knn_scores)
    svm_mean, svm_std = np.mean(svm_scores), np.std(svm_scores)
    bac_mean, bac_std = np.mean(bac_scores), np.std(bac_scores)
    
    print(f"{model_name} - KNN Accuracy: {knn_mean:.2f} ± {knn_std:.2f}")
    print(f"{model_name} - SVM Accuracy: {svm_mean:.2f} ± {svm_std:.2f}")
    print(f"{model_name} - BAC Accuracy: {bac_mean:.2f} ± {bac_std:.2f}")
    
    return (knn_mean, knn_std), (svm_mean, svm_std), (bac_mean, bac_std)

# Parte 1.a: FFT + DR
print("=== Parte 1.a: FFT + DR ===")

# Aplicar PCA às características da FFT
data_fft_pca = compress_features_pca(data_fft)

# Aplicar LDA às características da FFT
data_fft_lda = compress_features_lda(data_fft, labels)

# Treinar e avaliar modelos com FFT + PCA
results_fft_pca = train_and_evaluate(data_fft_pca, labels, "FFT + PCA")

# Treinar e avaliar modelos com FFT + LDA
results_fft_lda = train_and_evaluate(data_fft_lda, labels, "FFT + LDA")

# Parte 1.b: Espectrogramas
print("\n=== Parte 1.b: Espectrogramas ===")
results_spec = train_and_evaluate(data_spec, labels, "Spectrogram Features")

# Parte 1.c: Espectrogramas comprimidos (PCA e LDA)
print("\n=== Parte 1.c: Espectrogramas Comprimidos (PCA e LDA) ===")
results_pca = train_and_evaluate(data_spec_pca, labels, "Compressed Spectrogram Features - PCA")
results_lda = train_and_evaluate(data_spec_lda, labels, "Compressed Spectrogram Features - LDA")

# Gráficos comparativos
models = ["KNN", "SVM", "BAC"]
x = np.arange(len(models))

# Plot for FFT + PCA vs FFT + LDA
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, [r[0] for r in results_fft_pca], yerr=[r[1] for r in results_fft_pca], width=0.4, label="FFT + PCA", capsize=5)
plt.bar(x + 0.2, [r[0] for r in results_fft_lda], yerr=[r[1] for r in results_fft_lda], width=0.4, label="FFT + LDA", capsize=5)
plt.xticks(ticks=x, labels=models)
plt.ylabel("Accuracy")
plt.title("Comparison FFT + PCA vs FFT + LDA")
plt.legend()
plt.show()

# Plot for FFT vs Spectrograms
plt.figure(figsize=(10, 6))
plt.bar(x, [r[0] for r in results_spec], yerr=[r[1] for r in results_spec], width=0.4, label="Spectrogram", capsize=5)
plt.xticks(ticks=x, labels=models)
plt.ylabel("Accuracy")
plt.title("Spectrograms")
plt.legend()
plt.show()

# Plot for Compressed Spectrograms (PCA)
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, [r[0] for r in results_pca], yerr=[r[1] for r in results_pca], width=0.4, label="PCA", capsize=5)
plt.bar(x + 0.2, [r[0] for r in results_lda], yerr=[r[1] for r in results_lda], width=0.4, label="LDA", capsize=5)
plt.xticks(ticks=x, labels=models)
plt.ylabel("Accuracy")
plt.title("Comparison Compressed Spectrograms PCA vs LDA")
plt.legend()
plt.show()

# Combined Plot for FFT + PCA, FFT + LDA, and Compressed Spectrograms (PCA, LDA)
plt.figure(figsize=(12, 8))
width = 0.2
plt.bar(x - width, [r[0] for r in results_fft_pca], yerr=[r[1] for r in results_fft_pca], width=width, label="FFT + PCA", capsize=5)
plt.bar(x, [r[0] for r in results_fft_lda], yerr=[r[1] for r in results_fft_lda], width=width, label="FFT + LDA", capsize=5)
plt.bar(x + width, [r[0] for r in results_pca], yerr=[r[1] for r in results_pca], width=width, label="Compressed Spectrograms - PCA", capsize=5)
plt.bar(x + 2 * width, [r[0] for r in results_lda], yerr=[r[1] for r in results_lda], width=width, label="Compressed Spectrograms - LDA", capsize=5)
plt.xticks(ticks=x, labels=models)
plt.ylabel("Accuracy")
plt.title("Comparison of Different Feature Extraction Methods")
plt.legend()
plt.show()

# Display number of features
print(f"\nNumber of FFT features: {data_fft.shape[1]}")
print(f"Number of FFT + PCA features: {data_fft_pca.shape[1]}")
print(f"Number of FFT + LDA features: {data_fft_lda.shape[1]}")
print(f"Number of Spectrogram features: {data_spec.shape[1]}")
print(f"Number of PCA features: {data_spec_pca.shape[1]}")
print(f"Number of LDA features: {data_spec_lda.shape[1]}")

print("\nModelos treinados e comparados com sucesso.")