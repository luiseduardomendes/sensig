import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikegen as spikegen
from torch.utils.data import TensorDataset, DataLoader
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


# Parámetros
n_neighbors = 5  # Número de vecinos para KNN
n_components = 10  # Número de componentes para reducción de dimensión

# Directorios con imágenes de las clases
dataset_path = "dataset"
labels_map = {"thumbs_up": 0, "thumbs_down": 1, "thumbs_left": 2, "thumbs_right": 3, "fist_closed": 4}


# Invertir labels_map para visualización
labels_map_inv = {v: k for k, v in labels_map.items()}

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Aplicar un pooling 4x4
    pooled_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)
    
    edges = cv2.Canny(pooled_image, 50, 150)
    return edges

def extract_features(image):
    # Aplicar FFT y obtener la magnitud del espectro
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude_spectrum.flatten()

# Implementación del clasificador Balanced Center (BAC)
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

# Convert data to PyTorch tensors
def convert_to_spikes(data, num_steps=20):
    """
    Convert features to spike trains using rate coding.
    """
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to [0,1]
    spike_data = spikegen.rate(data, num_steps=num_steps)  # Shape: (num_steps, num_samples, num_features)
    
    # Permute to match PyTorch format (num_samples, num_steps, num_features)
    return spike_data.permute(1, 0, 2)  # Swap first two dimensions

def latency_encoding(data, num_steps=20):
    """
    Convert input features to spike latency encoding.
    Earlier spikes represent higher values.
    """
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to [0,1]
    latencies = (1 - data) * num_steps  # Convert values to latencies
    spikes = torch.zeros((data.shape[0], num_steps, data.shape[1]))  # Spike tensor
    
    for i in range(num_steps):
        spikes[:, i, :] = (latencies <= i).float()  # Fire if latency <= step
    
    return spikes

def population_coding(data, num_neurons=5):
    """
    Convert features to a population-based representation.
    Each feature is represented across multiple neurons.
    """
    data = (data - data.min()) / (data.max() - data.min())  # Normalize
    expanded = torch.repeat_interleave(data.unsqueeze(-1), num_neurons, dim=-1)  # Repeat along new dim
    
    return expanded  # Shape: (samples, features, num_neurons)

def phase_coding(data, num_steps=20):
    """
    Convert features into phase-based spike encoding.
    The phase shift is determined by feature intensity.
    """
    data = (data - data.min()) / (data.max() - data.min())  # Normalize
    phases = data * 2 * np.pi  # Convert values to phase shifts

    spikes = torch.zeros((data.shape[0], num_steps, data.shape[1]))  # Create tensor
    for i in range(num_steps):
        spikes[:, i, :] = (torch.sin(i * 2 * np.pi / num_steps + phases) > 0).float()
    
    return spikes

# Cargar y procesar los datos
data, labels = [], []
for label_name, label in labels_map.items():
    folder_path = os.path.join(dataset_path, label_name)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        edges = preprocess_image(file_path)
        features = extract_features(edges)
        data.append(features)
        labels.append(label)

# Convertir a numpy array
data = np.array(data)
labels = np.array(labels)

# Aplicar reducción de dimensionalidad (PCA y LDA)
pca = PCA(n_components=n_components)
lda = LDA()
data_pca = pca.fit_transform(data, labels)
data_lda = lda.fit_transform(data, labels)

# Check dimensions
pca_features = data_pca.shape[1]  # Number of PCA components
lda_features = data_lda.shape[1]  # Number of LDA components

print(f"PCA feature count: {pca_features}")
print(f"LDA feature count: {lda_features}")

# Dividir los datos en entrenamiento y prueba
X_train_pca, X_test_pca, y_train, y_test = train_test_split(data_pca, labels, test_size=0.2, random_state=42)
X_train_lda, X_test_lda, _, _ = train_test_split(data_lda, labels, test_size=0.2, random_state=42)

# Entrenar modelos con PCA
knn_pca = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_pca.fit(X_train_pca, y_train)
svm_pca = SVC(kernel='linear')
svm_pca.fit(X_train_pca, y_train)
bac_pca = BalancedCenterClassifier()
bac_pca.fit(X_train_pca, y_train)

# Entrenar modelos con LDA
knn_lda = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_lda.fit(X_train_lda, y_train)
svm_lda = SVC(kernel='linear')
svm_lda.fit(X_train_lda, y_train)
bac_lda = BalancedCenterClassifier()
bac_lda.fit(X_train_lda, y_train)

# Evaluar modelos
y_pred_knn_pca = knn_pca.predict(X_test_pca)
y_pred_svm_pca = svm_pca.predict(X_test_pca)
y_pred_bac_pca = bac_pca.predict(X_test_pca)
y_pred_knn_lda = knn_lda.predict(X_test_lda)
y_pred_svm_lda = svm_lda.predict(X_test_lda)
y_pred_bac_lda = bac_lda.predict(X_test_lda)

accuracy_knn_pca = accuracy_score(y_test, y_pred_knn_pca)
accuracy_svm_pca = accuracy_score(y_test, y_pred_svm_pca)
accuracy_bac_pca = accuracy_score(y_test, y_pred_bac_pca)
accuracy_knn_lda = accuracy_score(y_test, y_pred_knn_lda)
accuracy_svm_lda = accuracy_score(y_test, y_pred_svm_lda)
accuracy_bac_lda = accuracy_score(y_test, y_pred_bac_lda)

print(f"Precisión del modelo KNN (PCA): {accuracy_knn_pca:.2f}")
print(f"Precisión del modelo SVM (PCA): {accuracy_svm_pca:.2f}")
print(f"Precisión del modelo BAC (PCA): {accuracy_bac_pca:.2f}")
print(f"Precisión del modelo KNN (LDA): {accuracy_knn_lda:.2f}")
print(f"Precisión del modelo SVM (LDA): {accuracy_svm_lda:.2f}")
print(f"Precisión del modelo BAC (LDA): {accuracy_bac_lda:.2f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_pca_tensor = torch.tensor(X_train_pca, dtype=torch.float32).to(device)
X_test_pca_tensor = torch.tensor(X_test_pca, dtype=torch.float32).to(device)
X_train_lda_tensor = torch.tensor(X_train_lda, dtype=torch.float32).to(device)
X_test_lda_tensor = torch.tensor(X_test_lda, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Convert PCA and LDA features to spike trains
num_steps = 20  # Number of time steps for spike encoding
X_train_pca_spikes = convert_to_spikes(X_train_pca_tensor)
X_test_pca_spikes = convert_to_spikes(X_test_pca_tensor)

X_train_lda_spikes = convert_to_spikes(X_train_lda_tensor)
X_test_lda_spikes = convert_to_spikes(X_test_lda_tensor)

X_train_pca_latency = latency_encoding(X_train_pca_tensor)
X_test_pca_latency = latency_encoding(X_test_pca_tensor)

X_train_lda_latency = latency_encoding(X_train_lda_tensor)
X_test_lda_latency = latency_encoding(X_test_lda_tensor)

X_train_pca_population = population_coding(X_train_pca_tensor)
X_test_pca_population = population_coding(X_test_pca_tensor)

X_train_lda_population = population_coding(X_train_lda_tensor)
X_test_lda_population = population_coding(X_test_lda_tensor)

X_train_pca_phase = phase_coding(X_train_pca_tensor)
X_test_pca_phase = phase_coding(X_test_pca_tensor)

X_train_lda_phase = phase_coding(X_train_lda_tensor)
X_test_lda_phase = phase_coding(X_test_lda_tensor)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"X_train_pca_spikes shape: {X_train_pca_spikes.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")

# Create Dataloaders
batch_size = 32
dataset_pca = TensorDataset(X_train_pca_spikes, y_train_tensor)
dataloader_pca = DataLoader(dataset_pca, batch_size=batch_size, shuffle=True)

dataset_lda = TensorDataset(X_train_lda_spikes, y_train_tensor)
dataloader_lda = DataLoader(dataset_lda, batch_size=batch_size, shuffle=True)

dataset_pca = TensorDataset(X_train_pca_latency, y_train_tensor)
dataloader_pca = DataLoader(dataset_pca, batch_size=batch_size, shuffle=True)

dataset_lda = TensorDataset(X_test_pca_latency, y_train_tensor)
dataloader_lda = DataLoader(dataset_lda, batch_size=batch_size, shuffle=True)

# Define SNN model
class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk_out = []
        for step in range(x.shape[1]):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_out.append(spk2)
        
        return torch.stack(spk_out, dim=1).sum(dim=1)  # Aggregate over time

# Initialize models
hidden_dim = 50
output_dim = len(labels_map)

# Initialize models with correct input dimensions
snn_pca = SNN(pca_features, hidden_dim, output_dim)
snn_lda = SNN(lda_features, hidden_dim, output_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_pca = torch.optim.Adam(snn_pca.parameters(), lr=0.001)
optimizer_lda = torch.optim.Adam(snn_lda.parameters(), lr=0.001)

# Training loop
def train_snn(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Train SNNs
print("Training SNN with PCA features...")
train_snn(snn_pca, dataloader_pca, optimizer_pca)

print("Training SNN with LDA features...")
train_snn(snn_lda, dataloader_lda, optimizer_lda)

# Evaluation function
def evaluate_snn(model, X_test_spikes, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_spikes)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    return accuracy

# Evaluate models
accuracy_snn_pca = evaluate_snn(snn_pca, X_test_pca_spikes, y_test_tensor)
accuracy_snn_lda = evaluate_snn(snn_lda, X_test_lda_spikes, y_test_tensor)

print(f"Accuracy of SNN (PCA): {accuracy_snn_pca:.2f}")
print(f"Accuracy of SNN (LDA): {accuracy_snn_lda:.2f}")
