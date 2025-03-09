import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from PIL import Image, ImageTk
from scipy.signal import spectrogram

# Parámetros
n_neighbors = 5  # Número de vecinos para KNN
n_components = 10  # Número de componentes para reducción de dimensión
video_path = "/mnt/c/Users/Luis/Pictures/Camera Roll/WIN_20250213_14_40_23_Pro.mp4"

dataset_path = "dataset"
labels_map = {"thumbs_up": 0, "thumbs_down": 1, "thumbs_left": 2, "thumbs_right": 3, "fist_closed": 4}
labels_map_inv = {v: k for k, v in labels_map.items()}

scaler = StandardScaler()
pca = PCA(n_components=n_components)
lda = None  # LDA se inicializa después de conocer las etiquetas

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    pooled_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(pooled_image, 50, 150)
    return edges

def extract_features(image):
    f, t, Sxx = spectrogram(image, fs=30)  # Espectrograma con una frecuencia de muestreo ficticia de 30Hz
    compressed_spectrogram = np.log1p(Sxx).flatten()  # Aplicamos log para reducir la escala
    return compressed_spectrogram

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

def load_model(model_name):
    with open(model_name, "rb") as f:
        return pickle.load(f)

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition App")
        
        self.model = None
        self.cap = cv2.VideoCapture(video_path)
        
        self.label = tk.Label(self.root, text="Seleccione un modelo:")
        self.label.pack()
        
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.root, textvariable=self.model_var)
        self.model_combobox["values"] = [
            "KNN (PCA)", "SVM (PCA)", "BAC (PCA)",
            "KNN (LDA)", "SVM (LDA)", "BAC (LDA)"
        ]
        self.model_combobox.pack()
        
        self.start_button = tk.Button(self.root, text="Iniciar", command=self.start_classification)
        self.start_button.pack()
        
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def start_classification(self):
        model_files = {
            "KNN (PCA)": "knn_pca_model.pkl",
            "SVM (PCA)": "svm_pca_model.pkl",
            "BAC (PCA)": "bac_pca_model.pkl",
            "KNN (LDA)": "knn_lda_model.pkl",
            "SVM (LDA)": "svm_lda_model.pkl",
            "BAC (LDA)": "bac_lda_model.pkl"
        }
        
        selected_model = self.model_var.get()
        if selected_model in model_files:
            self.model = load_model(model_files[selected_model])
            self.update_frame()
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        edges = preprocess_image(frame)
        features = extract_features(edges).reshape(1, -1)
        
        # Normalización y reducción de dimensión
        features = scaler.fit_transform(features)
        if "PCA" in self.model_var.get():
            features = pca.fit(features).transform(features)
        elif "LDA" in self.model_var.get():
            global lda
            if lda is None:
                lda = LDA(n_components=min(len(labels_map) - 1, features.shape[1]))
            features = lda.fit(features, np.zeros((features.shape[0],))).transform(features)
        
        prediction = self.model.predict(features)[0]
        
        label_text = labels_map_inv[prediction]
        cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)
    
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
