import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Parâmetros globais
labels_map = {"thumbs_up": 0, "thumbs_down": 1, "thumbs_left": 2, "thumbs_right": 3, "fist_closed": 4}
n_components = 10  # Número de componentes para o PCA
dataset_path = "dataset"  # Caminho para o conjunto de dados

# Função para pré-processar a imagem
def preprocess_image(image):
    # Se a imagem for um caminho (string), carregue-a
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image}.")
            return None
    # Se a imagem já for um array numpy (frame da câmera), converta para tons de cinza
    elif isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        print("Erro: Tipo de imagem não suportado.")
        return None
    
    # Aplicar blur e redimensionamento
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

# Extração de características FFT
def extract_fft_features(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    if np.isnan(magnitude_spectrum).any() or np.isinf(magnitude_spectrum).any():
        print("Erro: magnitude_spectrum contém valores NaN ou infinitos.")
        return np.zeros_like(magnitude_spectrum.flatten())
    return magnitude_spectrum.flatten()

# Extração de características do espectrograma
def extract_spectrogram_features(image):
    f, t, Sxx = spectrogram(image.flatten())
    return np.log(Sxx + 1).flatten()

# Função para carregar e processar dados
def load_and_process_data():
    data_fft, data_spec, labels = [], [], []
    for label_name, label in labels_map.items():
        folder_path = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder_path):
            print(f"Erro: Pasta {folder_path} não encontrada.")
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            image = preprocess_image(file_path)
            if image is None:
                continue
            
            data_fft.append(extract_fft_features(image))
            data_spec.append(extract_spectrogram_features(image))
            labels.append(label)
    
    # Converter para numpy array
    data_fft = np.array(data_fft)
    data_spec = np.array(data_spec)
    labels = np.array(labels)
    
    return data_fft, data_spec, labels

# Função para treinar modelos e PCA
def train_models_and_pca():
    # Carregar e processar dados
    data_fft, data_spec, labels = load_and_process_data()
    
    # Dividir dados em treino e teste
    X_train_fft, X_test_fft, y_train, y_test = train_test_split(data_fft, labels, test_size=0.2, random_state=42)
    X_train_spec, X_test_spec, _, _ = train_test_split(data_spec, labels, test_size=0.2, random_state=42)
    
    # Treinar PCA para FFT
    pca_fft = PCA(n_components=n_components)
    X_train_fft_compressed = pca_fft.fit_transform(X_train_fft)
    X_test_fft_compressed = pca_fft.transform(X_test_fft)
    
    # Treinar PCA para espectrogramas
    pca_spec = PCA(n_components=n_components)
    X_train_spec_compressed = pca_spec.fit_transform(X_train_spec)
    X_test_spec_compressed = pca_spec.transform(X_test_spec)
    
    # Treinar modelos
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='linear')
    
    # Treinar kNN e SVM com FFT + PCA
    knn.fit(X_train_fft_compressed, y_train)
    svm.fit(X_train_fft_compressed, y_train)
    
    # Avaliar modelos
    y_pred_knn = knn.predict(X_test_fft_compressed)
    y_pred_svm = svm.predict(X_test_fft_compressed)
    
    print(f"Acurácia kNN (FFT + PCA): {accuracy_score(y_test, y_pred_knn):.2f}")
    print(f"Acurácia SVM (FFT + PCA): {accuracy_score(y_test, y_pred_svm):.2f}")
    
    # Salvar modelos e PCA
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm, f)
    with open("pca_fft.pkl", "wb") as f:
        pickle.dump(pca_fft, f)
    with open("pca_spec.pkl", "wb") as f:
        pickle.dump(pca_spec, f)
    
    print("Modelos e PCA treinados e salvos com sucesso.")

# Função para classificação em tempo real com FFT e DR
def classify_real_time_fft_dr(model, pca):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break
        
        # Pré-processamento do frame
        processed_image = preprocess_image(frame)
        if processed_image is None:
            print("Erro: Falha no pré-processamento do frame.")
            continue
        
        # Extração de características FFT
        features = extract_fft_features(processed_image).reshape(1, -1)
        if features.size == 0:
            print("Erro: Vetor de características vazio.")
            continue
        
        # Redução de dimensionalidade (PCA)
        features_compressed = pca.transform(features)
        
        # Classificação
        prediction = model.predict(features_compressed)[0]
        label_name = [key for key, val in labels_map.items() if val == prediction][0]
        
        # Exibir o resultado na tela
        cv2.putText(frame, label_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Classificação em Tempo Real (FFT + DR)", frame)
        
        # Sair ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Função principal
def main():
    # Treinar modelos e PCA (executar apenas uma vez)
    train_models_and_pca()
    
    # Carregar modelo e PCA
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pca_fft.pkl", "rb") as f:
        pca = pickle.load(f)
    
    # Iniciar classificação em tempo real
    classify_real_time_fft_dr(model, pca)

if __name__ == "__main__":
    main()