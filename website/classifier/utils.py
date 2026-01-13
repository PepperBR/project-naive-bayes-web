import pickle
import numpy as np
import cv2
from pathlib import Path
from django.conf import settings
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
import tempfile
import os


class ModelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = Path(settings.BASE_DIR) / \
            'ml_models' / 'modelo_ia.pkl'
        self.scaler_path = Path(settings.BASE_DIR) / 'ml_models' / 'scaler.pkl'

    def load_model(self):
        """Carrega o modelo treinado do arquivo .pkl"""
        if self.model is None:
            try:
                with open(self.model_path, 'rb') as file:
                    self.model = pickle.load(file)
            except FileNotFoundError:
                raise Exception(f"Modelo não encontrado em {self.model_path}")
            except Exception as e:
                raise Exception(f"Erro ao carregar o modelo: {str(e)}")
        return self.model

    def load_scaler(self):
        """Carrega o scaler salvo durante o treinamento"""
        if self.scaler is None:
            try:
                with open(self.scaler_path, 'rb') as file:
                    self.scaler = pickle.load(file)
            except FileNotFoundError:
                raise Exception(f"Scaler não encontrado em {self.scaler_path}")
            except Exception as e:
                raise Exception(f"Erro ao carregar o scaler: {str(e)}")
        return self.scaler

    def extrair_features_avancadas(self, img_array):
        """Extrai features usando HOG, Canny, Harris e LBP"""
        img = cv2.resize(img_array, (128, 128))

        features_hog = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=False)

        edges = cv2.Canny(img, 100, 200)
        densidade_bordas = np.array([np.sum(edges > 0) / edges.size])

        dst = cv2.cornerHarris(img, 2, 3, 0.04)
        densidade_cantos = np.array(
            [np.sum(dst > 0.01 * dst.max()) / dst.size])

        lbp = local_binary_pattern(img, 8, 1, method="uniform")
        hist_lbp, _ = np.histogram(
            lbp.ravel(), bins=10, range=(0, 10), density=True)

        return np.hstack([features_hog, densidade_bordas, densidade_cantos, hist_lbp])

    def preprocess_image(self, image_file):
        """Preprocessa a imagem e extrai features"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            for chunk in image_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        try:
            img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise Exception("Não foi possível ler a imagem")

            features = self.extrair_features_avancadas(img)
            features = features.reshape(1, -1)

            return features

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def predict(self, image_file):
        """Faz a predição da classe da imagem"""
        model = self.load_model()
        scaler = self.load_scaler()

        features = self.preprocess_image(image_file)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        class_mapping = {
            0: "Acidente de trânsito grave",
            1: "Acidente de trânsito moderado",
            2: "Não é acidente"
        }

        return class_mapping.get(prediction, "Classe desconhecida")


# Instância global do preditor
predictor = ModelPredictor()
