import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from data_loader import DataLoader


class VisionFeatureExtractor:
    def __init__(self, images_path: dict):
        self.images_path = images_path
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.target_size = (224, 224)

    def extract_features(self) -> dict:
        features = {}
        for index, image_path in self.images_path.items():
            img = image.load_img(image_path, target_size=self.target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            vector = self.model.predict(img_array, verbose=0)
            features[index] = vector.flatten()
        return features

if __name__ == "__main__":
    data_manager = DataLoader(
        data_path=r"C:\Users\patry\Desktop\projekt_systemy_obliczen_inteligentych\projekt1_cechy\Multimodalna_ekstrakcja_i_analiza_cech_film-w\dataset\raw\IMDB_four_genre_larger_plot_description.csv",
        data_size=600)
    data_manager.load_data()
    extractor = VisionFeatureExtractor(data_manager.images_path)
    results = extractor.extract_features()
    print(results)