import os
import random
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import load_img, img_to_array, to_categorical
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VisionFeatureExtractor:
    """
     Klasa do wyciagniecia cech z posterow
    """
    def __init__(self, images_path: dict, labels_info: dict, train_indices: list):
        """
        Inicjalizacja modelu
        """
        self.images_path = images_path
        self.labels_info = labels_info
        self.train_indices = train_indices
        self.target_size = (224, 224)
        self.base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.label_encoder = None
        self.trainable_model = None
        self.verify = {}

    def train_model(self, epoch=5, batch_size=16):
        train_genres = [self.labels_info[idx][1] for idx in self.train_indices]
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(train_genres)
        num_classes = len(self.label_encoder.classes_)
        output = Dense(num_classes, activation='softmax')(self.base_model.output)
        self.trainable_model = Model(inputs=self.base_model.input, outputs=output)
        for layer in self.base_model.layers[:-10]:
            layer.trainable = False
        for layer in self.base_model.layers[-10:]:
            layer.trainable = True
        self.trainable_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        x_train, y_train = [], []
        for idx, label in zip(self.train_indices, encoded_labels):
            img = load_img(self.images_path[idx], target_size=self.target_size)
            x_train.append(preprocess_input(img_to_array(img)))
            y_train.append(label)
        x_train = np.array(x_train)
        y_train = to_categorical(y_train, num_classes=num_classes)
        self.trainable_model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1)

    def collect_vectors_genre(self, genre, vector, desc):
        if genre not in self.verify:
            self.verify[genre] = [[vector, desc]]
        else:
            self.verify[genre].append([vector, desc])

    def calculate_similarity_vision(self):
        """
        Test podobieństwa cosinusowego samych reprezentacji wizualnych (Spotkanie 1 i 2)
        """
        genres = list(self.verify.keys())
        print("\n--- TEST: Podobieństwo reprezentacji wizualnych (Plakaty) ---")
        for i in range(len(genres)):
            for j in range(i, len(genres)):
                g1 = genres[i]
                g2 = genres[j]
                if g1 == g2:
                    if len(self.verify[g1]) >= 2:
                        movie1, movie2 = random.sample(self.verify[g1], 2)
                    else:
                        continue
                else:
                    movie1 = random.choice(self.verify[g1])
                    movie2 = random.choice(self.verify[g2])
                vec1 = movie1[0]
                vec2 = movie2[0]
                sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
                desc1_short = movie1[1][:40] + "..."
                desc2_short = movie2[1][:40] + "..."
                print(f"Para plakatów: [{g1}] vs [{g2}] -> Podobieństwo wizualne: {sim:.4f}")
                print(f"  Film 1 ({g1}): {desc1_short}")
                print(f"  Film 2 ({g2}): {desc2_short}\n")
        print("-------------------------------------------------------------------")


    def extract_features(self) -> dict:
        """
         Przepuszczanie danych przez model z uzyskaniem wektorow cech
        """
        features = {}
        for index, image_path in self.images_path.items():
            img = load_img(image_path, target_size=self.target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            raw_vector = self.base_model.predict(img_array, verbose=0).flatten()
            info = self.labels_info[index]
            self.collect_vectors_genre(info[1], raw_vector, str(info[0]))
            probs = self.trainable_model.predict(img_array, verbose=0)[0]
            profile = {self.label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)}
            features[index] = profile

        self.calculate_similarity_vision()
        return features