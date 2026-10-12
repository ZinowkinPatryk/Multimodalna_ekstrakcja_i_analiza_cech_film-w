import numpy as np
from text_model import TextFeatureExtractor
from vision_model import VisionFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

class ProfileAnalyzer:
    def __init__(self, images_path: dict, labels_info: dict) -> None:
        self._images_path = images_path
        self._labels_info = labels_info
        self._text_model = None
        self._vision_model = None
        self.fused_vectors = {}
        self.user_profile_vector = None

    def _create_vectors(self) -> tuple:
        self._text_model = TextFeatureExtractor(label_info=self._labels_info)
        self._vision_model = VisionFeatureExtractor(images_path=self._images_path)
        text_vector = self._text_model.extract_features()
        vision_vector = self._vision_model.extract_features()
        return text_vector, vision_vector

    def fusion_vectors(self) -> dict:
        text_vector, vision_vector = self._create_vectors()
        for index in self._labels_info.keys():
            if index in text_vector and index in vision_vector:
                self.fused_vectors[index] = np.concatenate((text_vector[index], vision_vector[index]))
        return self.fused_vectors

    def build_user_profile(self, profile_weights: dict):
        weighted_vectors = []
        for index, info in self._labels_info.items():
            genre = str(info[1])
            for pref_genre, weight in profile_weights.items():
                if pref_genre.lower() in genre.lower():
                    weighted_vectors.append(self.fused_vectors[index] * weight)
                    break
        if not weighted_vectors:
            raise ValueError("No profile weights found")
        self.user_profile_vector = np.mean(weighted_vectors, axis=0)

    def calculate_similarity(self, top_n=5):
        if self.user_profile_vector is None:
            raise RuntimeError("No user profile found")
        recommendations = []
        profile_2d = self.user_profile_vector.reshape(1, -1)
        for index, fused_vector in self.fused_vectors.items():
            movie_2d = fused_vector.reshape(1, -1)
            sim_score = cosine_similarity(profile_2d, movie_2d)[0][0]
            genre = str(self._labels_info[index][1])
            desc = str(self._labels_info[index][0])
            recommendations.append((index, sim_score, genre, desc))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]




