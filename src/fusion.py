import numpy as np
from text_model import TextFeatureExtractor
from vision_model import VisionFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity


class ProfileAnalyzer:
    """
    Klasa do analizy profilu wykorzystująca Późną Fuzję (Late Fusion).
    """

    def __init__(self, images_path: dict, labels_info: dict, train_indices: list, test_indices: list) -> None:
        self._images_path = images_path
        self._labels_info = labels_info
        self.train_indices = train_indices
        self.test_indices = test_indices
        self._text_model = None
        self._vision_model = None
        self.fused_vectors = {}
        self.all_genres = []

    def _create_vectors(self) -> tuple:
        """
        Uruchamianie i uzyskiwanie wektorow cech z modelow wizyjnego i tekstowego
        """
        self._text_model = TextFeatureExtractor(label_info=self._labels_info, train_indices=self.train_indices)
        self._vision_model = VisionFeatureExtractor(images_path=self._images_path, labels_info=self._labels_info,
                                                    train_indices=self.train_indices)
        self._vision_model.train_model(epoch=3)
        text_profiles = self._text_model.extract_features()
        vision_profiles = self._vision_model.extract_features()
        return text_profiles, vision_profiles

    def fusion_vectors(self) -> dict:
        """
        Późna Fuzja (Late Fusion): Łączenie wyliczonych prawdopodobieństw z obu modeli w jeden zbalansowany profil.
        """
        text_profiles, vision_profiles = self._create_vectors()
        genres_set = set()
        for idx in self._labels_info.keys():
            if idx in text_profiles:
                genres_set.update(text_profiles[idx].keys())
        self.all_genres = sorted(list(genres_set))
        for index in self._labels_info.keys():
            if index in text_profiles and index in vision_profiles:
                combined_profile = {}
                for g in self.all_genres:
                    text_p = text_profiles[index].get(g, 0.0)
                    vision_p = vision_profiles[index].get(g, 0.0)
                    combined_profile[g] = (text_p + vision_p) / 2.0

                self.fused_vectors[index] = combined_profile
        return self.fused_vectors

    def calculate_similarity(self, user_profile, top_n=5):
        """
        Wyliczanie podobienstwa cosinusowego połączonego profilu procentowego z profilem wskazanym przez użytkownika.
        """
        recommendations = []
        user_prof_clean = {k.lower(): v for k, v in user_profile.items()}
        user_array = np.array([user_prof_clean.get(g.lower(), 0.0) for g in self.all_genres]).reshape(1, -1)
        for index in self.test_indices:
            movie_profile = self.fused_vectors[index]
            movie_array = np.array([movie_profile[g] for g in self.all_genres]).reshape(1, -1)
            sim_score = cosine_similarity(user_array, movie_array)[0][0]
            genre = str(self._labels_info[index][1])
            desc = str(self._labels_info[index][0])
            pretty_profile = {k: round(v, 4) for k, v in movie_profile.items() if v >= 0.01}
            recommendations.append(
                (index, sim_score, genre, dict(sorted(pretty_profile.items(), key=lambda x: x[1], reverse=True))))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]