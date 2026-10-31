from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import random

class TextFeatureExtractor:
    """
    Klasa do pozyskanai cech z tekstu
    """
    def __init__(self, label_info: dict, train_indices: list, max_vocab_size=5000):
        """
        Inicjalizacja modelu TfidfVectorizer
        """
        self.label_info = label_info
        self.vectorizer = TfidfVectorizer(max_features=max_vocab_size)
        self.verify = {}
        train_texts = []
        train_genres = []
        for idx in train_indices:
            info = self.label_info[idx]
            desc = str(info[0] if isinstance(info, list) else str(info["desc"]))
            train_texts.append(desc)
            train_genres.append(info[1])
        X_train = self.vectorizer.fit_transform(train_texts)
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_train, train_genres)

    def collect_vectors_genre(self, genre, vector, desc):
        if genre not in self.verify.keys():
            self.verify[genre] = [[vector, desc]]
        else:
            self.verify[genre].append([vector, desc])

    def calculete_similarity_text(self):
        genres = list(self.verify.keys())
        print("\n--- TEST: Podobieństwo reprezentacji tekstowych ---")
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
                print(f"Para: [{g1}] vs [{g2}] -> Podobieństwo: {sim:.4f}")
                print(f"  Film 1 ({g1}): {desc1_short}")
                print(f"  Film 2 ({g2}): {desc2_short}\n")
        print("-------------------------------------------------------------------")


    def extract_features(self):
        features = {}
        for index, info in self.label_info.items():
            desc = str(info[0] if isinstance(info, list) else str(info["desc"]))
            vector = self.vectorizer.transform([desc]).toarray()
            self.collect_vectors_genre(info[1], vector, desc)
            probs = self.classifier.predict_proba(vector)[0]
            profile = {self.classifier.classes_[i]: float(p) for i, p in enumerate(probs)}
            features[index] = profile
        self.calculete_similarity_text()
        return features

    @property
    def verified(self):
        return self.verify
