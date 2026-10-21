from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import DataLoader

class TextFeatureExtractor:
    """
    Klasa do pozyskanai cech z tekstu
    """
    def __init__(self, label_info: dict, max_vocab_size=10000):
        """
        Inicjalizacja modelu TfidfVectorizer
        """
        self.label_info = label_info
        self.vectorizer = TfidfVectorizer(max_features=max_vocab_size)
        texts = []
        for info in self.label_info.values():
            desc = str(info[0] if isinstance(info, list) else str(info["desc"]))
            texts.append(desc)
        self.vectorizer.fit(texts)

    def extract_features(self):
        """
        Przepuszczenie danych przez model z uzyskaniem wektora cech
        """
        features = {}
        for index, info in self.label_info.items():
            desc = str(info[0] if isinstance(info, list) else str(info["desc"]))
            vector = self.vectorizer.transform([desc]).toarray()
            features[index] = vector.flatten()
        return features


if __name__ == "__main__":
    data_manager = DataLoader(
        data_path=r"C:\Users\patry\Desktop\projekt_systemy_obliczen_inteligentych\projekt1_cechy\Multimodalna_ekstrakcja_i_analiza_cech_film-w\dataset\raw\IMDB_four_genre_larger_plot_description.csv",
        data_size=10
    )
    data_manager.load_data()
    text_extractor = TextFeatureExtractor(data_manager.labels_info)
    text_results = text_extractor.extract_features()
    print(text_results)