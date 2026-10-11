from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D
from tensorflow.keras import Sequential
from data_loader import DataLoader


class TextFeatureExtractor:
    def __init__(self, label_info: dict, max_vocab_size=10000, sequence_length=15, embedding_dim=128):
        self.label_info = label_info
        self.vectorize_layer = TextVectorization (
            max_tokens=max_vocab_size,
            output_mode='int',
            output_sequence_length=sequence_length
        )
        texts = []
        for info in self.label_info.values():
            desc = str(info[0] if isinstance(info, list) else str(info["desc"]))
            texts.append(desc)
        self.vectorize_layer.adapt(texts)

        self.model = Sequential([
            self.vectorize_layer,
            Embedding(input_dim=len(self.vectorize_layer.get_vocabulary()), output_dim=embedding_dim),
            GlobalAveragePooling1D()
        ])

    def extract_feature(self):
        features = {}
        for index, info in self.label_info.items ():
            desc = str (info[0] if isinstance (info, list) else str (info["desc"]))
            vector = self.model.predict ([desc], verbose=0)
            features[index] = vector.flatten ()
        return features


if __name__ == "__main__":
    data_manager = DataLoader(
        data_path=r"/Volumes/X9 Pro/Projekt systemy obliczen inteligetnych/Multimodalna_ekstrakcja_i_analiza_cech_film-w/dataset/raw/IMDB_four_genre_larger_plot_description.csv",
        data_size=10
    )
    data_manager.load_data()
    text_extractor = TextFeatureExtractor(data_manager.labels_info)
    text_results = text_extractor.extract_feature()
    print(text_results)
