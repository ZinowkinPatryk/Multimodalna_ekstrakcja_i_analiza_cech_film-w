from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from fusion import ProfileAnalyzer

if __name__ == "__main__":
    data_manager = DataLoader(data_size=1000)
    data_manager.load_data()
    all_inidces = list(data_manager.labels_info.keys())
    train_idx, test_idx = train_test_split(all_inidces, test_size=0.2, random_state=42)

    analyzer = ProfileAnalyzer(
        images_path=data_manager.images_path,
        labels_info=data_manager.labels_info,
        train_indices=train_idx,
        test_indices=test_idx
    )

    analyzer.fusion_vectors()

    my_profile = {
        "Action": 0.4,
        "Comedy": 0.1,
        "Documentary" : 0.5
    }

    results = analyzer.calculate_similarity(user_profile=my_profile, top_n=5)

    print("\n--- TOP 5 REKOMENDACJI ---")
    for i in results:
        print("------------------")
        print(f"Indeks: {i[0]} | Zgodność (Cosine Similarity): {i[1]:.4f}")
        print(f"Rzeczywisty gatunek: {i[2]}")
        print("Multimodalny profil filmu:")
        for genre, prob in i[3].items():
            print(f" - {genre}: {prob}")
        print("------------------")