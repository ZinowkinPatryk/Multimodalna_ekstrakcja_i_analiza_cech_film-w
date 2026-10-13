from data_loader import DataLoader
from fusion import ProfileAnalyzer


if __name__ == "__main__":
    data_manager = DataLoader(data_size=200)
    data_manager.load_data()
    analyzer = ProfileAnalyzer(data_manager.images_path, data_manager.labels_info)
    analyzer.fusion_vectors()

    my_profile = {
        "action": 1.0,
        "Comedy": 0.5,
    }
    analyzer.build_user_profile(my_profile)
    results = analyzer.calculate_similarity(top_n=5)
    print(f"results: {results}")

