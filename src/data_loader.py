"""
Dataset loader,
We will test the three variant of data size:
1 - 500
2 - 750
3 - 1000
The reason of this changing data size is to verify and check
that this small different will be noticeable.
"""
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_path: str, data_size: int):
        self.data_size = data_size
        self.data_path = data_path
        self._images_path = {}
        self._labels_info = {}

    def load_data(self) -> None:
        if self.verify_path(self.data_path):
            print("loading data...")
            data = pd.read_csv(self.data_path)
            self._split_data(data)
        else:
            raise FileNotFoundError("The filepath does not exist")

    def _split_data(self, data: pd.DataFrame) -> None:
        image_path = Path(self.data_path).parent
        image_path = image_path / "IMDB four_genre_posters"
        print("splitting data...")
        counter = 0
        for i, (image, genre, desc) in enumerate(zip(data["movie_id"], data["genre"], data["description"])):
            if counter == self.data_size:
                break
            path = image_path / genre.capitalize() / (image + ".jpg")
            if self.verify_path(path):
                self._images_path[counter] = path
                self._labels_info[counter] = [desc, genre]
                counter += 1

    @staticmethod
    def verify_path(path : str) -> bool:
        return Path(path).exists()

    @property
    def images_path(self) -> dict[int, Path]:
        return self._images_path

    @property
    def labels_info(self) -> dict[int, list]:
        return self._labels_info

    def __repr__(self) -> str:
        return (f"<DataLoader: size={self.data_size}, loaded_items={len(self._images_path)}, path='{self.data_path}\n"
                f"images_path_size={len(self._images_path)}, labels_path_size={len(self._labels_info)}'>")


if __name__ == "__main__":
    data_manager = DataLoader(
        data_path=r"C:\Users\patry\Desktop\projekt_systemy_obliczen_inteligentych\projekt1_cechy\Multimodalna_ekstrakcja_i_analiza_cech_film-w\dataset\raw\IMDB_four_genre_larger_plot_description.csv",
        data_size=600)
    data_manager.load_data()
    print(data_manager)