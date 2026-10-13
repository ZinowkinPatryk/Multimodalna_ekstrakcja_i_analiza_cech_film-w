"""
Dataset loader,
We will test the three variant of data size:
1 - 500
2 - 750
3 - 1000
The reason of this changing data size is to verify and check
that this small different will be noticeable.
"""
import sys
import pandas as pd
from pathlib import Path
from datasets import load_dataset

class DataLoader:
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.data_path = None
        self.dataset_name = None
        self._output_dir = None
        self._images_path = {}
        self._labels_info = {}

    def load_data(self) -> None:
        if len(sys.argv) <= 2:
            raise FileNotFoundError("System require a arguments (two of them)")
        match sys.argv[1]:
            case "disc":
                self.data_path = sys.argv[2]
                if self.verify_path(self.data_path):
                    print("loading data...")
                    data = pd.read_csv(self.data_path)
                    self._split_data(data)
                else:
                    raise FileNotFoundError("The filepath does not exist")
            case "cloud":
                self.dataset_name = sys.argv[2]
                dataset = load_dataset(self.dataset_name, split="train")
                safe_dir_name = self.dataset_name.replace("/", "_")
                self._output_dir = Path(rf"Multimodalna_ekstrakcja_i_analiza_cech_film-w\dataset\raw\{safe_dir_name}")
                self._output_dir.mkdir(parents=True, exist_ok=True)
                self._process_hf_data(dataset)
            case _:
                raise FileNotFoundError("WRONG ARGUMENT")


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

    def _process_hf_data(self, dataset)->None:
        counter = 0
        for item in dataset:
            if counter >= self.data_size:
                break
            try:
                image = item["image"]
                desc = item.get("text", item.get("plot",'No found description'))
                if isinstance(desc, list) and len(desc) > 0:
                    desc = " ".join(desc)
                genres = item.get("genres", item.get("labels", ["Unknown"]))
                genre = genres[0] if isinstance(genres, list) and len(genres) > 0 else str(genres)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_path = self._output_dir / f"movie_{counter}.jpg"
                image.save(image_path)
                self._images_path[counter] = image_path
                self._labels_info[counter] = [desc, genre]
                counter += 1
            except:
                print("failed to process image or description")
                continue
        if len(self._images_path) == 0 or len(self._labels_info) == 0:
            raise FileNotFoundError("SOMETHING WENT WRONG")

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
