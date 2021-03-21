import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Any, Callable, Union

import numpy as np
from sklearn.model_selection import KFold


class RandomSplit:
    def __init__(
        self,
        data_path: Union[Path, List[Path]],
        save_path: Path,
        save_filename: Union[str, Path],
        image_types: List[str] = ["jpg"],
        first_split_ratio: float = 1 / 3,
        n_splits: int = 5,
        key_fn: Callable[[str], Any] = lambda x: x,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.save_path = save_path
        self.save_filename = Path(save_filename)
        self.image_types = image_types
        self.first_split_ratio = first_split_ratio
        self.n_splits = n_splits
        self.key_fn = key_fn
        self.seed = seed

        self.total_splits = int(n_splits / (1 - first_split_ratio))

        self.__save_files = None
        self.__group_images = None

    def __call__(self, seed: int = None):
        total_splits, seed = self.total_splits, seed or self.seed
        kfold = KFold(n_splits=total_splits, shuffle=True, random_state=seed)

        _, splits = zip(*kfold.split(self.group_images))

        self.save_seed(seed)
        self.save_split(0, np.concatenate(splits[0 : -self.n_splits]))
        for i, current_split in enumerate(splits[-self.n_splits :]):
            self.save_split(i + 1, current_split)

    @property
    def save_files(self):
        if not self.__save_files:
            n_digits = 1 + len(str(self.n_splits))
            name, suffix = self.save_filename.name.split(".", maxsplit=1)
            self.__save_files = [
                self.save_path / f"{name}_{splitno:0{n_digits}d}.{suffix}"
                for splitno in range(0, self.n_splits + 1)
            ]
        return self.__save_files

    @property
    def group_images(self):
        if not self.__group_images:
            group_images_dict = defaultdict(list)
            if isinstance(self.data_path, (list, tuple)):
                data_paths = self.data_path
            else:
                data_paths = [self.data_path]

            for data_path in data_paths:
                if data_path.is_file():
                    with data_path.open(mode="r") as stream:
                        for line in stream:
                            key = self.key_fn(line[:-1])
                            group_images_dict[key].append(line[:-1])
                else:
                    for img_type in self.image_types:
                        for image in data_path.glob(f"*.{img_type}"):
                            image_name = image.name
                            # ignore hidden files
                            if image_name[0] != ".":
                                key = self.key_fn(image_name)
                                group_images_dict[key].append(image_name)
            self.__group_images = list(group_images_dict.values())
        return self.__group_images

    def save_seed(self, seed: int):
        name, suffix = self.save_filename.name.split(".", maxsplit=1)
        save_file = self.save_path / f"{name}_seed.{suffix}"
        print(f"Save seed into {save_file}")
        with save_file.open(mode="w") as stream:
            stream.write(f"{seed}\n")

    def save_split(self, splitno: int, current_split):
        save_file = self.save_files[splitno]

        print(f"Save split {splitno} into {save_file}")

        with save_file.open(mode="w") as stream:
            for idx in current_split:
                lines = map(lambda line: line + "\n", self.group_images[idx])
                stream.writelines(lines)


def split_key_fn(input: str, *, key: str, dataset: str):
    if key == "sample":
        return input

    if key == "identity":
        if dataset == "msmt":
            return input.split(" ")[1]
        return input.split("_")[0]

    raise KeyError(f"Unknown key type: {key}")


def random_split(
    root: Union[str, Path],
    dataset: str,
    key_fn: Union[str, Callable[[str], Any]] = "sample",
    subset: str = "train",
    combineall: bool = False,
    **kwargs,
):
    root = root if isinstance(root, Path) else Path(root)
    if dataset == "msmt":
        save_path = root / "MSMT17_V1"
        data_path = save_path / f"list_{subset}.txt"
        if subset == "train" and combineall:
            data_path = [data_path, save_path / "list_val.txt"]
    else:
        assert dataset in ["dukemtmc", "market"]
        if dataset == "market":
            save_path = root / "Market-1501-v15.09.15"
        else:
            save_path = root / "DukeMTMC-reID"

        data_path = save_path / "bounding_box_train"

    if isinstance(key_fn, str):
        key_fn = partial(split_key_fn, key=key_fn, dataset=dataset)

    kwargs["data_path"] = data_path
    kwargs["save_path"] = save_path
    kwargs["key_fn"] = key_fn
    kwargs.setdefault("save_filename", f"list_{subset}.txt")

    return RandomSplit(**kwargs)()


def main():
    parser = argparse.ArgumentParser(description="Dataset split tool")
    parser.add_argument(
        "--root",
        type=str,
        default=Path.cwd() / "datasets",
        help="datasets root path",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["market", "dukemtmc", "msmt"]
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["train", "query", "gallery"],
        default="train",
        help="a subset of dataset",
    )
    parser.add_argument(
        "--combineall", action="store_true", help="combine msmt17 train and val"
    )
    parser.add_argument(
        "--key-fn",
        type=str,
        default="sample",
        choices=["sample", "identity"],
        help="a function to get the key of the sample",
    )
    parser.add_argument(
        "--first-split-ratio",
        type=float,
        default=1 / 3,
        help="first split ratio",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="number of splits apart from the first split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed to reproduce the splitation",
    )

    args = parser.parse_args()

    random_split(**vars(args))


if __name__ == "__main__":
    main()
