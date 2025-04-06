from pathlib import Path
from typing import List


def get_labeled_image_paths(target_dir: Path) -> List[Path]:
    paths = []
    with open(target_dir / "label.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                line = line.lstrip("# ")
                paths.append(target_dir / line)

    return paths


def main() -> None:
    data_path = Path("C:\\Users\\big_b\\.cache\\kagglehub\\datasets\\ngoduy\\dataset-for-face-detection\\versions\\1\\"
                     "Dataset_FDDB\\Dataset_FDDB")
    paths = list(data_path.rglob("*.jpg"))
    labeled_paths = get_labeled_image_paths(data_path)
    print(f"# paths: {len(paths)} | # labels: {len(labeled_paths)}")


if __name__ == "__main__":
    main()
