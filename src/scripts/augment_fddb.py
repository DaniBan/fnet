from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

from src.dataset.fddb.utils import load_annotations


def main():
    target_dir = Path("C:\\Users\\<username>\\.cache\\kagglehub\\datasets\\ngoduy\\dataset-for-face-detection\\versions\\1"
                      "\\Dataset_FDDB\\Dataset_FDDB")

    # Define output paths
    out_dir = target_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    augmented_images_dirname = "images_augmented"
    augmented_labels_filename = "labels_augmented.txt"

    # Load images and annotations
    image_paths, image_annotations = load_annotations(target_dir)
    data = {path: image_annotations[path] for path in image_paths}

    # Define transformations
    transform_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3)
    ], bbox_params=A.BboxParams(format="pascal_voc"))

    # Augment dataset
    augmented_data = []
    labels = ""
    for path, bboxes in list(data.items()):
        img = Image.open(path)
        transformed_data = transform_pipeline(image=np.array(img), bboxes=bboxes)

        # Convert the transformed image back to uint8
        transformed_image = transformed_data["image"]

        # Convert to PIL Image
        transformed_image = Image.fromarray(transformed_image)
        augmented_data.append({"image": transformed_image, "bboxes": transformed_data["bboxes"]})

        # Follow the original dataset path pattern for writing
        img_path_out = Path(augmented_images_dirname, *path.parts[-5:-1]).with_name(path.stem + "_augmented.jpg")

        # Extend labels with the current image
        labels += f"# {Path(*img_path_out.parts[1:])}:\n"
        for bbox in transformed_data["bboxes"]:
            labels += f"{int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n"

        # Save image
        (out_dir / img_path_out).parent.mkdir(parents=True, exist_ok=True)
        with open(out_dir / img_path_out, "wb") as f:
            transformed_image.save(f, format="JPEG")

    # Save labels
    with open(out_dir / augmented_labels_filename, "w") as f:
        print(labels)
        f.write(labels)


if __name__ == "__main__":
    main()
