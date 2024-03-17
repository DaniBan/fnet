import os.path

from src.dataset.custom_lfw import FaceDataset


def build_datasets(target_dir, transform_train=None, transform_test=None):
    train_path = os.path.join(target_dir, "train")
    test_path = os.path.join(target_dir, "test")
    if transform_train is None:
        train_dataset = FaceDataset(train_path)
    else:
        train_dataset = FaceDataset(train_path, transform_train)

    if transform_test is None:
        test_dataset = FaceDataset(test_path)
    else:
        test_dataset = FaceDataset(test_path, transform_test)

    return train_dataset, test_dataset
