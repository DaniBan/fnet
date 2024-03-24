import os.path

from src.dataset.custom_lfw import FaceDataset


def build_datasets(target_dir,
                   transform_train=None,
                   transform_test=None,
                   target_transform_train=None,
                   target_transform_test=None):

    train_path = os.path.join(target_dir, "train")
    test_path = os.path.join(target_dir, "test")

    train_dataset = FaceDataset(train_path, transform_train, target_transform_train)
    test_dataset = FaceDataset(test_path, transform_test, target_transform_test)

    return train_dataset, test_dataset
