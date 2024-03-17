from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def build_dataloader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     shuffle: bool = False) -> DataLoader:
    """
    Build a data loder from a given dataset.
    :param dataset: A PyTorch dataset.
    :param batch_size: The batch size.
    :param num_workers: The number of workers.
    :param shuffle: If true shuffles the data. (default: False)
    :return: A DataLoader object.
    """

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle)

    return dataloader
