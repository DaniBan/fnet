from pathlib import Path

import torch
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from src.dataset.fddb.fddb import FDDBDataset
from src.dataset.fddb.utils import split_train_test
from src.utils.fasterrcnn import run_inference


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = fasterrcnn_resnet50_fpn()
    model.load_state_dict(
        torch.load("../../states/25_07_19__13_31_17/fasterrcnn_resnet50_fpn.pth")
    )
    # model.roi_heads.score_thresh = 0.001
    model = model.to(device)

    new_size = (256, 256)  # (H, W)
    data_transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor()
    ])
    train_data, test_data = split_train_test(Path(
        "C:\\Users\\big_b\\.cache\\kagglehub\\datasets\\ngoduy\\dataset-for-face-detection\\versions\\1\\Dataset_FDDB\\"
        "Dataset_FDDB"))
    test_dataset = FDDBDataset(test_data, transform=data_transform, target_shape=new_size)

    X, y = test_dataset[55]
    print(X.shape)

    run_inference(model, X, y, device=device)


if __name__ == "__main__":
    main()
