import logging
from pathlib import Path

import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from src.dataset.fddb.fddb import FDDBDataset
from src.dataset.fddb.utils import split_train_test
from src.utils.fasterrcnn import run_inference, run_inference_n  # noqa

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format for log messages
)
logger = logging.getLogger(__name__)


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = fasterrcnn_resnet50_fpn()
    model.load_state_dict(
        torch.load("../../states/fasterrcnn_resnet50_fpn/25_09_19T22_48_05/model.pth")
    )
    model = model.to(device)

    new_size = (240, 240)  # (H, W)
    train_data, test_data = split_train_test(Path(
        "C:\\Users\\big_b\\.cache\\kagglehub\\datasets\\ngoduy\\dataset-for-face-detection\\versions\\1\\Dataset_FDDB\\"
        "Dataset_FDDB"))
    test_dataset = FDDBDataset(test_data, to_tensor=True, target_size=new_size)

    torch.manual_seed(42)
    idx_list = torch.randint(0, len(test_dataset), (9,)).tolist()
    data = [test_dataset[idx] for idx in idx_list]
    images, targets = zip(*data)
    images = list(images)
    targets = list(targets)

    logger.info(f"Running inference on the images of the following indexes: {idx_list}")
    run_inference_n(model, images, targets, device=device)
    # run_inference(model, images[1], targets[1], device=device)


if __name__ == "__main__":
    main()
