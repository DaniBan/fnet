import os

from torchvision import transforms

from src.dataset.custom_lfw import FaceDataset
from src.scripts.dataloader import build_dataloader
from src.dataset.data_factory import build_datasets
from src.utils.visualization import display_item
from src.utils.visualization import display_random_items
from src.utils.visualization import print_image
from src.models.tiny_vgg import TinyVGG

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), "data")
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = FaceDataset(target_dir="data/full", transform=data_transform)
    train_dataset, test_dataset = build_datasets(target_dir="data",
                                                 transform_train=data_transform,
                                                 transform_test=data_transform)
    train_dataloader = build_dataloader(train_dataset,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=True)

    img_0 = train_dataset[0][0]
    model_tiny_vgg = TinyVGG(input_shape=3, hidden_units=10, output_shape=9)
    model_tiny_vgg(img_0.unsqueeze(dim=0))
    # display_random_items(train_dataset, 4, 4, 42)
