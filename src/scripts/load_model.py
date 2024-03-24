import torch
from src.models.tiny_vgg import TinyVGG
from src.dataset.custom_lfw import FaceDataset
from torchvision import transforms
from src.utils.visualization import plot_prediciton, display_item

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = TinyVGG(3, 10, 18)
model.load_state_dict(torch.load("../../states/24_03_24__15_16_29/tinyVgg", map_location=torch.device("cpu")))
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = FaceDataset("../../data/full", transform)

img_0, labels_0 = dataset[0]
img_0_batch = img_0.unsqueeze(dim=0)
with torch.inference_mode():
    y_pred = model(img_0_batch.to(device))
    # print(y_pred.shape)
    plot_prediciton(img_0, y_pred.squeeze(dim=0))
