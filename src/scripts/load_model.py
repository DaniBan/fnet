import torch
from src.models.vgg import TinyVGG
from src.models.vgg import VGG16
from src.dataset.custom_lfw import FaceDataset
from torchvision import transforms
from src.utils.visualization import plot_prediciton
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = TinyVGG(3, 10, 18)
model.load_state_dict(torch.load("../../states/24_04_02__19_45_04/tinyVgg", map_location=torch.device("cpu")))
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = FaceDataset("../../data/full", transform)

# img_0, labels_0 = dataset[30]
# img_0_batch = img_0.unsqueeze(dim=0)

img = cv2.imread("test_images/dani_resized.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = transform(img)
# img_tensor_reshaped = img_tensor.permute(2, 0, 1)
img_batch = img_tensor.unsqueeze(dim=0)

with torch.inference_mode():
    y_pred = model(img_batch.to(device))
    plot_prediciton(img_tensor, y_pred.squeeze(dim=0))
