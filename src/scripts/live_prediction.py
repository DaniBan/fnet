import cv2
import torch
from src.models.vgg import TinyVGG
from src.dataset.custom_lfw import FaceDataset
from torchvision import transforms

img_size = 250
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("ERROR: could not access camera")
    exit(-1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = TinyVGG(3, 10, 18)
model.load_state_dict(torch.load("../../states/24_04_02__19_45_04/tinyVgg.pth", map_location=torch.device("cpu")))
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

while True:
    ret, frame = camera.read()
    if not ret:
        print("ERROR: could not read frame")
        exit(-1)

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (img_size, img_size))
    model_input = transform(frame)
    # model_input = torch.Tensor(model_input)
    print(model_input.shape)
    # model_input = model_input.permute(2, 0, 1)

    with torch.inference_mode():
        model_input_batch = model_input.unsqueeze(dim=0).to(device)
        y_pred = model(model_input_batch).cpu()
        y_coord = y_pred[0][1::2]
        x_coord = y_pred[0][::2]
        coordinates = zip(x_coord, y_coord)
        for (xi, yi) in coordinates:
            print(f"{xi} {yi}")
            frame = cv2.circle(frame, (int(xi), int(yi)), radius=2, color=(0, 0, 255))

    cv2.imshow(winname="Camera", mat=frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
