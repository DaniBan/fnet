import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def run_inference(model, data, target, device):
    model = model.to(device)
    data = data.to(device)
    print(f"Data:\n{data.shape}")
    model.eval()
    with torch.inference_mode():
        y_pred = model(data.unsqueeze(dim=0))

        print(f"Prediction:\n{y_pred[0]}")
        if y_pred[0]["boxes"].shape[0] == 0:
            print("No bounding boxes found.")
            return

        y_pred = [{
            "boxes": sub_pred["boxes"].cpu(),
            "labels": sub_pred["labels"].cpu(),
            "scores": sub_pred["scores"].cpu()
        } for sub_pred in y_pred]
        pred_elem = y_pred[0]
        best_bbox_idx = pred_elem["scores"].argmax()
        print(f"Best bbox: {pred_elem['boxes'][best_bbox_idx]}")

        # Plot prediction
        data = data.cpu()
        _, ax = plt.subplots(1, figsize=(9, 9))
        print(data)
        ax.imshow(data.permute(1, 2, 0))

        x1, y1, x2, y2 = pred_elem["boxes"][best_bbox_idx]
        x1_orig, y1_orig, x2_orig, y2_orig = target["boxes"][0]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", fill=False)
        rect2 = patches.Rectangle((x1_orig, y1_orig), x2_orig - x1_orig, y2_orig - y1_orig, linewidth=2,
                                  edgecolor="red", fill=False)
        ax.add_patch(rect)
        ax.add_patch(rect2)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()