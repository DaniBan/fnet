import cv2
import numpy as np


def resize_image(img, size=(28, 28)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w
    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


img = cv2.imread("vivi.jpg")
img_resized = resize_image(img, (250, 250))
cv2.imshow("Image", img_resized)
cv2.imwrite("vivi_resized.jpg", img_resized)
cv2.waitKey(0)
