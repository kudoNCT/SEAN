import sys
import cv2.cv2 as cv2
import time
import torch

import numpy as np
from mmdet.apis import inference_detector, init_detector


def callBack(*arg):
    pass



class Face_detection():
    def __init__(self, model='10g'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'-Device for face detection: {device}')
        if model == '10g':
            self.model = init_detector('weights/scrfd_10g.py', 'weights/scrfd_10g.pth', device=device)
        elif model == '500m':
            self.model = init_detector('hair_segmentation_model/module/save_weights/scrfd_500m_bnkps.py', 'hair_segmentation_model/module/save_weights/scrfd_500m_KPS.pth', device=device)

    def run(self, img: np.array, thresh_confident=0.7):
        result = inference_detector(self.model, img)
        bboxes = np.vstack(result)

        face_bboxes = bboxes[bboxes[:, -1] > thresh_confident]
        out = []
        for box in face_bboxes:
            bbox_int = box.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            out.append((left_top, right_bottom))
        return out


def cut_part_from_img(landmarks, img: np.array, part: str, rate=6):
    dict_landmarks = {'l_eye': [36, 37, 38, 39, 40, 41],
                      'r_eye': [42, 43, 44, 45, 45, 47],
                      'l_brow': [17, 18, 19, 20, 21],
                      'r_brow': [22, 23, 24, 25, 26],
                      'lip': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]}
    assert part in list(dict_landmarks.keys())
    part_landmark_points = dict_landmarks[part]
    x = []
    y = []
    for point in part_landmark_points:
        x.append(landmarks.part(point).x)
        y.append(landmarks.part(point).y)
    x1 = min(x)
    x2 = max(x)
    y1 = min(y)
    y2 = max(y)

    im_h, im_w = img.shape[1], img.shape[0]
    w = x2 - x1
    h = y2 - y1

    x1 = x1 - int(w / rate) if x1 - int(w / rate) > 0 else 0
    y1 = y1 - int(h / rate) if y1 - int(h / rate) > 0 else 0

    x2 = x2 + int(w / rate) if x2 + int(w / rate) < im_w else im_w
    y2 = y2 + int(h / rate) if y2 + int(w / rate) < im_h else im_h

    return [x1, y1, x2, y2]


def make_erode_trimap(mask: np.ndarray, kernel_size=5, iteration=2, alpha=1.0) -> np.ndarray:
    #mask = mask.astype(np.float)/255
    mask = mask.astype(np.float)
    original_mask = mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.float)
    img_erosion = mask.copy()
    for i in range(iteration - 1):
        img_erosion = cv2.erode(img_erosion, kernel=kernel, iterations=1)
        mask += img_erosion
    mask = (mask * alpha) / np.max(mask * alpha)
    erode_mask = cv2.GaussianBlur(mask, (3, 3), 20) * original_mask
    return (erode_mask*255).astype(np.uint8)



