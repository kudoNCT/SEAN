import sys
from model.BiSeNet import BiSeNet_inference
from utils.transform import roi_tanh_polar_warp_torch, roi_tanh_polar_restore_torch
from utils.dlib_test import Face_detection, make_erode_trimap

import torch
import cv2.cv2 as cv2
import glob
import torchvision.transforms as transforms
import numpy as np
from torch.nn.functional import softmax
import os


class SegmentationNet():
    def __init__(self, device,im_size=513, model_face_detection='500m'):
        self.device = device
        weights_folder = glob.glob('hair_segmentation_model/module/save_weights/skin_hair/*.pt')[0]

        checkpoint = torch.load(weights_folder,map_location=self.device)

        self.segment_net = BiSeNet_inference(n_classes=3)
        self.segment_net.to(self.device)
        net_dict = self.segment_net.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()}
        pretrained_dict_keys = pretrained_dict.keys()
        for k in list(pretrained_dict_keys):
            if k.split('.')[0] == 'module':
                pretrained_dict[k.replace('module.','')] = pretrained_dict[k]
                del pretrained_dict[k]
        pretrained_dict_keys = pretrained_dict.keys()
        for k in list(pretrained_dict_keys):
            if k.split('.')[0] == 'conv_out32' or k.split('.')[0] == 'conv_out16':
                del pretrained_dict[k]
            elif k.split('.')[1] == 'conv_out32' or k.split('.')[1] == 'conv_out16':
                del pretrained_dict[k]

        net_dict.update(pretrained_dict)
        self.segment_net.load_state_dict(net_dict)
        print('loaded pretrained Inference!!!!!')

        self.segment_net.eval()
        print('- Loaded weights.')

        self.tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.face_detector = Face_detection(model_face_detection)
        self.im_size = im_size

    def run(self, frame: np.array):
        print(f'frame.shape: {frame.shape}')
        faces = self.face_detector.run(frame, 0.2)
        print(f'len faces: {len(faces)}')
        mask = torch.zeros((frame.shape[0], frame.shape[1]), dtype=torch.bool).to(self.device)

        for face in faces:
            x1 = face[0][0]
            y1 = face[0][1]
            x2 = face[1][0]
            y2 = face[1][1]

            input_frame = torch.unsqueeze(self.tranform(frame.copy()), 0).to(self.device)
            tensor_bbox = torch.unsqueeze(torch.Tensor([x1, y1, x2, y2]), 0).to(self.device)

            rt_polar_im = roi_tanh_polar_warp_torch(input_frame,
                                                    tensor_bbox,
                                                    target_width=self.im_size,
                                                    target_height=self.im_size, keep_aspect_ratio=True)
            out = self.segment_net(rt_polar_im)

            out = softmax(out, 1)
            out[:, 0] = 1 - out[:, 0]

            out = roi_tanh_polar_restore_torch(out, tensor_bbox, frame.shape[1], frame.shape[0],
                                               keep_aspect_ratio=True)
            out[:, 0] = 1 - out[:, 0]
            out = out.argmax(1)

            out = torch.where(out == 2, 1, 0)
            mask = torch.bitwise_or(mask, out.bool())
        print(f'mask.shape {mask.shape}')
        return (np.squeeze(mask.cpu().numpy(), 0)*1).astype(np.uint8)

def check_model():
    net = SegmentationNet()
    frame = cv2.imread('source.jpg')
    predict = net.run(frame)
    predict = make_erode_trimap(predict)
    print(f'predict.shape : {predict.shape}')
    cv2.imshow('show', predict)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




