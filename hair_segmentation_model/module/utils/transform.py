import random
import cv2
import time
import math
import torch

import numpy as np
import torch.nn.functional as tf
import PIL.ImageEnhance as ImageEnhance

from typing import Optional, Any, Callable, Union


def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)


def count_FPS(prev_frame_time, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    return prev_frame_time


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im=im,
                    lb=lb,
                    )


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


def roi_tanh_polar_warp_cv(image: np.ndarray, roi: np.ndarray, target_width: int, target_height: int,
                           angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                           border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0,
                           keep_aspect_ratio: bool = False) -> np.ndarray:
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    roi_radii = (roi[2:4] - roi[:2]) / np.pi ** 0.5
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / target_width),
                                                   np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / target_height)),
                                       axis=-1)
    radii = normalised_dest_indices[..., 0]
    orientation_x = np.cos(normalised_dest_indices[..., 1])
    orientation_y = np.sin(normalised_dest_indices[..., 1])

    if keep_aspect_ratio:
        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(
            roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
    else:
        src_radii = np.arctanh(radii)
        src_x_indices = roi_radii[0] * src_radii * orientation_x
        src_y_indices = roi_radii[1] * src_radii * orientation_y
    src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                    roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)
    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_polar_restore_cv(warped_image: np.ndarray, roi: np.ndarray, image_width: int, image_height: int,
                              angular_offset: float = 0.0, interpolation: Optional[int] = cv2.INTER_LINEAR,
                              border_mode: Optional[int] = cv2.BORDER_CONSTANT, border_value: Any = 0,
                              keep_aspect_ratio: bool = False) -> np.ndarray:
    warped_height, warped_width = warped_image.shape[:2]
    roi_center = (roi[2:4] + roi[:2]) / 2.0
    roi_radii = (roi[2:4] - roi[:2]) / np.pi ** 0.5
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    dest_indices = np.stack(np.meshgrid(np.arange(image_width), np.arange(image_height)), axis=-1).astype(float)
    normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                             [sin_offset, cos_offset]]))
    if keep_aspect_ratio:
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                         roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]
    else:
        normalised_dest_indices /= roi_radii
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)

    src_radii = np.tanh(radii)
    warped_image = np.pad(np.pad(warped_image, [(1, 1), (0, 0)] + [(0, 0)] * (warped_image.ndim - 2), mode='wrap'),
                          [(0, 0), (1, 0)] + [(0, 0)] * (warped_image.ndim - 2), mode='edge')
    src_x_indices = src_radii * warped_width + 1.0
    src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                            2.0 / np.pi) * warped_height, warped_height) + 1.0

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def arctanh(x: torch.Tensor) -> torch.Tensor:
    return torch.log((1.0 + x) / (1.0 - x).clamp(1e-9)) / 2.0


def roi_tanh_polar_warp_torch(images: torch.Tensor, rois: torch.Tensor, target_width: int, target_height: int,
                        angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                        padding: str = 'zeros', keep_aspect_ratio: bool = False) -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(images.size()[:1] + (target_height, target_width, 2),
                        dtype=images.dtype, device=images.device)
    step = 1.0 / float(target_width)
    warped_radii = arctanh(torch.arange(0.0, 1.0, step, dtype=grids.dtype,
                                        device=grids.device)).unsqueeze(0).expand((target_height, target_width))
    thetas = torch.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / target_height, dtype=grids.dtype,
                          device=grids.device).unsqueeze(-1).expand((target_height, target_width))
    orientation_x = torch.cos(thetas)
    orientation_y = torch.sin(thetas)
    if not keep_aspect_ratio:
        orientation_x *= warped_radii
        orientation_y *= warped_radii

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    for roi_center, roi_radii, grid, cos_offset, sin_offset in zip(roi_centers, rois_radii, grids,
                                                                   cos_offsets, sin_offsets):
        if keep_aspect_ratio:
            src_radii = warped_radii * (roi_radii[0] * roi_radii[1] / torch.sqrt(
                roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
            warped_x_indices = src_radii * orientation_x
            warped_y_indices = src_radii * orientation_y
        else:
            warped_x_indices = roi_radii[0] * orientation_x
            warped_y_indices = roi_radii[1] * orientation_y
        src_x_indices, src_y_indices = (cos_offset * warped_x_indices - sin_offset * warped_y_indices,
                                        cos_offset * warped_y_indices + sin_offset * warped_x_indices)
        grid[..., 0] = (roi_center[0] + src_x_indices) / (image_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = (roi_center[1] + src_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_polar_restore_torch(warped_images: torch.Tensor, rois: torch.Tensor, image_width: int, image_height: int,
                           angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                           padding: str = 'zeros', keep_aspect_ratio: bool = False) -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(warped_images.size()[:1] + (image_height, image_width, 2),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_width, dtype=warped_images.dtype, device=warped_images.device)
    dest_y_indices = torch.arange(image_height, dtype=warped_images.dtype, device=warped_images.device)
    dest_indices = torch.cat((dest_x_indices.unsqueeze(0).expand((image_height, image_width)).unsqueeze(-1),
                              dest_y_indices.unsqueeze(-1).expand((image_height, image_width)).unsqueeze(-1)), -1)

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    warped_images = tf.pad(tf.pad(warped_images, [0, 0, 1, 1], mode='circular'), [1, 0, 0, 0], mode='replicate')
    for roi_center, roi_radii, grid, cos_offset, sin_offset in zip(roi_centers, rois_radii, grids,
                                                                   cos_offsets, sin_offsets):
        normalised_dest_indices = dest_indices - roi_center
        normalised_dest_indices[..., 0], normalised_dest_indices[..., 1] = (
            cos_offset * normalised_dest_indices[..., 0] + sin_offset * normalised_dest_indices[..., 1],
            cos_offset * normalised_dest_indices[..., 1] - sin_offset * normalised_dest_indices[..., 0])
        if keep_aspect_ratio:
            radii = normalised_dest_indices.norm(dim=-1)
            normalised_dest_indices[..., 0] /= radii.clamp(min=1e-9)
            normalised_dest_indices[..., 1] /= radii.clamp(min=1e-9)
            radii *= torch.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                                roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]
        else:
            normalised_dest_indices /= roi_radii
            radii = normalised_dest_indices.norm(dim=-1)
        thetas = torch.atan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0])
        grid[..., 0] = (torch.tanh(radii) * 2.0 * warped_width + 2) / warped_width - 1.0
        grid[..., 1] = ((thetas / math.pi).remainder(2.0) * warped_height + 2) / (warped_height + 1.0) - 1.0

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_polar_to_roi_tanh(warped_images: torch.Tensor, rois: torch.Tensor, target_width: int = 0,
                               target_height: int = 0, interpolation: str = 'bilinear', padding: str = 'zeros',
                               keep_aspect_ratio: bool = False) -> torch.Tensor:
    target_width = warped_images.size()[-1] if target_width <= 0 else target_width
    target_height = warped_images.size()[-2] if target_height <= 0 else target_height
    warped_height, warped_width = warped_images.size()[-2:]
    half_roi_sizes = (rois[:, 2:4] - rois[:, :2]) / 2.0
    rois_radii = half_roi_sizes * 2.0 / math.pi ** 0.5

    grids = torch.zeros(warped_images.size()[:1] + (target_height, target_width, 2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    warped_x_indices = arctanh(torch.arange(-1.0, 1.0, 2.0 / target_width, dtype=grids.dtype,
                                            device=grids.device) + 1.0 / target_width)
    warped_y_indices = arctanh(torch.arange(-1.0, 1.0, 2.0 / target_height, dtype=grids.dtype,
                                            device=grids.device) + 1.0 / target_height)
    dest_indices = torch.cat((warped_x_indices.unsqueeze(0).expand((target_height, target_width)).unsqueeze(-1),
                              warped_y_indices.unsqueeze(-1).expand((target_height, target_width)).unsqueeze(-1)), -1)

    warped_images = tf.pad(tf.pad(warped_images, [0, 0, 1, 1], mode='circular'), [1, 0, 0, 0], mode='replicate')
    for half_roi_size, roi_radii, grid in zip(half_roi_sizes, rois_radii, grids):
        normalised_dest_indices = dest_indices.clone()
        normalised_dest_indices[..., 0] *= half_roi_size[0]
        normalised_dest_indices[..., 1] *= half_roi_size[1]
        if keep_aspect_ratio:
            radii = normalised_dest_indices.norm(dim=-1)
            normalised_dest_indices[..., 0] /= radii.clamp(min=1e-9)
            normalised_dest_indices[..., 1] /= radii.clamp(min=1e-9)
            radii *= torch.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                                roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]
        else:
            normalised_dest_indices /= roi_radii
            radii = normalised_dest_indices.norm(dim=-1)
        thetas = torch.atan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0])
        grid[..., 0] = (torch.tanh(radii) * 2.0 * warped_width + 2) / warped_width - 1.0
        grid[..., 1] = ((thetas / math.pi).remainder(2.0) * warped_height + 2) / (warped_height + 1.0) - 1.0

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    while (1):
        _, frame = cap.read()
        prev_frame_time = count_FPS(prev_frame_time, frame)
        show_image('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
