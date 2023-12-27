import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    def __init__(self, target_size, do_flip=True, mode='Train'):
        
        # spatial augmentation params
        self.target_size = target_size
        self.mode = mode
        
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.3
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def normalize_transform(self, img1, img2):
        img1, img2 = img1 / 255, img2 / 255
        img1 = (img1 - self.mean) / self.std
        img2 = (img2 - self.mean) / self.std

        return img1, img2

    # def to_torch_transform(self, img1, img2, flow, mask):
    def spatial_transform(self, img1, img2, flow, mask, input_size):
        # randomly sample scale
        ht, wd = input_size
        ht_tar, wd_tar = self.target_size
        scale_y = ht_tar / ht
        scale_x = wd_tar / wd

        img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
        flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                mask = mask[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                mask = mask[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        return img1, img2, flow, mask

    def __call__(self, img1, img2, flow, mask, input_size):
        img1, img2, flow, mask = self.spatial_transform(img1, img2, flow, mask, input_size)
        if self.mode == 'Train':
            img1, img2 = self.color_transform(img1, img2)
            # img1, img2 = self.eraser_transform(img1, img2)  # Performance is pool, Don't use this for now 
        img1, img2 = self.normalize_transform(img1, img2)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        mask = np.ascontiguousarray(mask)

        return img1, img2, flow, mask
