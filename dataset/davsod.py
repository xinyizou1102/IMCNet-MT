import os
import cv2
import random
from PIL import Image
from glob import glob
import torch
from torch.utils import data
from dataset.augmentor import FlowAugmentor
import numpy as np
import re 



def random_list(li):
    for i in range(0, 100):
        index1 = random.randint(0, len(li) - 1)
        index2 = random.randint(0, len(li) - 1)
        li[index1], li[index2] = li[index2], li[index1]
    return li


class DAVISAllPairsCV2(data.Dataset):
    def __init__(self, root, trainsize, sub_set='TrainingSet', transform=None, return_size=True):

        self.img_dir_name, self.GT_dir_name, self.flow_dir_name, self.replace_format = 'JPEGImages', 'Annotations', 'flow', False
        self.root = root
        self.return_size = return_size
        self.dataset_image_pair_list = self._generate_image_pair_list(self.root)
        self.augmenter = FlowAugmentor(target_size=[trainsize, trainsize], do_flip=True, mode='Train')

        self.transform = transform

    def _generate_image_pair_list(self, root):
        test_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 
                     'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 
                     'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']
        sub_set_list = os.listdir(os.path.join(root, 'JPEGImages', '480p'))
        dataset_image_pair_list = []
        for sub_set_name in sub_set_list:
            if sub_set_name not in test_list:
                sub_set_images = os.listdir(os.path.join(root, 'JPEGImages', '480p', sub_set_name))
                sub_set_images.sort()
                for i in range(len(sub_set_images)-1):
                    img_1 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i])
                    img_2 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i+1])
                    image_pairs = [img_1, img_2]
                    dataset_image_pair_list.append(image_pairs)
                for i in range(0, len(sub_set_images)-3, 3):
                    img_1 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i])
                    img_2 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i+3])
                    image_pairs = [img_1, img_2]
                    dataset_image_pair_list.append(image_pairs)
                for i in range(0, len(sub_set_images)-5, 5):
                    img_1 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i])
                    img_2 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i+5])
                    image_pairs = [img_1, img_2]
                    dataset_image_pair_list.append(image_pairs)
                for i in range(0, len(sub_set_images)-7, 7):
                    img_1 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i])
                    img_2 = os.path.join(root, 'JPEGImages', '480p', sub_set_name, sub_set_images[i+7])
                    image_pairs = [img_1, img_2]
                    dataset_image_pair_list.append(image_pairs)

        return random_list(dataset_image_pair_list)

    def __getitem__(self, item):
        image_pairs = self.dataset_image_pair_list[item]
        img1_path, img2_path = image_pairs[0], image_pairs[1]
        
        label_path = img1_path.replace(self.img_dir_name, self.GT_dir_name).replace('jpg', 'png')
        if self.replace_format:
            label_path = label_path.replace('jpg', 'png')
        flow_path = img1_path.replace(self.img_dir_name, self.flow_dir_name)

        label, image1, image2 = cv2.imread(label_path, 0), cv2.imread(img1_path), cv2.imread(img2_path)
        # import pdb; pdb.set_trace()
        flow, _ = image1[:, :, 0:2], 0  # Didn't need flow now 
        if label.max() > 0:
            label = label / label.max()

        image1, image2, flow, label = self.augmenter(image1, image2, flow, label, image1.shape[:2])

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        label = torch.from_numpy(label).unsqueeze(0).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        sample = {'image1': image1, 'image2': image2, 'label': label, 'flow': flow}

        pos_list = [i.start() for i in re.finditer('/', label_path)]
        label_name = label_path[pos_list[-3]+1:]
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.dataset_image_pair_list)



class DAVSODAllPairsCV2(data.Dataset):
    def __init__(self, root, trainsize, sub_set='TrainingSet', transform=None, return_size=True):
        if sub_set not in ['TrainingSet', 'ValidationSet', 'TestSet/DAVSOD']:
            print('[ERROR]: Sub Set is wrong. You need to confirm your code.')
            import pdb; pdb.set_trace()
        if sub_set in ['TrainingSet', 'ValidationSet']:
            self.img_dir_name, self.GT_dir_name, self.flow_dir_name, self.replace_format = 'Imgs', 'GT_object_level', 'flow', False
        else:
            self.img_dir_name, self.GT_dir_name, self.flow_dir_name, self.replace_format = 'Frame', 'GT', 'flow_vis_raft', True
        self.root = root
        self.sub_set = sub_set
        self.return_size = return_size
        self.dataset_image_pair_list = self._generate_image_pair_list(self.root, self.sub_set, self.img_dir_name)
        self.augmenter = FlowAugmentor(target_size=[trainsize, trainsize], do_flip=True, mode='Train')

        self.transform = transform

    def _generate_image_pair_list(self, root, sub_set, img_dir_name):
        sub_set_list = os.listdir(os.path.join(root, sub_set))
        dataset_image_pair_list = []
        for sub_set_name in sub_set_list:
            sub_set_images = os.listdir(os.path.join(root, sub_set, sub_set_name, img_dir_name))
            sub_set_images.sort()
            for i in range(len(sub_set_images)-1):
                img_1 = os.path.join(root, sub_set, sub_set_name, img_dir_name, sub_set_images[i])
                img_2 = os.path.join(root, sub_set, sub_set_name, img_dir_name, sub_set_images[i+1])
                image_pairs = [img_1, img_2]
                dataset_image_pair_list.append(image_pairs)
        # import pdb; pdb.set_trace()
        return random_list(dataset_image_pair_list)

    def __getitem__(self, item):
        image_pairs = self.dataset_image_pair_list[item]
        img1_path, img2_path = image_pairs[0], image_pairs[1]
        
        label_path = img1_path.replace(self.img_dir_name, self.GT_dir_name)
        if self.replace_format:
            label_path = label_path.replace('jpg', 'png')
        flow_path = img1_path.replace(self.img_dir_name, self.flow_dir_name)

        label, image1, image2 = cv2.imread(label_path, 0), cv2.imread(img1_path), cv2.imread(img2_path)
        flow = image1[:, :, 0:2]
        if label.max() > 0:
            label = label / label.max()

        image1, image2, flow, label = self.augmenter(image1, image2, flow, label, image1.shape[:2])

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        label = torch.from_numpy(label).unsqueeze(0).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        sample = {'image1': image1, 'image2': image2, 'label': label, 'flow': flow}

        pos_list = [i.start() for i in re.finditer(r'\\', label_path)]
        label_name = label_path[pos_list[-3]+1:]
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.dataset_image_pair_list)


class DAVSODAllPairsCV2TestFBMS(data.Dataset):
    def __init__(self, root, trainsize, sub_set='TrainingSet', return_size=True):

        self.img_dir_name, self.GT_dir_name, self.flow_dir_name, self.replace_format = 'Imgs', 'GT_object_level', 'flow', True
        self.root = root
        self.sub_set = sub_set
        self.return_size = return_size
        self.dataset_image_list = self._generate_image_list_from_gt(self.root, self.sub_set, self.GT_dir_name)
        self.augmenter = FlowAugmentor(target_size=[trainsize, trainsize], do_flip=False, mode='Test')

    def _generate_image_list_from_gt(self, root, sub_set, GT_dir_name):
        sub_set_list = os.listdir(os.path.join(root, sub_set))
        dataset_image_pair_list = []
        for sub_set_name in sub_set_list:
            sub_set_images = os.listdir(os.path.join(root, sub_set, sub_set_name, GT_dir_name))
            sub_set_images.sort()
            for i in range(len(sub_set_images)):
                gt = os.path.join(root, sub_set, sub_set_name, GT_dir_name, sub_set_images[i])
                dataset_image_pair_list.append(gt)

        return dataset_image_pair_list

    def __getitem__(self, item):
        label_path = self.dataset_image_list[item]
        img1_path = label_path.replace(self.GT_dir_name, self.img_dir_name)
        img2_path = img1_path[:-9]+str((int(img1_path[-9:].split('.')[0])+1)).zfill(5)+'.png'

        flow_path = img1_path.replace(self.img_dir_name, self.flow_dir_name)

        print(flow_path)
        label = cv2.imread(label_path, 0)
        image1 = cv2.imread(img1_path)
        image2 = cv2.imread(img2_path)
        flow = image1[:, :, 0:2]
        
        h, w = label.shape
        size = (h, w)
        if label.max() > 0:
            label = label / label.max()

        image1, image2, flow, label = self.augmenter(image1, image2, flow, label, image1.shape[:2])

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        label = torch.from_numpy(label).unsqueeze(0).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        sample = {'image1': image1, 'image2': image2, 'label': label, 'flow': flow}

        pos_list = [i.start() for i in re.finditer('/', label_path)]
        label_name = label_path[pos_list[-3]+1:]
        sample['label_name'] = label_name
        sample['size'] = size

        return sample

    def __len__(self):
        return len(self.dataset_image_list)


class DAVSODAllPairsCV2Test(data.Dataset):
    def __init__(self, root, sub_set='TrainingSet', return_size=True):
        # if sub_set not in ['TestSet/DAVSOD', 'TestSet/DAVSOD-Normal-25', 'TestSet/DAVSOD-Difficult-20', 'TestSet/SegTrack-V2']:
        #     print('[ERROR]: Sub Set is wrong. You need to confirm your code.')
        #     import pdb; pdb.set_trace()

        self.img_dir_name, self.GT_dir_name, self.flow_dir_name, self.replace_format = 'Frame', 'GT', 'flow', True
        self.root = root
        self.sub_set = sub_set
        self.return_size = return_size
        self.dataset_image_pair_list = self._generate_image_pair_list(self.root, self.sub_set, self.img_dir_name)
        self.augmenter = FlowAugmentor(target_size=[384, 384], do_flip=False, mode='Test')

    def _generate_image_pair_list(self, root, sub_set, img_dir_name):
        sub_set_list = os.listdir(os.path.join(root, sub_set))
        dataset_image_pair_list = []
        for sub_set_name in sub_set_list:
            sub_set_images = os.listdir(os.path.join(root, sub_set, sub_set_name, img_dir_name))
            sub_set_images.sort()
            for i in range(len(sub_set_images)-1):
                img_1 = os.path.join(root, sub_set, sub_set_name, img_dir_name, sub_set_images[i])
                img_2 = os.path.join(root, sub_set, sub_set_name, img_dir_name, sub_set_images[i+1])
                image_pairs = [img_1, img_2]
                dataset_image_pair_list.append(image_pairs)

        return dataset_image_pair_list

    def __getitem__(self, item):
        image_pairs = self.dataset_image_pair_list[item]
        img1_path, img2_path = image_pairs[0], image_pairs[1]
        
        label_path = img1_path.replace(self.img_dir_name, self.GT_dir_name)
        if self.replace_format:
            label_path = label_path.replace('jpg', 'png')
        flow_path = img1_path.replace(self.img_dir_name, self.flow_dir_name).replace('jpg', 'png')

        label = cv2.imread(label_path, 0)
        image1 = cv2.imread(img1_path)
        image2 = cv2.imread(img2_path)
        flow = image1[:, :, 0:2]
        h, w = label.shape
        size = (h, w)
        if label.max() > 0:
            label = label / label.max()

        image1, image2, flow, label = self.augmenter(image1, image2, flow, label, image1.shape[:2])

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        label = torch.from_numpy(label).unsqueeze(0).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        sample = {'image1': image1, 'image2': image2, 'label': label, 'flow': flow}

        pos_list = [i.start() for i in re.finditer(r'\\', label_path)]
        label_name = label_path[pos_list[-3]+1:]
        sample['label_name'] = label_name
        sample['size'] = size

        return sample

    def __len__(self):
        return len(self.dataset_image_pair_list)



class EvalDAVSOD(data.Dataset):
    def __init__(self, pre_root, gt_root):

        self.img_dir_name, self.GT_dir_name, self.flow_dir_name, self.replace_format = 'Frame', 'GT', 'flow_vis_raft', True
        self.gt_list = self._generate_gt_list(gt_root, 'DAVSOD', 'GT')
        self.pre_list = self._generate_pre_list(pre_root)

    def _generate_gt_list(self, root, sub_set, img_dir_name):
        sub_set_list = os.listdir(os.path.join(root, sub_set))
        gt_list = []
        for sub_set_name in sub_set_list:
            sub_set_images = os.listdir(os.path.join(root, sub_set, sub_set_name, img_dir_name))
            sub_set_images.sort()
            for i in range(len(sub_set_images)-1):
                mask_path = os.path.join(root, sub_set, sub_set_name, img_dir_name, sub_set_images[i])
                gt_list.append(mask_path)

        return gt_list

    def _generate_pre_list(self, root):
        sub_set_list = os.listdir(root)
        pre_list = []
        for sub_set_name in sub_set_list:
            sub_set_images = os.listdir(os.path.join(root, sub_set_name))
            sub_set_images.sort()
            for i in range(len(sub_set_images)):
                pre_path = os.path.join(root, sub_set_name, sub_set_images[i])
                pre_list.append(pre_path)
        
        return pre_list

    def __getitem__(self, item):
        gt, pre = cv2.imread(self.gt_list[item], 0), cv2.imread(self.pre_list[item], 0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        pre = torch.from_numpy(pre).unsqueeze(0)

        sample = {'gt': gt, 'pre': pre}

        return sample

    def __len__(self):
        return len(self.gt_list)


if __name__ == '__main__':
    print("hello world")