import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import torch.utils.data as data
from tqdm import tqdm
from config import param as option

from model.get_model import get_model
from dataset.davsod import DAVSODAllPairsCV2Test, DAVSODAllPairsCV2TestFBMS
from torchvision import transforms 


def eval_mae(loader, cuda=True):
    avg_mae, img_num, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in loader:
            if cuda:
                pred, gt = trans(pred).cuda(), trans(gt).cuda()
            else:
                pred, gt = trans(pred), trans(gt)
            mae = torch.abs(pred - gt).mean()
            if mae == mae: # for Nan
                avg_mae += mae
                img_num += 1.0
        avg_mae /= img_num
    return avg_mae


def process_DAVIS(generator, eval_dataset):
    # Begin the testing process
    pre_root = option['eval_save_path']

    test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
    save_path_base = os.path.join(pre_root, test_epoch_num+'_epoch/'+eval_dataset+'_binary')

    # Begin to inference and save masks
    print('========== Begin to inference and save masks ==========')
    dataset = DAVSODAllPairsCV2Test(option['root'], sub_set='TestSet/'+eval_dataset)
    test_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    progress_bar = tqdm(test_loader, desc='Testing')
    for i, pack in enumerate(progress_bar):
        image1, image2, gts, flow, (WW, HH), label_name = pack['image1'].cuda(), pack['image2'].cuda(), pack['label'].cuda(), pack['flow'].cuda(), pack['size'], pack['label_name']
        saliency_list, _ = generator.forward(image1, image2, flow)
        res = saliency_list[-1]
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res[res >= 0.5] = 255
        res[res < 0.5] = 0
        save_path = os.path.join(save_path_base, label_name[0].split('/')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, label_name[0].split('/')[-1])
        cv2.imwrite(save_path, res)


def process_FBMS(generator, eval_dataset):
    # Begin the testing process
    pre_root = option['eval_save_path']

    test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
    save_path_base = os.path.join(pre_root, test_epoch_num+'_epoch/'+eval_dataset)

    # Begin to inference and save masks
    print('========== Begin to inference and save masks ==========')
    dataset = DAVSODAllPairsCV2TestFBMS(option['root'], sub_set='TestSet/')
    test_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    progress_bar = tqdm(test_loader, desc='Testing')
    for i, pack in enumerate(progress_bar):
        image1, image2, gts, flow, (WW, HH), label_name = pack['image1'].cuda(), pack['image2'].cuda(), pack['label'].cuda(), pack['flow'].cuda(), pack['size'], pack['label_name']
        saliency_list, _ = generator.forward(image1, image2, flow)
        res = saliency_list[-1]
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path = os.path.join(save_path_base, label_name[0].split('/')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, label_name[0].split('/')[-1])
        cv2.imwrite(save_path, res)


def process_single_dataset(generator, eval_dataset):
    # Begin the testing process
    pre_root = option['eval_save_path']

    test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
    save_path_base = os.path.join(pre_root, test_epoch_num+'_epoch/'+eval_dataset)

    # Begin to inference and save masks
    print('========== Begin to inference and save masks ==========')
    dataset = DAVSODAllPairsCV2Test(option['root'], sub_set='TestSet/'+eval_dataset)
    test_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    import time
    import numpy as np
    time_list = []
    progress_bar = tqdm(test_loader, desc='Testing')
    for i, pack in enumerate(progress_bar):
        image1, image2, gts, flow, (WW, HH), label_name = pack['image1'].cuda(), pack['image2'].cuda(), pack['label'].cuda(), pack['flow'].cuda(), pack['size'], pack['label_name']
        (WW, HH) = gts.shape[2:]
        torch.cuda.synchronize()
        start = time.time()
        saliency_list, vis_feat = generator.forward(image1, image2, flow)
        cat_img = cv2.hconcat([cv2.resize(vis_feat['atten'], (HH, WW)), cv2.resize(vis_feat['semantic_feat'], (HH, WW)), cv2.resize(vis_feat['hypercorr_encoded_v2'], (HH, WW))])
        # cat_img = cv2.resize(vis_feat['semantic_feat'], (HH, WW))  # ablation 1
        # cat_img = cv2.resize(vis_feat['hypercorr_encoded_v2'], (HH, WW))  # ablation 2
        # cat_img = cv2.hconcat([cv2.resize(vis_feat['semantic_feat'], (HH, WW)), cv2.resize(vis_feat['hypercorr_encoded_v2'], (HH, WW))])  # ablation 3
        torch.cuda.synchronize()
        end = time.time()
        res = saliency_list[-1]
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        save_path = os.path.join(save_path_base, label_name[0].split('\\')[0])
        save_path_vis = os.path.join(save_path_base, label_name[0].split('\\')[0]).replace(save_path_base, save_path_base + '_vis')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_vis):
            os.makedirs(save_path_vis)
        save_path = os.path.join(save_path, label_name[0].split('\\')[-1])
        save_path_vis = os.path.join(save_path_vis, label_name[0].split('\\')[-1])
        cv2.imwrite(save_path, res)
        cv2.imwrite(save_path_vis, cat_img)
        time_list.append(end-start)
    print("process time:", np.mean(time_list))


if __name__ == "__main__":
    generator = get_model(option)
    generator.load_state_dict(torch.load(option['checkpoint']))
    generator.eval()
    eval_dataset_list = ['DAVSOD', 'DAVSOD-Difficult-20', 'DAVSOD-Normal-25']  # , 'SegTrack-V2', 'DAVIS']
    for eval_dataset in eval_dataset_list:
        print('[INFO]: Process [{}] dataset'.format(eval_dataset))
        if eval_dataset == 'DAVIS':
            # process_DAVIS(generator, eval_dataset)
            process_single_dataset(generator, eval_dataset)
        elif eval_dataset == 'FBMS':
            process_FBMS(generator, eval_dataset)
        else:
            process_single_dataset(generator, eval_dataset)
