import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import inspect


    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]

        return torch.mean(torch.stack(c))


def set_seed(seed=1024):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def visualize_all(in_pred, in_gt, path):
    for kk in range(in_pred.shape[0]):
        pred, gt = in_pred[kk, :, :, :], in_gt[kk, :, :, :]
        pred = (pred.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        gt = (gt.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        cat_img = cv2.hconcat([pred, gt])
        save_path = path + '/vis_temp/'   
        # Save vis images this temp folder, based on this experiment's folder.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = '{:02d}_cat.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)


def visualize_list(input_list, path):
    for kk in range(input_list[0].shape[0]):
        show_list = []
        for i in input_list:
            tmp = i[kk, :, :, :]
            tmp = (tmp.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
            show_list.append(tmp)
        cat_img = cv2.hconcat(show_list)
        save_path = path + '/vis_temp/'   
        # Save vis images this temp folder, based on this experiment's folder.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = '{:02d}_cat.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)


def visualize_flow(in_pred, path):
    # import pdb; pdb.set_trace()
    for kk in range(in_pred.shape[0]):
        pred = in_pred[kk, :, :, :]
        pred = (pred.permute(1, 2, 0).detach().cpu().numpy().squeeze()).astype(np.uint8)
        save_path = path + '/vis_temp/'   
        # Save vis images this temp folder, based on this experiment's folder.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = '{:02d}_flow.png'.format(kk)
        cv2.imwrite(save_path + name, pred)


def visualize_img(img, path, name):
    save_path = path + '/vis_temp/'   
    # Save vis images this temp folder, based on this experiment's folder.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name = '{}.png'.format(name)
    if img is not None and (save_path+name) is not None:
        cv2.imwrite(save_path + name, img)


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid
