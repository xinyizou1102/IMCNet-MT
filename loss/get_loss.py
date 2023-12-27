from loss.structure_loss import structure_loss
from loss.ssim import SSIM
import torch.nn as nn
import torch

def _iou(pred, target, size_average = True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        IoU = IoU + (1-IoU1)

    return IoU/b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


nn_bce = nn.BCELoss().cuda()
nn_iou = IOU()
nn_ssim = SSIM(window_size=11, size_average=True)
def bce_ssim_loss_with_sigmoid(pred, target, weight):
    pred = torch.sigmoid(pred)
    bce_out = nn_bce(pred, target)
    ssim_out = 1 - nn_ssim(pred,target)
    iou_out = nn_iou(pred,target)

    loss = bce_out + ssim_out + iou_out

    return loss


def bce_loss_with_sigmoid(pred, gt, weight):
    loss_fun = nn_bce
    
    return loss_fun(torch.sigmoid(pred), gt)


def get_loss(option):
    if option['loss'] == 'structure':
        loss_fun = structure_loss
    elif option['loss'] == 'bce':
        loss_fun = bce_loss_with_sigmoid
    elif option['loss'] == 'bas_loss':
        loss_fun = bce_ssim_loss_with_sigmoid

    return loss_fun


def cal_loss(pred, gt, loss_fun, weight=None):
    if isinstance(pred, list):
        loss = 0
        for i in pred:
            loss_curr = loss_fun(i, gt, weight)
            loss += loss_curr
        loss = loss / len(pred)
    else:
        loss = loss_fun(pred, gt, weight)

    return loss
