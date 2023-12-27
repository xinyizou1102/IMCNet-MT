import os
import cv2
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from glob import glob
from dataset.get_loader import get_loader
from config import param as option
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import set_seed, visualize_all
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from optim.get_optim import get_optim
from torch.utils.tensorboard import SummaryWriter
from trainer.basic_train import train_one_epoch


if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    model = get_model(option)
    optimizer, scheduler = get_optim(option, model.parameters())
    train_loader = get_loader(option)
    writer = SummaryWriter(option['log_path'])
    # code_backup_path = os.path.join(option['log_path'], 'CodeBackup')
    # if not os.path.exists(code_backup_path):
        # os.makedirs(code_backup_path)

    for epoch in range(1, (option['epoch']+1)):
        model, loss_record, vis_feat_dict = train_one_epoch(epoch, model, optimizer, train_loader, loss_fun)
        writer.add_scalar('loss', loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()
        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(model.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, loss_record.show()) + '.pth')
