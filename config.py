import os
import time
import argparse


parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--backbone', type=str, default='swin', choices=['R101', 'R50', 'swin', 'simple'])
parser.add_argument('--log_info', type=str, default='xr1_mod')
parser.add_argument('--ckpt', type=str, default='F:/MTexperiments/VideoSOD1129/VideoSOD/experiments/swin_xr1_mod_2.5e-05/models/30_0.2619.pth')
args = parser.parse_args()

# Configs
param = {}

# Training Config
param['epoch'] = 30
param['seed'] = 1234
param['batch_size'] = 4
param['save_epoch'] = 5       # How many rounds should the model be saved
param['lr'] = 2.5e-5
param['trainsize'] = 384
param['optim'] = "Adam"
param['decay_rate'] = 0.5
param['decay_epoch'] = 12
param['beta'] = [0.99, 0.999]  # Adam parameters
param['size_rates'] = [1]     # multi-scale training  [0.75, 1, 1.25]/[1]
param['pretrain'] = "model/swin/swin_base_patch4_window12_384.pth"
param['backbone'] = args.backbone
param['loss'] = 'structure'


# Dataset Config
training_path = 'F:\DAVSOD'
param['image_root'] = training_path + '/img/'
param['gt_root'] = training_path + '/gt/'
param['root'] = training_path



# Experiment Dir Config
log_info = args.backbone + '_' + args.log_info    # the name of the experiment
param['training_info'] = log_info + '_' + str(param['lr'])
param['log_path'] = 'experiments/{}'.format(param['training_info'])
param['ckpt_save_path'] = param['log_path'] + '/models/'
print('[INFO]: Experiments saved in: ', param['training_info'])


# Test Config
param['testsize'] = 384
param['checkpoint'] = args.ckpt
param['eval_save_path'] = param['log_path'] + '/save_images/'
