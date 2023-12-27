import torch
from loss.lscloss import LocalSaliencyCoherence
import numpy as np
from tqdm import tqdm
from config import param as option
from torch.autograd import Variable
from utils import AvgMeter, visualize_all, visualize_list, visualize_img
from loss.get_loss import get_loss, cal_loss


def train_one_epoch(epoch, model_list, optimizer_list, train_loader, loss_fun):
    generator = model_list
    generator_optimizer = optimizer_list
    generator.train()
    loss_record, lsc_loss_record = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        if True:
            generator_optimizer.zero_grad()
            
            image1, image2, gts, flow = pack['image1'].cuda(), pack['image2'].cuda(), pack['label'].cuda(), pack['flow'].cuda()
            # Inference Once
            ref_pre, vis_feat_dict = generator(image1, image2, flow)

            # Optimize generator
            gt_loss = cal_loss(ref_pre, gts, loss_fun)
            supervised_loss = gt_loss  # +0.3*lsc_loss
            supervised_loss.backward()
            generator_optimizer.step()

            # import pdb; pdb.set_trace()
            # visualize_list([torch.sigmoid(ref_pre[0]), torch.sigmoid(ref_pre[1]), torch.sigmoid(ref_pre[2]), torch.sigmoid(ref_pre[3]), torch.sigmoid(ref_pre[4]), gts], option['log_path'])
            vis_list = [torch.sigmoid(i) for i in ref_pre]
            vis_list.append(gts)
            visualize_list(vis_list, option['log_path'])
            # visualize_img(cv2.vconcat([vis_feat_dict['hypercorr_encoded_v2'], vis_feat_dict['semantic_feat']]), option['log_path'], 'feature_vis')
            # visualize_img(cv2.vconcat([vis_feat_dict['atten'], vis_feat_dict['atten_fuse_v2'], vis_feat_dict['hypercorr_encoded_v2'], vis_feat_dict['semantic_feat']]), option['log_path'], 'feature_vis')

            loss_record.update(supervised_loss.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}')

    return generator, loss_record, vis_feat_dict
