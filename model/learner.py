import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.base.conv4d import CenterPivotConv4d as Conv4d


def vis_feat(input_feat, name=None):
    if input_feat.shape[0] > 1:
        feat_list = []
        for i in range(input_feat.shape[0]):
            feat_np = input_feat[i, :, :, :].mean(0).detach().cpu().numpy()
            feat_norm = (feat_np-feat_np.min())/(feat_np.max()-feat_np.min()) * 255
            # im_color =  cv2.applyColorMap(np.array(cv2.resize(feat_norm, None, fx=4, fy=4), np.uint8), cv2.COLORMAP_WINTER)
            im_color = cv2.resize(feat_norm, None, fx=4, fy=4)
            feat_list.append(im_color)
        cat_img = cv2.hconcat(feat_list)
        return cat_img
    else:
        feat_np = input_feat[0].mean(0).detach().cpu().numpy()
        feat_norm = (feat_np-feat_np.min())/(feat_np.max()-feat_np.min()) * 255
        
        return cv2.applyColorMap(np.uint8(cv2.resize(feat_norm, None, fx=4, fy=4)), cv2.COLORMAP_JET)

class SelfAttnFuse(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttnFuse, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim//2, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim//2, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.InstanceNorm = nn.InstanceNorm2d(128, affine=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, appearance_1, appearance_2, motion):
        # Using motion feature to attention appearance feature.
        bs, channel, width, height = motion.shape
        W_z = self.query_conv(appearance_2).view(bs, -1, width*height).permute(0, 2, 1)
        W_u_key = self.key_conv(appearance_1).view(bs, -1, width*height)
        W_u_value = self.value_conv(motion).view(bs, -1, width*height)
        energy = torch.bmm(W_z, W_u_key)
        attention = self.softmax(energy)

        out = torch.bmm(W_u_value, attention.permute(0, 2, 1))
        out = out.view(bs, channel, width, height)
        # out = self.InstanceNorm(out)
        # motion = self.InstanceNorm(motion)

        # fuse_out = self.gamma*out + motion
        # import ipdb; ipdb.set_trace()
        fuse_out = out * motion

        return fuse_out, out


class SelfAttnFuseSame(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttnFuseSame, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.InstanceNorm = nn.InstanceNorm2d(128, affine=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, appearance_1, appearance_2, motion):
        # Using motion feature to attention appearance feature.
        bs, channel, width, height = motion.shape
        W_z = self.query_conv(appearance_2).view(bs, -1, width*height).permute(0, 2, 1)
        W_u_key = self.key_conv(appearance_1).view(bs, -1, width*height)
        W_u_value = self.value_conv(motion).view(bs, -1, width*height)
        energy = torch.bmm(W_z, W_u_key)
        attention = self.softmax(energy)

        out = torch.bmm(W_u_value, attention.permute(0, 2, 1))
        out = out.view(bs, channel, width, height)
        # out = self.InstanceNorm(out)
        # motion = self.InstanceNorm(motion)

        # fuse_out = self.gamma*out + motion
        # import ipdb; ipdb.set_trace()
        fuse_out = out * motion

        return fuse_out, out


class MotionLearner(nn.Module):
    def __init__(self, inch):
        super(MotionLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4
                # building_block_layers.append(SepConv4d(inch, outch))
                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        self.encoder_corr_layer4 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_corr_layer3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_corr_layer2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        self.attn_2 = SelfAttnFuse(in_dim=128)
        self.attn_3 = SelfAttnFuse(in_dim=128)
        self.attn_4 = SelfAttnFuse(in_dim=128)
        self.attn_final = SelfAttnFuseSame(in_dim=64)
        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))
                            
        self.decoder3 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder4 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))

        self.decoder5 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder6 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))

        self.decoder7_sem = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder8_sem = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))

        self.decoder9_sem = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder10_sem = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))

        self.decoder11_sem = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder12_sem = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))
        
        self.decoder13_sem = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder14_sem = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
                                      nn.Conv2d(outch2, 1, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid, img_feats_pyramid_1):
        # Cost Volume filtering
        vis_feat_dict = {}
        corr_4 = self.encoder_layer4(hypercorr_pyramid[0])  # [4, 1, 12, 12, 12, 12] -> [4, 128, 12, 12, 2, 2]
        corr_3 = self.encoder_layer3(hypercorr_pyramid[1])
        corr_2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Decoder 1st stage
        feature_sem_4 = self.decoder7_sem(img_feats_pyramid_1[3])
        feature_sem_4_up = F.interpolate(feature_sem_4, (feature_sem_4.size(-1) * 2,) * 2, mode='bilinear', align_corners=True)
        logit_mask_sem_4 = self.decoder8_sem(feature_sem_4)

        corr_4_decode = self.encoder_corr_layer4(corr_4)
        bsz, ch, ha, wa, hb, wb = corr_4_decode.size()
        corr_4_decode_reshape = corr_4_decode.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        atten_fuse_4, _ = self.attn_4(feature_sem_4, feature_sem_4, corr_4_decode_reshape)
        hypercorr_decoded = self.decoder1(atten_fuse_4)
        logit_mask_4 = self.decoder2(hypercorr_decoded)

        corr_4_decode_up = self.interpolate_support_dims(corr_4_decode, corr_3.size()[-4:-2])
        # [4, 128, 12, 12, 2, 2] -> [4, 128, 24, 24, 2, 2]
        # Decoder 2nd stage
        feature_sem_3 = self.decoder9_sem(img_feats_pyramid_1[2])+feature_sem_4_up
        feature_sem_3_up = F.interpolate(feature_sem_3, (feature_sem_3.size(-1) * 2,) * 2, mode='bilinear', align_corners=True)
        logit_mask_sem_3 = self.decoder10_sem(feature_sem_3)
  
        corr_3_decode = self.encoder_corr_layer3(corr_3+corr_4_decode_up)
        bsz, ch, ha, wa, hb, wb = corr_3_decode.size()
        corr_3_decode_reshape = corr_3_decode.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        atten_fuse_3, _ = self.attn_3(feature_sem_3, feature_sem_3, corr_3_decode_reshape)
        hypercorr_decoded = self.decoder3(atten_fuse_3)
        logit_mask_3 = self.decoder4(hypercorr_decoded)

        corr_3_decode_up = self.interpolate_support_dims(corr_3_decode, corr_2.size()[-4:-2])

        # Decoder 3rd stage
        feature_sem_2 = self.decoder11_sem(img_feats_pyramid_1[1])+feature_sem_3_up
        feature_sem_2_up = F.interpolate(feature_sem_2, (feature_sem_2.size(-1) * 2,) * 2, mode='bilinear', align_corners=True)
        logit_mask_sem_2 = self.decoder12_sem(feature_sem_2)

        corr_2_decode = self.encoder_corr_layer2(corr_2+corr_3_decode_up)
        bsz, ch, ha, wa, hb, wb = corr_2_decode.size()
        corr_2_decode_reshape = corr_2_decode.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        atten_fuse_2, atten_map = self.attn_2(feature_sem_2, feature_sem_2, corr_2_decode_reshape)
        hypercorr_decoded = self.decoder5(atten_fuse_2)
        logit_mask_2 = self.decoder6(hypercorr_decoded)
        hypercorr_decoded = F.interpolate(hypercorr_decoded, (hypercorr_decoded.size(-1) * 2,) * 2, mode='bilinear',
                                          align_corners=True)

        # Decoder output
        feature_sem_1 = self.decoder13_sem(img_feats_pyramid_1[0])+feature_sem_2_up
        # logit_mask_sem_1 = self.decoder15_sem(feature_sem_1)

        fuse_feat_final, _ = self.attn_final(feature_sem_1, feature_sem_1, hypercorr_decoded)
        logit_mask_final = self.decoder14_sem(fuse_feat_final)


        vis_feat_dict.update({"atten": vis_feat(atten_map)})
        vis_feat_dict.update({"atten_fuse_v2": vis_feat(atten_fuse_2)})
        vis_feat_dict.update({"semantic_feat": vis_feat(feature_sem_2)})
        vis_feat_dict.update({"hypercorr_encoded_v2": vis_feat(corr_2_decode_reshape)})

        return [logit_mask_4, logit_mask_sem_4, logit_mask_3, logit_mask_sem_3, logit_mask_sem_2, logit_mask_2, logit_mask_final], vis_feat_dict
