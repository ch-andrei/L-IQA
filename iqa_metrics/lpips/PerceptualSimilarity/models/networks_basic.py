from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from . import pretrained_networks as pn

import iqa_metrics.lpips.PerceptualSimilarity.models as util
# import self_attention.models as models_att_cnn
#
# from self_attention.models.bert import BertConfig, Learned2DRelativeSelfAttention, BertEncoder, BertLayer, BertAttention


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):  # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1. * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


# Learned perceptual metric
class PNetA2(nn.Module):
    def __init__(self, pnet_rand=False, pnet_tune=True):
        super(PNetA2, self).__init__()

        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand

        self.scaling_layer = ScalingLayer()

        # alex-net configuration
        self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)

        self.net_features = pn.sliced_alexnet(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        self.net_diff = pn.sliced_alexnet(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)

        # run with offset=0, to get responses from all 5 slices
        outs1 = self.net_features.forward(in0_input, offset=0)
        outs2 = self.net_features.forward(in1_input, offset=0)

        for i, (feats1, feats2) in enumerate(zip(outs1, outs2)):
            feats1 = util.normalize_tensor(feats1)
            feats2 = util.normalize_tensor(feats2)
            diff = (feats1 - feats2) ** 2  # L2 distance

            pdiff = self.net_diff.forward(diff, offset=i+1)
            pdiff = pdiff[-1]  # get the last layer
            pdiff_sum = pdiff if i == 0 else pdiff_sum + pdiff

        pdiff_sum = util.normalize_tensor(pdiff_sum)

        pdiff_sum = self.avgpool(pdiff_sum)
        pdiff_sum = torch.flatten(pdiff_sum, 1)

        return self.classifier(pdiff_sum).view(-1, 1, 1, 1)


# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='alex', pnet_rand=False, pnet_tune=False, use_dropout=True, version='0.1',
                 attention_type=None, attention_config=None):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'resnet50':
            net_type = pn.resnet
            self.chns = [64, 256, 512, 1024, 2048]
        else:
            raise TypeError("Unsupported pnet_type [{}]".format(pnet_type))

        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.attention_type = attention_type

        if attention_type is None or not attention_type:
            # original implementation
            print("LPIPS using NetLinLayer.")
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.diff_net = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        else:
            if attention_type == "SAGAN":
                print("LPIPS using SAGAN Self-Attention modules.")
                self.at0 = NetLinSelfAttentionSagan(self.chns[0], use_dropout=use_dropout)
                self.at1 = NetLinSelfAttentionSagan(self.chns[1], use_dropout=use_dropout)
                self.at2 = NetLinSelfAttentionSagan(self.chns[2], use_dropout=use_dropout)
                self.at3 = NetLinSelfAttentionSagan(self.chns[3], use_dropout=use_dropout)
                self.at4 = NetLinSelfAttentionSagan(self.chns[4], use_dropout=use_dropout)
            # elif attention_type == "BERT-small":
            #     print("LPIPS using BERT-small Self-Attention modules.")
            #     assert attention_config is not None
            #     self.at0 = NetLinSelfAttentionBertSmall(attention_config, self.chns[0], use_dropout=use_dropout)
            #     self.at1 = NetLinSelfAttentionBertSmall(attention_config, self.chns[1], use_dropout=use_dropout)
            #     self.at2 = NetLinSelfAttentionBertSmall(attention_config, self.chns[2], use_dropout=use_dropout)
            #     self.at3 = NetLinSelfAttentionBertSmall(attention_config, self.chns[3], use_dropout=use_dropout)
            #     self.at4 = NetLinSelfAttentionBertSmall(attention_config, self.chns[4], use_dropout=use_dropout)
            # elif attention_type == "BERT":
            #     print("LPIPS using BERT Self-Attention modules.")
            #     assert attention_config is not None
            #     self.at0 = NetLinSelfAttentionBert(attention_config, self.chns[0])
            #     self.at1 = NetLinSelfAttentionBert(attention_config, self.chns[1])
            #     self.at2 = NetLinSelfAttentionBert(attention_config, self.chns[2])
            #     self.at3 = NetLinSelfAttentionBert(attention_config, self.chns[3])
            #     self.at4 = NetLinSelfAttentionBert(attention_config, self.chns[4])
            else:
                raise TypeError("Unsupported attention_type [{}]".format(attention_type))
            self.diff_net = [self.at0, self.at1, self.at2, self.at3, self.at4]

    def forward(self, in0, in1, retPerLayer=False):
        # print('input.shape', in0.shape, in1.shape)

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (in0, in1) if self.version == '0.0' else (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)

        feats0, feats1, diffs = {}, {}, {}

        # print("diffs[kk].shape")
        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            # print('diffs[{}].shape'.format(kk), diffs[kk].shape)

        res = []
        for kk in range(self.L):
            pnet_diffs = self.diff_net[kk](diffs[kk])
            averaged = spatial_average(pnet_diffs, keepdim=True)
            res.append(averaged)

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if retPerLayer:
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SelfAttentionSagan(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim=None, k=1):
        super(SelfAttentionSagan, self).__init__()

        if out_dim is None:
            out_dim = in_dim

        self.out_dim = out_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # print('x.size', m_batchsize, C, width, height)
        x = x.float()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.out_dim, width, height)

        # out = self.gamma * out + x
        return out


class NetLinSelfAttentionSagan(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, use_dropout=False):
        # print("NetLinLayer chn_in", chn_in)
        super(NetLinSelfAttentionSagan, self).__init__()

        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [SelfAttentionSagan(chn_in, 1), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class NetLinSelfAttentionBertSmall(nn.Module):
#     def __init__(self, attention_config, chn_dim, use_dropout=True):
#         super(NetLinSelfAttentionBertSmall, self).__init__()
#
#         bert_config = BertConfig.from_dict(attention_config)
#
#         # scale input size to to Bert dimension
#         self.features_scale = nn.Linear(chn_dim, bert_config.hidden_size)
#
#         self.register_buffer("attention_mask", torch.tensor(1.0))
#         self.model = Learned2DRelativeSelfAttention(bert_config, output_attentions=False)
#
#         layers = [nn.Dropout(), ] if use_dropout else []
#         layers += [nn.Conv2d(bert_config.hidden_size, 1, 1, stride=1, padding=0, bias=False), ]
#         self.classifier = nn.Sequential(*layers)
#
#         print("self.classifier.requires_grad", self.classifier.requires_grad)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)  # transpose for bert
#         x = self.features_scale(x)
#         x = self.model(x, self.attention_mask)
#         x = x.permute(0, 3, 1, 2)  # transpose after bert
#         return self.classifier(x)
#
#
# # from self_attention.models.bert import BertAttention, BertConfig
# class NetLinSelfAttentionBert(nn.Module):
#     def __init__(self, attention_config, chn_dim):
#         super(NetLinSelfAttentionBert, self).__init__()
#
#         bert_config = BertConfig.from_dict(attention_config)
#
#         # scale input size to to Bert dimension
#         self.features_scale = nn.Linear(chn_dim, bert_config.hidden_size)
#
#         self.register_buffer("attention_mask", torch.tensor(1.0))
#         self.attention = BertAttention(bert_config, output_attentions=False)
#
#         intermediate_size = bert_config.hidden_size
#
#         self.intermediate = nn.Sequential(
#             nn.Dropout(),
#             nn.Conv2d(bert_config.hidden_size, intermediate_size, 1, stride=1, padding=0, bias=False),
#             nn.Conv2d(intermediate_size, 1, 1, stride=1, padding=0, bias=False),
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         x = self.features_scale(x)
#         x = self.attention(x, self.attention_mask)
#         x = x.permute(0, 3, 2, 1)
#         x = self.intermediate(x)
#         return x
#

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if use_sigmoid:
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if (self.colorspace == 'RGB'):
            (N, C, X, Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
                               dim=3).view(N)
            return value
        elif (self.colorspace == 'Lab'):
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)),
                            util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if (self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var


class DSSIM(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if (self.colorspace == 'RGB'):
            value = util.dssim(1. * util.tensor2im(in0.data), 1. * util.tensor2im(in1.data), range=255.).astype('float')
        elif (self.colorspace == 'Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)),
                               util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype(
                'float')
        ret_var = Variable(torch.Tensor((value,)))
        if (self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network', net)
    print('Total number of parameters: %d' % num_params)
