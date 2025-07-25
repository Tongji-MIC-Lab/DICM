import time
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moco.conv2former import Encoder, Decoder, Encoder_IF, Decoder_IF, GlobalLocalFilter, gnconv
from moco.hyper_codec import TransHyperScale, TransDecoder, TransDecoder2
from criterion.distribution_mixture import DiscretizedMixGaussLoss
from utils import vit2_init, xavier_uniform_init
from moco.prob_model import Entropy
import torch.utils.checkpoint as checkpoint
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResBlock2D(nn.Module):
    def __init__(self, num_channels, kernel_size, bn=False):
        super(ResBlock2D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*2
        if bn:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                          stride=1, padding=[(ks-1)//2 for ks in kernel_size], bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                          stride=1, padding=[(ks-1)//2 for ks in kernel_size], bias=False))
        else:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                          stride=1, padding=[(ks-1)//2 for ks in kernel_size], bias=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                          stride=1, padding=[(ks-1)//2 for ks in kernel_size], bias=True))

    def forward(self, input):
        return input + self.res_block(input)

class ResBlocks2D(nn.Module):
    def __init__(self, num_channels, kernel_size=3):
        super(ResBlocks2D, self).__init__()
        self.res_blocks = nn.Sequential(
            ResBlock2D(num_channels, kernel_size, bn=False),
            ResBlock2D(num_channels, kernel_size, bn=False),
            ResBlock2D(num_channels, kernel_size, bn=False)
        )

    def forward(self, input):
        return input + self.res_blocks(input)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetComFea(nn.Module):
    def __init__(self, block, layers, args, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetComFea, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # module for compressing feature
        fea_N = 128
        fea_M = 192
        embed_dim = 384

        self.quantizer = quantizer()

        # use FactorizedEntropy
        self.IF_FM = FactorizedEntropy(fea_M)

        # feature encoder and decoder
        # information filtering module
        # 4x down sample feature
        self.IF_enc=Encoder_IF(input_dim=256, depths=[1, 1, 2],base_dim=fea_N)    #depths=[3, 3, 6] new depth=[1,1,2]
        self.IF_dec = Decoder_IF(input_dim=256, depths=[1, 1, 2], base_dim=fea_N)  #depths=[3, 3, 6] new depth=[1,1,2]

        # feature codec
        self.codec_enc = Encoder(base_dim=fea_N, gnconv=[
                       partial(gnconv, order=2, s=1.0/3.0),
                       partial(gnconv, order=3, s=1.0/3.0),
                       partial(gnconv, order=4, s=1.0/3.0, h=24, w=13, gflayer=GlobalLocalFilter),
                   ])
        self.codec_dec = Decoder(base_dim=fea_N, gnconv=[
                       partial(gnconv, order=2, s=1.0/3.0),
                       partial(gnconv, order=3, s=1.0/3.0),
                       partial(gnconv, order=4, s=1.0/3.0, h=24, w=13, gflayer=GlobalLocalFilter),
                   ])

        # self.codec_enc = nn.Sequential(
        #     nn.Conv2d(fea_N*2, fea_N, 1, 1),
        #     ResBlocks2D(fea_N, kernel_size=3),
        #     nn.Conv2d(fea_N, fea_N, 5, 2, 2),
        #     ResBlocks2D(fea_N, kernel_size=3),
        #     nn.Conv2d(fea_N, fea_M, 5, 2, 2),
        # )
        # self.codec_dec = nn.Sequential(
        #     nn.ConvTranspose2d(fea_M, fea_N, 5, 2, 2, 1),
        #     ResBlocks2D(fea_N, kernel_size=3),
        #     nn.ConvTranspose2d(fea_N, fea_N, 5, 2, 2, 1),
        #     ResBlocks2D(fea_N, kernel_size=3),
        #     nn.Conv2d(fea_N, fea_N*2, 1, 1),
        # )
        self.hyper_enc = TransHyperScale(cin=fea_M, cout=fea_M, scale=2, down=True, opt=args)
        self.hyper_dec = TransHyperScale(cin=fea_M, scale=2, down=False, opt=args)
        # Probability model of hyperprior information
        self.prob_model = Entropy(fea_M)

        # AR Transformer Entropy Model and PN module.
        if not args.pretraining_stage:
            if (args.na == 'unidirectional'):
                self.cit_ar = TransDecoder(cin=fea_M, opt=args)
            elif (args.na == 'bidirectional'):
                TransDecoder2.train_scan_mode = 'random' if args.mask_ratio > 0 else 'default'
                self.cit_ar = TransDecoder2(cin=fea_M, opt=args)
            else:
                raise ValueError("No such na.")
            self.cit_ar.apply(vit2_init)

            # Parameter Network
            self.cit_pn = torch.nn.Sequential(
                nn.Conv2d(embed_dim * 2, embed_dim * args.mlp_ratio, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim * args.mlp_ratio, fea_M * args.K * args.num_parameter, 1, 1, 0),
            )
            self.cit_pn.apply(xavier_uniform_init)

            # Construct Criterion
            self.criterion_entropy = DiscretizedMixGaussLoss(rgb_scale=False, x_min=-args.table_range, x_max=args.table_range - 1,
                                                        num_p=args.num_parameter, L=args.table_range * 2)

            self.hyper_enc.apply(vit2_init)
            self.hyper_dec.apply(vit2_init)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, training, use_codec, clamp, return_global_fea):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # information filtering
        fea = x
        omni_fea, IF_y_prob = self.information_filter(x, training)

        t_start=time.time()
        # compress the feature
        if use_codec:
            with torch.no_grad():
                if clamp:
                    omni_fea = omni_fea.clamp(-clamp, clamp)
                omni_fea_hat, comp_y_prob, comp_z_prob = self.compress_fea(
                    omni_fea, training)
                end_time = time.time() - t_start
                print("codec_time: {:.6f}".format(end_time))

            x = omni_fea_hat
        else:
            omni_fea_hat, comp_y_prob, comp_z_prob = None, None, None, None
            x = omni_fea


        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        global_fea = x

        x = self.fc(x)

        # fea_hat : 256x(H/4)x(W/4)
        if return_global_fea:
            return x, fea, omni_fea, IF_y_prob, omni_fea_hat, comp_y_prob, comp_z_prob, global_fea
        else:
            return x, fea, omni_fea, IF_y_prob, omni_fea_hat, comp_y_prob, comp_z_prob

    def forward(self, x, training, use_codec=False, clamp=None, return_global_fea=False):
        return self._forward_impl(x, training, use_codec, clamp, return_global_fea)

    def forward_with_feats(self, x, training, use_codec=False):
        feats = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # information filtering
        fea = x
        omni_fea, IF_y_prob = self.information_filter(x, training)

        # compress the feature
        if use_codec:
            omni_fea_hat, comp_y_prob, comp_z_prob = self.compress_fea(
                omni_fea, training)
            x = omni_fea_hat
        else:
            omni_fea_hat, comp_y_prob, comp_z_prob = None, None, None, None
            x = omni_fea

        feats.append(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        # omni_fea : 256x(H/4)x(W/4)
        return x, fea, omni_fea, IF_y_prob, omni_fea_hat, comp_y_prob, comp_z_prob, feats

    def information_filter(self, fea, training):
        fea_y = self.IF_enc(fea)
        fea_y_hat = self.quantizer(fea_y, training)
        fea_hat = self.IF_dec(fea_y - (fea_y - fea_y.round()).detach())

        fea_y_prob = self.IF_FM(fea_y_hat)
        return fea_hat, fea_y_prob

    def compress_fea(self, fea, training):
        fea_y = self.codec_enc(fea)
        fea_y_hat = self.quantizer(fea_y, training)
        x_hat = self.codec_dec(fea_y_hat)

        z = self.hyper_enc(fea_y)
        z_hat = self.quantizer(z, training)
        y_hyper = self.hyper_dec(z_hat)
        # Auto-regressive Transformer Entropy Model
        feat_ar = self.cit_ar(fea_y_hat)
        # Merge 2 features and Parameter Network
        feat_merge = torch.cat([y_hyper, feat_ar], 1)
        predicted_param = self.cit_pn(feat_merge)

        y_prob=self.criterion_entropy(fea_y_hat, predicted_param)
        z_prob = self.prob_model(z_hat)

        return x_hat, y_prob, z_prob



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def _resnet_cf(arch, block, layers, args, pretrained, progress, **kwargs):
    model = ResNetComFea(block, layers, args, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        msg = model.load_state_dict(state_dict, strict=False)
        print('load weights and the message is {}'.format(msg))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_cf(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_cf('resnet50', Bottleneck, [3, 4, 6, 3], args, pretrained, progress,
                      **kwargs)

'''
def resnet50_hypercompress(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_hypercomress('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                                **kwargs)
'''

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


# -----------------------------------------------------------------------------
# compressing tools
# -----------------------------------------------------------------------------
class FactorizedEntropy(nn.Module):
    # def __init__(self, channel, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6):
    def __init__(self, channel, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-9):
        super(FactorizedEntropy, self).__init__()
        self.channel = channel
        self.init_scale = init_scale
        self.filters = filters
        self.likelihood_bound = likelihood_bound
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        _filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / _filters[i + 1]))
            matrix = nn.Parameter(torch.Tensor(
                1, channel, _filters[i + 1], _filters[i]))
            nn.init.constant_(matrix, init)
            self.matrices.append(matrix)

            bias = nn.Parameter(torch.Tensor(1, channel, _filters[i + 1], 1))
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(bias)

            if i < len(self.filters):
                factor = nn.Parameter(torch.Tensor(
                    1, channel, _filters[i + 1], 1))
                nn.init.constant_(factor, 0)
                self.factors.append(factor)

    def forward(self, input):
        likelihood = self._likelihood(input)
        likelihood = torch.clamp(likelihood, min=self.likelihood_bound)
        return likelihood

    def _logits_cumulative(self, input):
        logits = input
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            matrix = F.softplus(matrix)
            logits = matrix.matmul(logits)
            bias = self.biases[i]
            logits += bias
            if i < len(self.factors):
                factor = self.factors[i]
                factor = torch.tanh(factor)
                logits += factor * torch.tanh(logits)
        return logits

    def _likelihood(self, input):
        shape = input.shape
        B, C = input.shape[0:2]
        input = input.view(B, C, -1).permute(0, 2,
                                             1).contiguous().view(B, -1, C, 1, 1)
        lower = self._logits_cumulative(input - .5)
        upper = self._logits_cumulative(input + .5)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(
            sign * upper) - torch.sigmoid(sign * lower))
        likelihood = likelihood.view(
            B, -1, C).permute(0, 2, 1).contiguous().view(shape)
        return likelihood



class quantizer(nn.Module):
    def __init__(self):
        super(quantizer, self).__init__()

    def quantize(self, input, mode):
        if mode == 'noise':
            noise = input.new(input.shape).uniform_(-0.5, 0.5)
            output = input + noise
            return output
        elif mode == 'round_training':
            output = input - (input - input.round()).detach()
            return output
        else:
            assert mode == 'round'
            output = input.round()
            return output

    def forward(self, x, training):
        x = self.quantize(x, "noise" if training else "round")
        return torch.clamp(x, -128, 127)



if __name__ == '__main__':
    model = resnet50_cf(True)

    x = torch.ones((4, 3, 224, 224))

    out = model(x, training=False)
