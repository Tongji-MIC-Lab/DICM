import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from moco.ops import GDN
import time

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class Block_conv2former(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Encoder_IF(nn.Module):
    def __init__(self, input_dim=256, depths=[4, 4, 8],base_dim=128,
                 block=Block_conv2former, uniform_init=False, **kwargs
                 ):
        super().__init__()

        dims = [base_dim, base_dim, base_dim, base_dim * 3 // 2]
        self.in_con=nn.Conv2d(input_dim,base_dim,1,1)
        self.downsample_layers = nn.ModuleList() # 3 intermediate downsampling conv layers

        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                    LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        for i in range(3):
            stage=nn.Sequential(*[block(base_dim) for j in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(base_dim, eps=1e-6) # final norm layer

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        print('Loading Encoder_IF')

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(3):
            for j, blk in enumerate(self.stages[i]):
                x = blk(x)
            x = self.downsample_layers[i](x)
        return x #(N, C, H, W)

    def forward(self, x):
        x=self.in_con(x)
        x = self.forward_features(x)
        return x


class Decoder_IF(nn.Module):
    def __init__(self,output_dim=256, depths=[4, 4, 8], base_dim=128,
                 block=Block_conv2former, uniform_init=False, **kwargs
                 ):
        super().__init__()

        dims = [base_dim * 3 // 2, base_dim, base_dim, base_dim]
        self.upsample_layers = nn.ModuleList() # 3 intermediate upsampling conv layers

        for i in range(3):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(dims[i],dims[i+1], kernel_size=5, stride=2,padding=2,output_padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks

        for i in range(3):
            stage = nn.Sequential(*[block(base_dim) for j in range(depths[i])])
            self.stages.append(stage)

        self.out=nn.Conv2d(base_dim,output_dim,1,1)

        self.norm = nn.LayerNorm(base_dim, eps=1e-6) # final norm layer

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        print('Loading Decoder_IF')

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(3):
            x = self.upsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                x = blk(x)

        return x #(N, C, H, W)

    def forward(self, x):
        x = self.forward_features(x)
        x=self.out(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim=256, depths=[1, 1, 1], base_dim=128, drop_path_rate=0.5,  # depth=[2,2,2] new depth=[1,1,1]
                 layer_scale_init_value=1e-6,
                 gnconv=gnconv, block=Block, uniform_init=False, **kwargs
                 ):
        super().__init__()

        dims = [input_dim, base_dim, base_dim * 3//2]

        self.conv = nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers


        downsample_layer1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=5, stride=2,padding=2),
            nn.LeakyReLU(inplace=True) #GDN(dims[1])
        )
        self.downsample_layers.append(downsample_layer1)

        downsample_layer2 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True) #GDN(dims[2])
        )
        self.downsample_layers.append(downsample_layer2)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 3 or len(gnconv) == 4

        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.uniform_init = uniform_init

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stages[0](x)
        x=self.downsample_layers[0](x)
        x=self.stages[1](x)
        x=self.conv(x)
        x=self.stages[2](x)
        x=self.downsample_layers[1](x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_dim=256, depths=[1, 1, 1], base_dim=128, drop_path_rate=0.5,   # depth=[2,2,2] new depths=[1,1,1]
                 layer_scale_init_value=1e-6,
                 gnconv=gnconv, block=Block, uniform_init=False, **kwargs
                 ):
        super().__init__()

        dims = [base_dim * 3//2, base_dim, out_dim]

        self.conv=nn.Conv2d(dims[0],dims[1],kernel_size=3,stride=1,padding=1)
        # self.conv_out=nn.Conv2d(dims[2], 3, kernel_size=3, stride=1, padding=1)

        self.upsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers

        upsample_layer1 = nn.Sequential(
            nn.ConvTranspose2d(dims[0],dims[0], kernel_size=5, stride=2,padding=2,output_padding=1),
            nn.LeakyReLU(inplace=True) #GDN(dims[0])
        )
        self.upsample_layers.append(upsample_layer1)

        upsample_layer2 = nn.Sequential(
            nn.ConvTranspose2d(dims[1], dims[2], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True)#GDN(dims[2])
        )
        self.upsample_layers.append(upsample_layer2)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 3 or len(gnconv) == 4

        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.uniform_init = uniform_init

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.upsample_layers[0](x)
        x=self.stages[0](x)
        x=self.conv(x)
        x = self.stages[1](x)
        x = self.upsample_layers[1](x)
        x1 = self.stages[2](x)
        # x2 = self.conv_out(x1)
        return x1

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x