import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


def make_model(args, parent=False):
    return EnhancedFSMamba(args)


class pixelshuffle_block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, bias=True):
        super(pixelshuffle_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=bias)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class OSAG(nn.Module):
    def __init__(self, channel_num, window_size):
        super(OSAG, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)  # 禁用 inplace=True
        self.conv2 = nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.window_size = window_size

    def forward(self, x):
        _, _, h, w = x.size()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
        out = self.relu(self.conv1(x)).clone()  # 确保不会修改视图
        out = self.conv2(out)
        return out


class VitModule(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upsampling=4, window_size=8):
        super(VitModule, self).__init__()

        res_num = 5
        up_scale = upsampling if isinstance(upsampling, int) else upsampling[0]
        bias = True

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, window_size=window_size)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        self.window_size = window_size
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out


class RCAB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, reduction=16, bias=True, act=nn.ReLU(False), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)  # 确保所有 ReLU 使用 inplace=False
        self.body = nn.Sequential(*modules_body)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=False),  # 这里的 ReLU 确保 inplace=False
            nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        res += x
        return res


class RDNModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RDNModule, self).__init__()
        n_feats = 128  # 增加特征通道
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size // 2, bias=True)
        )

        self.body = nn.Sequential(
            *[RCAB(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.0) for _ in range(30)]  # 增加层数
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=kernel_size // 2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class EnhancedFSMamba(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EnhancedFSMamba, self).__init__()

        self.vit = VitModule(num_in_ch=args.n_colors, num_out_ch=args.n_colors, num_feat=128, upsampling=args.scale, window_size=8)
        self.rdn = RDNModule(args, conv)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        vit_out = self.vit(x)
        vit_out = nn.functional.interpolate(vit_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        rdn_out = self.rdn(x)
        rdn_out = nn.functional.interpolate(rdn_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        combined_out = vit_out + rdn_out
        out = combined_out + residual

        return out
