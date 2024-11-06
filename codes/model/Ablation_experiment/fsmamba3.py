import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
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

class RCAB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, reduction=16, bias=True, act=nn.ReLU(False), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0: 
                modules_body.append(act)  # 确保所有 ReLU 使用 inplace=False
        self.body = nn.Sequential(*modules_body)

        # Channel Attention (CA) Layer
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
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        self.body = nn.Sequential(
            *[RCAB(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.0) for _ in range(30)]  # 增加层数
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=kernel_size//2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class EnhancedFSMamba(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EnhancedFSMamba, self).__init__()

        self.rdn = RDNModule(args, conv)
        self.conv_in = nn.Conv2d(args.n_colors, 128, kernel_size=1)
        self.conv_out = nn.Conv2d(128, args.n_colors, kernel_size=1)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        rdn_out = self.rdn(x)
        rdn_out = nn.functional.interpolate(rdn_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        out = rdn_out + residual
        return out

if __name__ == '__main__':
    from option import args
    net = EnhancedFSMamba(args).cuda()

    optimizer_G = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=200, eta_min=1e-6)
    scaler = GradScaler()

    for epoch in range(200):
        for i in range(100):
            inputs = torch.rand(1, 3, 64, 64).cuda()
            targets = torch.rand(1, 3, 128, 128).cuda()

            optimizer_G.zero_grad()
            with autocast():
                outputs = net(inputs)
                loss_l1 = nn.L1Loss().cuda()(outputs, targets)

            scaler.scale(loss_l1).backward()
            scaler.step(optimizer_G)
            scaler.update()

        scheduler_G.step()
