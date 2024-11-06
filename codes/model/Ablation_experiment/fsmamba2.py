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

class VitModule(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upsampling=4, window_size=8):
        super(VitModule, self).__init__()

        res_num = 5
        up_scale = upsampling if isinstance(upsampling, int) else upsampling[0]
        bias = True

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
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

class EnhancedFSMamba(nn.Module):
    def __init__(self, args):
        super(EnhancedFSMamba, self).__init__()

        self.vit = VitModule(num_in_ch=args.n_colors, num_out_ch=args.n_colors, num_feat=128, upsampling=args.scale, window_size=8)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        vit_out = self.vit(x)
        vit_out = nn.functional.interpolate(vit_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        out = vit_out + residual

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
