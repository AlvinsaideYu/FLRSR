import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

def make_model(args, parent=False):
    return EnhancedFSMamba(args)

class pixelshuffle_block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, bias=True):
        super(pixelshuffle_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=bias)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, max(1, num_fea // 8), 1, 1, 0),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(1, num_fea // 8), num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fea):
        return self.conv_du(fea)

class LLBlock(nn.Module):
    def __init__(self, num_fea):
        super(LLBlock, self).__init__()
        self.channel1 = num_fea // 2
        self.channel2 = num_fea - self.channel1
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        )

        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.channel1, self.channel2], dim=1)

        x1 = self.convblock(x1)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A * x1), dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B * x2), dim=1)

        c = torch.cat((self.fuse1(P), self.fuse2(Q)), dim=1)
        out = self.fuse(c)
        return out

class AF(nn.Module):
    def __init__(self, num_fea):
        super(AF, self).__init__()
        self.CA1 = CALayer(num_fea)
        self.CA2 = CALayer(num_fea)
        self.fuse = nn.Conv2d(num_fea * 2, num_fea, 1)

    def forward(self, x1, x2):
        x1 = self.CA1(x1) * x1
        x2 = self.CA2(x2) * x2
        return self.fuse(torch.cat((x1, x2), dim=1))

class FEB(nn.Module):
    def __init__(self, num_fea):
        super(FEB, self).__init__()
        self.CB1 = LLBlock(num_fea)
        self.CB2 = LLBlock(num_fea)
        self.CB3 = LLBlock(num_fea)
        self.AF1 = AF(num_fea)
        self.AF2 = AF(num_fea)

    def forward(self, x):
        x1 = self.CB1(x)
        x2 = self.CB2(x1)
        x3 = self.CB3(x2)
        f1 = self.AF1(x3, x2)
        f2 = self.AF2(f1, x1)
        return x + f2

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

        self.feb = FEB(args.n_colors)
        self.vit = VitModule(num_in_ch=args.n_colors, num_out_ch=args.n_colors, num_feat=128, upsampling=args.scale, window_size=8)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        feb_out = self.feb(x)
        feb_out = nn.functional.interpolate(feb_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        vit_out = self.vit(x)
        vit_out = nn.functional.interpolate(vit_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        combined_out = feb_out + vit_out
        out = combined_out + residual

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
