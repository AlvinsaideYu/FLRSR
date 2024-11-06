import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


def make_model(args, parent=False):
    return EnhancedFSMamba(args)


class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        if num_fea <= 0:
            raise ValueError(f"Invalid num_fea value: {num_fea}")
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, max(1, num_fea // 8), 1, 1, 0),
            nn.ReLU(inplace=False),  # inplace=False to avoid modifying views
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


class RCAB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, reduction=16, bias=True, act=nn.ReLU(False), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)  # Ensure ReLU with inplace=False
        self.body = nn.Sequential(*modules_body)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=False),  # inplace=False to avoid view modifications
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
        n_feats = 128  # Increased feature channels
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        self.body = nn.Sequential(
            *[RCAB(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.0) for _ in range(30)]  # Increased layers
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

        self.feb = FEB(args.n_colors)
        self.rdn = RDNModule(args, conv)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        feb_out = self.feb(x)
        feb_out = nn.functional.interpolate(feb_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        rdn_out = self.rdn(x)
        rdn_out = nn.functional.interpolate(rdn_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        out = feb_out + rdn_out
        out += residual

        return out
