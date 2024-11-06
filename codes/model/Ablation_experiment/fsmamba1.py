import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from model import common
# import common
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

# 在其他地方的 ReLU 操作也需要进行类似的修改
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=False)  # 禁用 inplace

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = out1 + out2 + out3
        out = self.relu(out).clone()  # 使用 clone 避免视图问题
        return out


class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        if num_fea <= 0:
            raise ValueError(f"Invalid num_fea value: {num_fea}")
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, max(1, num_fea // 8), 1, 1, 0),
            nn.ReLU(inplace=False),  # 修改为 inplace=False
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

# class VitModule(nn.Module):
#     def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upsampling=4, window_size=8):
#         super(VitModule, self).__init__()

#         res_num = 5
#         up_scale = upsampling if isinstance(upsampling, int) else upsampling[0]
#         bias = True

#         residual_layer = []
#         self.res_num = res_num

#         for _ in range(res_num):
#             temp_res = OSAG(channel_num=num_feat, window_size=window_size)
#             residual_layer.append(temp_res)
#         self.residual_layer = nn.Sequential(*residual_layer)
#         self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

#         self.window_size = window_size
#         self.up_scale = up_scale

#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
#         return x

#     def forward(self, x):
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)

#         residual = self.input(x)
#         out = self.residual_layer(residual)

#         out = torch.add(self.output(out), residual)
#         out = self.up(out)

#         out = out[:, :, :H * self.up_scale, :W * self.up_scale]
#         return out

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return x * attn

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=False)  # 禁用 inplace

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = out1 + out2 + out3
        out = self.relu(out).clone()  # 避免视图冲突，使用 clone()
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

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        res += x
        return res


# class RDNModule(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(RDNModule, self).__init__()
#         n_feats = 128  # 增加特征通道
#         kernel_size = 3

#         self.head = nn.Sequential(
#             nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
#         )

#         self.body = nn.Sequential(
#             *[RCAB(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.0) for _ in range(30)]  # 增加层数
#         )

#         self.tail = nn.Sequential(
#             nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=kernel_size//2, bias=True)
#         )

#     def forward(self, x):
#         x = self.head(x)
#         x = self.body(x)
#         x = self.tail(x)
#         return x

class EnhancedFSMamba(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EnhancedFSMamba, self).__init__()

        # 只保留FEB模块
        self.feb = FEB(args.n_colors)
        
        # 使用简单的卷积层代替去掉的模块
        self.conv_in = nn.Conv2d(args.n_colors, 128, kernel_size=1)
        self.conv_out = nn.Conv2d(128, args.n_colors, kernel_size=1)
        self.multi_scale = MultiScaleBlock(128)

        # 上采样
        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        # 保留residual路径
        residual = self.upsample(x)

        # 只通过FEB模块处理
        feb_out = self.feb(x)
        feb_out = nn.functional.interpolate(feb_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # 使用多尺度卷积模块进行进一步处理
        combined_out = self.conv_in(feb_out)
        combined_out = self.multi_scale(combined_out)
        combined_out = self.conv_out(combined_out)

        # 将残差加回输出
        out = combined_out + residual

        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 64, stride=2),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128, stride=2),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256, stride=2),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512, stride=2),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        return self.model(img)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.loss_network = nn.Sequential(*list(vgg)[:36]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.mse_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0.
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    import psutil
    import time
    from option import args
    net = EnhancedFSMamba(args).cuda()  # 只初始化 EnhancedFSMamba
    discriminator = Discriminator((3, args.patch_size * args.scale, args.patch_size * args.scale)).cuda()
    perceptual_loss = PerceptualLoss().cuda()
    l1_loss = nn.L1Loss().cuda()
    adversarial_loss = nn.BCEWithLogitsLoss().cuda()
    
    from thop import profile

    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64 * 4, 64 * 4).cuda()
    y = net(x)
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.5fM" % (total / 1e6))
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        timer.toc()

    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))
    flops, params = profile(net, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))

    optimizer_G = torch.optim.Adam(net.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=200, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=200, eta_min=1e-6)

    scaler = GradScaler()

    for epoch in range(200):
        for i in range(100):
            inputs = torch.rand(1, 3, 64, 64).cuda()
            targets = torch.rand(1, 3, 128, 128).cuda()

            optimizer_G.zero_grad()
            with autocast():
                outputs = net(inputs)
                loss_l1 = l1_loss(outputs, targets)
                loss_perceptual = perceptual_loss(outputs, targets)
                loss_adv = adversarial_loss(discriminator(outputs), torch.ones_like(discriminator(outputs)))
                loss_G = loss_l1 + 0.006 * loss_perceptual + 0.001 * loss_adv

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            optimizer_D.zero_grad()
            with autocast():
                loss_real = adversarial_loss(discriminator(targets), torch.ones_like(discriminator(targets)))
                loss_fake = adversarial_loss(discriminator(outputs.detach()), torch.zeros_like(discriminator(outputs.detach())))
                loss_D = (loss_real + loss_fake) / 2

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

        scheduler_G.step()
        scheduler_D.step()
