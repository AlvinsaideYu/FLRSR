import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from model import common

def make_model(args, parent=False):
    return BasicFSMamba(args)

class pixelshuffle_block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, bias=True):
        super(pixelshuffle_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=bias)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

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

class BasicFSMamba(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(BasicFSMamba, self).__init__()

        self.attention = AttentionModule(128)
        self.conv_in = nn.Conv2d(args.n_colors, 128, kernel_size=1)
        self.conv_out = nn.Conv2d(128, args.n_colors, kernel_size=1)
        self.multi_scale = MultiScaleBlock(128)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        combined_out = self.conv_in(x)
        combined_out = self.multi_scale(combined_out)
        combined_out = self.attention(combined_out)
        combined_out = self.conv_out(combined_out)

        # 调整 combined_out 尺寸以匹配 residual
        combined_out = F.interpolate(combined_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

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
    net = BasicFSMamba(args).cuda()
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
