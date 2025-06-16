import torch
from torch import nn


# 构造卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, True),
            # nn.Hardtanh(-torch.pi/2, torch.pi/2),
            # nn.SELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, True)
            # nn.Hardtanh(-torch.pi/2, torch.pi/2)
            # nn.SELU()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_block(x)
        return x

    def _initialize_weights(self):
        for m in self.conv_block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# 构造卷积模块
class ConvBlock1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock1, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, True),
            # nn.Hardtanh(-torch.pi/2, torch.pi/2),
            # nn.SELU(),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_block(x)
        return x

    def _initialize_weights(self):
        for m in self.conv_block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # if isinstance(m, nn.Conv2d):
            #     nn.init.constant_(m.weight, torch.pi)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)


# 混合池化(Mixed Pooling)
class MixedPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, max_weight=0.5):
        super(MixedPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)
        self.max_weight = max_weight

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        return self.max_weight * max_pooled + (1 - self.max_weight) * avg_pooled


# 构造上采样模块--转置卷积只能通过这个方法定义
class TransposeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransposeConv, self).__init__()
        self.transpose_conv = nn.Sequential(
            # nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.transpose_conv(x)
        return x


# UNet网络
class UNet(nn.Module):
    # 输入是16个通道的双向*2重组*4实虚*2衍射场，输出重组*4编码
    def __init__(self, in_ch=16, out_ch=4):
        super(UNet, self).__init__()

        # 卷积参数设置
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4]
        # 最大池化
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.conv0_0 = ConvBlock(in_ch, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        # 转置卷积层
        self.transposeConv2_0 = TransposeConv(filters[2], filters[2])
        # 卷积层
        self.conv1_1 = ConvBlock(filters[2] + filters[1], filters[1])
        self.transposeConv1_1 = TransposeConv(filters[1], filters[1])
        self.conv0_1 = ConvBlock(filters[1] + filters[0], filters[0])

        # 全连接层
        self.conv9_9 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # 前向计算，输出重组*4编码矩阵
    def forward(self, input):
        o0_0 = self.conv0_0(input)
        o1_0 = self.conv1_0(self.maxPool(o0_0))

        o2_0 = self.transposeConv2_0(self.conv2_0(self.maxPool(o1_0)))

        o1_1 = self.transposeConv1_1(self.conv1_1(torch.cat((o1_0, o2_0), dim=1)))
        o0_1 = self.conv0_1(torch.cat((o1_1, o0_0), dim=1))

        output = self.conv9_9(o0_1)

        return output


# CNNMaxAvg网络 ####
class CNNMaxAvgSingle(nn.Module):
    # 输入是16个通道的双向*2重组*4实虚*2衍射场，输出重组*4编码
    def __init__(self, in_ch=8, out_ch=4):
        super(CNNMaxAvgSingle, self).__init__()
        # 卷积参数设置
        n1 = 4
        filters = [n1, n1 * 2]
        # 最大池化
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 最大池化
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.conv0_0 = ConvBlock1(in_ch, filters[0])
        # 卷积层
        self.conv1_0 = ConvBlock1(filters[1], filters[0])
        # 转置卷积层
        self.transposeConv1_0 = TransposeConv(filters[0], filters[0])
        # 全连接层
        self.conv9_9 = nn.Conv2d(filters[0] + filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.shuffle = nn.PixelShuffle(2)  # 图像重组函数
        self.unShuffle = nn.PixelUnshuffle(2)  # 图像分割函数

    # 前向计算，输出重组*4编码矩阵
    def forward(self, input):
        input_info = torch.cat((torch.real(input), torch.imag(input)), -3)
        input_unShuffle = self.unShuffle(input_info)
        o0_0 = self.conv0_0(input_unShuffle)
        o1_0 = self.conv1_0(torch.cat((self.maxPool(o0_0), self.avgPool(o0_0)), dim=1))
        output = self.conv9_9(torch.cat((o0_0, self.transposeConv1_0(o1_0)), dim=1))
        output_Shuffle = self.shuffle(output)
        return output_Shuffle


# CNNMaxAvg网络 ####
class FAN(nn.Module):
    # 输入是16个通道的双向*2重组*4实虚*2衍射场，输出重组*4编码
    def __init__(self, in_ch=24, out_ch=4):
        super(FAN, self).__init__()
        # 卷积参数设置
        n1 = 4
        filters = [n1, n1 * 3, n1 * 6]
        # 最大池化
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 最大池化
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.conv0_0 = ConvBlock1(in_ch, filters[0])
        # 卷积层
        self.conv1_0 = ConvBlock1(filters[1], filters[0])
        # 转置卷积层
        self.transposeConv1_0 = TransposeConv(filters[0], filters[0])
        # 全连接层
        self.conv9_9 = nn.Conv2d(filters[2], out_ch, kernel_size=1, stride=1, padding=0)

        self.shuffle = nn.PixelShuffle(2)  # 图像重组函数
        self.unShuffle = nn.PixelUnshuffle(2)  # 图像分割函数

    # 前向计算，输出重组*4编码矩阵
    def forward(self, input):
        input_info = torch.cat((torch.real(input), torch.imag(input)), -3)
        input_unShuffle = self.unShuffle(input_info)
        o0_0 = self.conv0_0(torch.cat((torch.cos(input_unShuffle), torch.sin(input_unShuffle), input_unShuffle), -3))
        i1_0 = self.maxPool(o0_0)
        o1_0 = self.conv1_0(torch.cat((torch.cos(i1_0), torch.sin(i1_0), i1_0), -3))
        i9_9 = torch.cat((o0_0, self.transposeConv1_0(o1_0)), dim=1)
        output = self.conv9_9(torch.cat((torch.cos(i9_9), torch.sin(i9_9), i9_9), -3))
        output_Shuffle = self.shuffle(output)
        return output_Shuffle


# CNNMaxAvg网络 ####
class FAN_cos(nn.Module):
    # 输入是16个通道的双向*2重组*4实虚*2衍射场，输出重组*4编码
    def __init__(self, in_ch=8, out_ch=4):
        super(FAN_cos, self).__init__()
        # 卷积参数设置
        n1 = 4
        filters = [n1, n1 * 2, n1 * 3]
        # 最大池化
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 最大池化
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.conv0_0 = ConvBlock1(in_ch, filters[0])
        # 卷积层
        self.conv1_0 = ConvBlock1(filters[1], filters[0])
        # 转置卷积层
        self.transposeConv1_0 = TransposeConv(filters[0], filters[0])
        # 全连接层
        self.conv9_9 = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)

        self.shuffle = nn.PixelShuffle(2)  # 图像重组函数
        self.unShuffle = nn.PixelUnshuffle(2)  # 图像分割函数

    # 前向计算，输出重组*4编码矩阵
    def forward(self, input):
        input_info = torch.cat((torch.real(input), torch.imag(input)), -3)
        input_unShuffle = self.unShuffle(input_info)
        o0_0 = self.conv0_0(torch.cos(input_unShuffle))
        i1_0 = torch.cat((self.maxPool(o0_0), self.avgPool(o0_0)), dim=1)
        o1_0 = self.conv1_0(torch.cos(i1_0))
        i9_9 = torch.cat((o0_0, self.transposeConv1_0(o1_0)), dim=1)
        output = self.conv9_9(torch.cos(i9_9))
        output_Shuffle = self.shuffle(output)
        return output_Shuffle


# CNNMaxAvg网络 ####
class CNNMaxAvgSingle_21(nn.Module):
    # 输入是16个通道的双向*2重组*4实虚*2衍射场，输出重组*4编码
    def __init__(self, in_ch=2, out_ch=1):
        super(CNNMaxAvgSingle_21, self).__init__()
        # 卷积参数设置
        n1 = 5
        filters = [n1, n1 * 2]
        # 最大池化
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 最大池化
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.conv0_0 = ConvBlock1(in_ch, filters[0])
        # 卷积层
        self.conv1_0 = ConvBlock1(filters[1], filters[0])
        # 转置卷积层
        self.transposeConv1_0 = TransposeConv(filters[0], filters[0])
        # 全连接层
        self.conv9_9 = nn.Conv2d(filters[0] + filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # 前向计算，输出重组*4编码矩阵
    def forward(self, input):
        input_info = torch.cat((torch.real(input), torch.imag(input)), -3)
        o0_0 = self.conv0_0(input_info)
        o1_0 = self.conv1_0(torch.cat((self.maxPool(o0_0), self.avgPool(o0_0)), dim=1))
        output = self.conv9_9(torch.cat((o0_0, self.transposeConv1_0(o1_0)), dim=1))
        return output


# 损失函数出自:Depth Map Prediction from a Single Imageusing a Multi-Scale Deep Network
class ScaleInvarintLoss(nn.Module):
    def __init__(self):
        super(ScaleInvarintLoss, self).__init__()

    def forward(self, valid_out, valid_gt):

        logdiff = torch.log(valid_out+1e-6) - torch.log(valid_gt+1e-6)
        # scale_inv_loss = fac * (torch.sqrt((logdiff ** 2).mean() - 0.85 * (logdiff.mean() ** 2)) * 10.0)
        # Paper 2
        scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85 * (logdiff.mean() ** 2))
        # print((logdiff ** 2).mean())
        # print((logdiff.mean() ** 2))
        # Paper 1
        # scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - logdiff.mean() ** 2)
        return scale_inv_loss


# 损失函数出自:4K-DMD
class NPCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NPCCLoss, self).__init__()
        self.eps = eps  # 避免除零问题

    def forward(self, recon_amps, target_amps):

        # 计算均值（对 H, W 进行均值归一化，而不是对 batch 归一化）
        recon_mean = recon_amps.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        target_mean = target_amps.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # 计算分子 (协方差) - 使用 mean() 归一化
        numerator = torch.mean((recon_amps - recon_mean) * (target_amps - target_mean), dim=(2, 3))

        # 计算分母 (标准差乘积)
        recon_var = torch.var(recon_amps, dim=(2, 3), unbiased=True)  # (B, C)
        target_var = torch.var(target_amps, dim=(2, 3), unbiased=True)

        denominator = torch.sqrt(recon_var * target_var + self.eps)  # (B, C)

        # 计算 NPCC（归一化互相关系数）
        npcc = numerator / (denominator + self.eps)  # 避免除零

        # 返回 1 - NPCC，保证 loss 范围为 [0, 2]，更易优化
        return 1 - npcc.mean()
    

# 损失函数出自:ZYN
class ZYNLoss(nn.Module):
    def __init__(self):
        super(ZYNLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, rec_amp_slm, image_amp_slm):
        """计算损失值"""
        coefficient_temp = torch.sum(image_amp_slm) / torch.sum(rec_amp_slm)
        loss_val = self.loss(coefficient_temp*rec_amp_slm*255, image_amp_slm*255) + coefficient_temp**10
        return loss_val, coefficient_temp