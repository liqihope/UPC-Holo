import numpy as np
import torch
import torch.fft
import torch.nn as nn
from Hyperparams import *

class Optics:
    def __init__(self, channel, first_z, delta_z, layer_num, factor=0.75, DIWP=True, grating_type='vertical', 
                 LCoS_res_h=2160, LCoS_res_w=3840, LCoS_pitch=3.6e-6, linear_conv=True, band_limit=True, rotate_xy = False):
        # 波长字典,键为通道,值为对应波长
        self.wavelength_dict = {
            0: np.array(4.5e-7),    # 'B'
            1: np.array(5.2e-7),    # 'G'
            2: np.array(6.38e-7)    # 'R'
        }
        self.channel = channel  # 线性卷积 padding
        self.wavelength = self.wavelength_dict[channel]  # 选择对应通道的波长
        self.linear_conv = linear_conv  # 线性卷积 padding
        self.band_limit = band_limit
        self.factor = factor  # 频谱滤波孔径比例
        self.device = Hyperparams.device  # GPU选择
        self.DIWP = DIWP  # 距离整数倍波长处理
        self.first_z = self.distance_int_wavelength_process(first_z)  # 首次传播距离
        self.delta_z = self.distance_int_wavelength_process(delta_z)  # delta传播距离
        self.layer_num = layer_num  # 深度数
        self.LCoS_res_h = LCoS_res_h  # 像素数高
        self.LCoS_res_w = LCoS_res_w  # 像素数宽
        self.LCoS_pitch = LCoS_pitch  # 像素大小，单位m，假设像素是正方形的
        # 默认不考虑高衍射级衍射导致像元投影大小变化的问题
        if rotate_xy:
            self.LCoS_res_w = self.LCoS_pitch * np.cos(np.arcsin(self.wavelength / 2 / self.LCoS_pitch))
        self.LCoS_LC_h = self.LCoS_res_h * self.LCoS_pitch  # 像面高,单位m
        self.LCoS_LC_w = self.LCoS_res_w * self.LCoS_pitch  # 像面宽,单位m
        self.dtype_t = torch.float32
        self.dtype_n = np.float32
        # 相位光栅
        self.phase_grating = self.phase_grating_init(grating_type=grating_type)
        # 预计算传递函数
        # ASM 反向传递函数
        self.h_backward_delta = self.precal_h(prop_z=self.delta_z)
        self.h_backward = self.precal_h(prop_z=-(self.first_z + self.delta_z * (self.layer_num - 1)))
        # ASM首次前向传递函数
        self.h_forward = self.precal_h(prop_z=self.first_z, factor=self.factor)
        self.h_forward_fzf = self.precal_h(prop_z=self.first_z, factor=self.factor)
        # ASM delta 前向传递函数
        self.h_forward_delta = self.precal_h(prop_z=self.delta_z, factor=self.factor)

    def distance_int_wavelength_process(self, z):
        # 距离整数倍波长处理
        if self.DIWP:
            return self.wavelength * round(z / self.wavelength)
        else:
            return z

    # def phase_grating_init(self):
    #     row_D_phase_grating = torch.zeros((1, 1, self.LCoS_res_h, self.LCoS_res_w)).to(self.device)
    #     row_D_phase_grating[..., 0::2, 0::1] = np.pi
    #     return row_D_phase_grating


    def phase_grating_init(self, grating_type='vertical'):
        """
        初始化光栅，支持纵向光栅、横向光栅和二维光栅。

        参数:
            grating_type (str): 光栅类型，可选 'vertical', 'horizontal', '2d'。
                                'vertical' 为纵向光栅，
                                'horizontal' 为横向光栅，
                                '2d' 为二维光栅。
        返回:
            row_D_phase_grating (torch.Tensor): 初始化的光栅。
        """
        # 创建一个全零的相位矩阵
        phase_grating = torch.zeros((1, 1, self.LCoS_res_h, self.LCoS_res_w)).to(self.device)

        if grating_type == 'vertical':
            # 纵向光栅：在宽度方向生成光栅（列隔行设置 π 相位）
            phase_grating[..., 0::2, :] = np.pi
        elif grating_type == 'vertical4':
            # 横向光栅：在高度方向生成光栅（行隔行设置 π 相位）
            phase_grating[..., 0::3, :] = np.pi
            phase_grating[..., 0::4, :] = np.pi
        elif grating_type == 'horizontal':
            # 横向光栅：在高度方向生成光栅（行隔行设置 π 相位）
            phase_grating[..., :, 0::2] = np.pi
        elif grating_type == 'horizontal4':
            # 横向光栅：在高度方向生成光栅（行隔行设置 π 相位）
            phase_grating[..., :, 0::3] = np.pi
            phase_grating[..., :, 0::4] = np.pi
        elif grating_type == '2d':
            # 二维光栅：在高度和宽度方向都生成光栅（棋盘格模式）
            phase_grating[..., 0::2, 0::2] = np.pi
            phase_grating[..., 1::2, 1::2] = np.pi
        elif grating_type == '2d4':
            # 二维光栅：在高度和宽度方向都生成光栅（棋盘格模式）
            phase_grating[..., 0::3, 0::3] = np.pi
            phase_grating[..., 0::4, 0::4] = np.pi
            phase_grating[..., 1::3, 1::3] = np.pi
            phase_grating[..., 1::4, 1::4] = np.pi
        else:
            raise ValueError("Invalid grating_type. Choose from 'vertical', 'horizontal', or '2d'.")
        
        return phase_grating


    def precal_h(self, prop_z, factor=1.0):
        # 线性卷积
        num_y = self.LCoS_res_h * 2 if self.linear_conv else self.LCoS_res_h
        num_x = self.LCoS_res_w * 2 if self.linear_conv else self.LCoS_res_w

        # SLM 采样 双倍
        y, x = (self.LCoS_pitch * float(num_y), self.LCoS_pitch * float(num_x))

        # 频谱采样：frequency coordinates sampling
        # 2023-9-7：频谱采样为什么不是（-1/2/dy,1/2/dy），和y有什么关系？
        # 2024-05-30 自答：因为采样定理不能等于，如果刚好错位一个采样点，1 / (2 * dy) / num_y = 1 / (2 * y)，取一半得到这个采样条件
        fy = np.linspace(-1 / (2 * self.LCoS_pitch) + 0.5 / (2 * y), 1 / (2 * self.LCoS_pitch) - 0.5 / (2 * y), num_y).astype(self.dtype_n)
        fx = np.linspace(-1 / (2 * self.LCoS_pitch) + 0.5 / (2 * x), 1 / (2 * self.LCoS_pitch) - 0.5 / (2 * x), num_x).astype(self.dtype_n)
        # fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), num_y).astype(dtype_n)
        # fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), num_x).astype(dtype_n)
        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)   H.dtype = float64
        # H(fx,fy) = exp{j*k*z*sqrt[1-(λ*fx)**2-(λ*fy)**2]}    其中k = 2*pi/λ = 2*pi*v/c
        # 故：代码中 H = k*sqrt[1-(λ*fx)**2-(λ*fy)**2]
        temp = 1 / (self.wavelength * 1e5) ** 2 - ((FX * 1e-5) ** 2 + (FY * 1e-5) ** 2)
        H_z = prop_z * 2 * np.pi * 1e5 * np.sqrt(temp).astype(self.dtype_n)
        H_prop_z = torch.tensor(H_z, dtype=self.dtype_t).unsqueeze(0).unsqueeze(0).to(self.device)

        # band-limited ASM - Matsushima et al. (2009)
        # 根据波长、采样间隔和传播距离计算出传播角度和最大传播距离，
        # 然后根据最大传播距离计算出空间频率的截止频率，
        # 从而生成一个频率域的滤波器。
        if self.band_limit:
            # x和y方向上传播的角度
            theta_y = np.arcsin(self.wavelength / self.LCoS_pitch)
            theta_x = np.arcsin(self.wavelength / self.LCoS_pitch)
            # x和y方向上的最大传播距离
            zmaxx = max(x / np.tan(theta_x), abs(prop_z))
            zmaxy = max(y / np.tan(theta_y), abs(prop_z))
            # x和y方向上的空间频率的最大截止频率
            fx_max = 1 / np.sqrt((2 * zmaxx * (1 / x)) ** 2 + 1) / self.wavelength * factor
            fy_max = 1 / np.sqrt((2 * zmaxy * (1 / y)) ** 2 + 1) / self.wavelength * factor
            # 与频域图像FX和FY形状相同的二值张量，在频域上对图像进行低通滤波
            H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=self.dtype_t).to(
                self.device)
        else:
            H_filter = torch.tensor(1, dtype=self.dtype_t).to(self.device)
        # 构造一个模为H_filter,相位为H_prop_z的复数张量，即：cpx_in = H_filter*exp(j*H_prop_z)
        transf_func_in = torch.polar(H_filter, H_prop_z)
        # transf_func_H = ifftshift{H_filter*exp(j*H_prop_z)}
        transf_func_H = torch.fft.ifftshift(transf_func_in)
        return transf_func_H

    def prop_asm(self, transf_func_H, u0=None, amp_in=None, phs_in=None):
        # ASM传播
        if u0 is None:
            amp_in = amp_in if amp_in is not None else torch.ones_like(phs_in)
            phs_in = phs_in if phs_in is not None else torch.zeros_like(amp_in)
            u0 = torch.polar(amp_in, phs_in)

        # 直接获取尺寸并计算填充
        b, c, h, w = u0.size()
        if self.linear_conv:
            u0 = nn.ReflectionPad2d((w // 2, w // 2, h // 2, h // 2))(u0)

        U1 = torch.fft.fftn(torch.fft.ifftshift(u0) / np.sqrt(h * w), dim=(-2, -1), norm='ortho')
        U2 = transf_func_H * U1
        u_out = torch.fft.ifftshift(torch.fft.ifftn(U2 * np.sqrt(h * w), dim=(-2, -1), norm='ortho'))
        if self.linear_conv:
            u_out = u_out[..., int(h / 2):int(h / 2) + h, int(w / 2):int(w / 2) + w]
        return u_out


# if __name__ == '__main__':
#     # 测试用例
#     optics = Optics(2, 0.010, 0.005*2**0, 2**2)
#     print(f"波长: {optics.wavelength}")
#     print(f"设备: {optics.device}")
#     print(f"h_forward: {optics.h_forward.shape}")
#     print(f"h_forward: {optics.h_forward_delta.shape}")
#     print(f"h_forward: {optics.h_backward.shape}")
#
#     # 测试distance_int_wavelength_process方法
#     test_z = 0.005
#     processed_z = optics.distance_int_wavelength_process(test_z)
#     print(f"处理后的距离: {processed_z}")