import shutil
import sys
import os
import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
from CNNs import *
from Optics import *
from Hyperparams import *
from RGBDDataset import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class Trainer:
    def __init__(self, optics_ins, phs_code_model, root_path, train_path='', eval_path='', PSD_name='/PSD_EVAL.npy'):
        self.train_path = train_path  # 训练数据路径
        self.eval_path = eval_path  # 验证数据路径
        self.root_path = root_path  # 测试数据路径
        self.model_best_path = self.root_path + '/model_best.pkl'
        self.PSD_path = self.root_path + PSD_name
        self.rgbd_res_h = optics_ins.LCoS_res_h
        self.rgbd_res_w = optics_ins.LCoS_res_w
        self.optics_ins = optics_ins
        self.device = Hyperparams.device  # GPU选择
        if train_path != '':
            self.train_Dataset_ins = RGBDDataset(self.train_path, self.optics_ins)  # 初始化数据集
        if eval_path != '':
            self.eval_Dataset_ins = RGBDDataset(self.eval_path, self.optics_ins)  # 初始化数据集
        self.best_loss = 99999
        self.psnr_best = 0
        # 初始化神经网络模型、优化器、损失函数
        self.phs_net = [phs_code_model.to(self.device), nn.Hardtanh(-torch.pi, torch.pi)]  # 初始化模型
        self.phs_net = nn.Sequential(*(self.phs_net))
        self.loss = nn.L1Loss().to(self.device)
        self.loss_MSE = nn.MSELoss().to(self.device)
        self.loss_SMS = nn.SmoothL1Loss().to(self.device)
        self.loss_SCL = ScaleInvarintLoss().to(self.device)
        self.loss_NPCC = NPCCLoss().to(self.device)
        self.loss_ZYN = ZYNLoss().to(self.device)
        self.optparse = self.phs_net.parameters()
        self.optimizer = torch.optim.Adam(self.optparse, lr=Hyperparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, Hyperparams.learning_rate_step_size, Hyperparams.learning_rate_gamma)

    def train(self):
        for epoch in tqdm(range(Hyperparams.epochs), desc="Epochs"):
            # 训练模式
            self.phs_net.train()
            loss_epoch = 0
            coefficient = 0
            # 记录开始时间
            start_time = time.time()
            train_loader = DataLoader(
                    self.train_Dataset_ins, 
                    batch_size=Hyperparams.batch_size, 
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True
                   )
            # 使用DataLoader迭代批次数据
            for batch_idx, (_, _, image_amp_slm, image_depth_slm, _) in tqdm(enumerate(train_loader),
                                                                                         total=len(train_loader),
                                                                                         desc="Training Batches",
                                                                                         position=1,
                                                                                         leave=True,
                                                                                         dynamic_ncols=True,
                                                                                         miniters=5):
                image_amp_slm = image_amp_slm.to(self.device)
                image_depth_slm = image_depth_slm.to(self.device)
                u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
                _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
                depth_num = self.optics_ins.layer_num
                for dep in range(depth_num):
                    if dep != 0:
                        u[image_depth_slm == (depth_num - dep - 1) / depth_num] = _u[
                            image_depth_slm == (depth_num - dep - 1) / depth_num]
                    if dep == depth_num - 1:
                        u = self.optics_ins.prop_asm(self.optics_ins.h_backward, u0=u)
                    else:
                        u = self.optics_ins.prop_asm(self.optics_ins.h_backward_delta, u0=u)
                        
                # 复振幅
                u = u / torch.max(torch.abs(u))
                # 实部虚部
                # ri = torch.cat((torch.real(u), torch.imag(u)), -3)
                # 幅值相位
                # aa = torch.cat((torch.abs(u), torch.angle(u)), -3)

                if Hyperparams.TEST_MODE and batch_idx >= Hyperparams.TEST_batch:  # 测试模式限制迭代批次
                    break
                depth_num = self.optics_ins.layer_num
                self.optimizer.zero_grad()  # 梯度重置为0
                phs_slm = self.phs_net(u)
                phs_slm = phs_slm - self.optics_ins.phase_grating

                # 全息再现
                rec_amp_slm = torch.zeros_like(image_amp_slm)
                rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
                for dep in range(depth_num):
                    if dep == 0:
                        rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward, phs_in=phs_slm)
                    else:
                        rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward_delta, u0=rec_u)
                    rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                        image_depth_slm == (depth_num - dep - 1) / depth_num]

                coefficient_temp = torch.sum(image_amp_slm) / torch.sum(rec_amp_slm)
                coefficient = coefficient + coefficient_temp
                
                loss_val = coefficient_temp * self.loss(coefficient_temp * rec_amp_slm, image_amp_slm)

                loss_val.backward()  # 损失函数反向传播

                self.optimizer.step()
                self.scheduler.step()
                loss_epoch = loss_epoch + loss_val.item()

            coefficient_avg = coefficient / len(self.train_Dataset_ins)
            torch.save(self.phs_net, self.root_path + '/' + 'model_' + str(epoch) + '.pkl')  # 保存模型
            end_time = time.time()
            print("Train {} | Epoch {}/{} : Training Loss: {:.4f}, Coefficient_Avg:{:.4f}, Time: {:.2f} seconds".format(
                    os.path.splitext(os.path.basename(self.root_path))[0],
                    epoch, Hyperparams.epochs, loss_epoch, coefficient_avg.item(), end_time - start_time))

            psnr_avg = self.evaluate(epoch)

            if self.psnr_best < psnr_avg:
                self.psnr_best = psnr_avg
                torch.save(self.phs_net, self.model_best_path)
            torch.cuda.empty_cache()  # 每一个epoch训练完，主动清空显卡缓存，防止训练中爆显存
            # torch.cuda.synchronize()  # 等待所有设备完成工作


    def evaluate(self, epoch):
        # 评估模式，弃用dropout/固定BN
        self.phs_net.eval()

        with torch.no_grad():
            coefficient = 0
            eval_loader = DataLoader(self.eval_Dataset_ins,
                                    batch_size=1,
                                    num_workers=0,
                                    pin_memory=True
                                    )
            # 使用DataLoader迭代批次数据
            for _, (img, image_ints_ts, image_amp_slm, image_depth_slm, rgb_path) in tqdm(enumerate(eval_loader),
                                                                                         total=len(eval_loader),
                                                                                         desc="Evaluating Batches",
                                                                                         position=1,
                                                                                         leave=True,
                                                                                         dynamic_ncols=True,
                                                                                         miniters=5):
                image_amp_slm = image_amp_slm.to(self.device)
                image_depth_slm = image_depth_slm.to(self.device)
                u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
                _u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
                depth_num = self.optics_ins.layer_num
                for dep in range(depth_num):
                    if dep != 0:
                        u[image_depth_slm == (depth_num - dep - 1) / depth_num] = _u[
                            image_depth_slm == (depth_num - dep - 1) / depth_num]
                    if dep == depth_num - 1:
                        u = self.optics_ins.prop_asm(self.optics_ins.h_backward, u0=u)
                    else:
                        u = self.optics_ins.prop_asm(self.optics_ins.h_backward_delta, u0=u)

                # 复振幅
                u = u / torch.max(torch.abs(u))
                # 实部虚部
                # ri = torch.cat((torch.real(u), torch.imag(u)), -3)
                # 幅值相位
                # aa = torch.cat((torch.abs(u), torch.angle(u)), -3)

                phs_slm = self.phs_net(u)
                if Hyperparams.SAVE:
                    max_phs = 2 * np.pi
                    output_phase = ((phs_slm - phs_slm.mean() + max_phs / 2) % max_phs) / max_phs
                    phase_out_8bit = ((output_phase[0, 0, ...]) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)
                    img_phs_path = r'D_phs_rec/' + os.path.splitext(os.path.basename(rgb_path[0]))[0] + '_' + \
                                   os.path.splitext(os.path.basename(self.root_path))[0] + '_phase.png' # + str(epoch) + '.png'  # 灰度图路径
                    cv2.imwrite(img_phs_path, phase_out_8bit)

                phs_slm = phs_slm - self.optics_ins.phase_grating
                # 全息再现
                rec_amp_slm = torch.zeros_like(image_amp_slm)
                rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
                for dep in range(depth_num):
                    if dep == 0:
                        rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward, phs_in=phs_slm)
                    else:
                        rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward_delta, u0=rec_u)
                    rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                        image_depth_slm == (depth_num - dep - 1) / depth_num]

                # 图像亮度、对比度线性变换
                rec_ints = rec_amp_slm ** 2
                coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
                coefficient = coefficient + coefficient_temp
                rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

                if Hyperparams.SAVE:
                    img_rec_path = r'D_phs_rec/' + os.path.splitext(os.path.basename(rgb_path[0]))[0] + '_' + \
                                   os.path.splitext(os.path.basename(self.root_path))[0] + '_rec.png'# + str(epoch) + '.png'  # 灰度图路径
                    cv2.imwrite(img_rec_path, rec_img_ints)

                rec = rec_img_ints
                img = img.squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)
                psnr_value2 = psnr(img, rec, data_range=255)
                # 计算Y通道分量的均方误差（MSE）
                mse = np.mean((img - rec) ** 2)
                # 计算PSNR（峰值信噪比）
                if mse == 0:
                    psnr_score = float('inf')
                else:
                    max_pixel_value = 255.0
                    psnr_value3 = 20 * np.log10(max_pixel_value / np.sqrt(mse))
                # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
                ssim_value = ssim(img, rec, data_range=255)
                PSNR_SSIM_data = np.array([os.path.splitext(os.path.basename(self.root_path))[0], epoch,
                                           os.path.splitext(os.path.basename(rgb_path[0]))[0], psnr_value2, psnr_value3, ssim_value, coefficient_temp.cpu()])
                self.save_train_data(self.PSD_path, PSNR_SSIM_data)
            coefficient_avg = coefficient / len(self.eval_Dataset_ins)
            PSNR_SSIM_data = self.load_train_data(self.PSD_path)
            LineData = np.zeros((1, 3), dtype='float32')
            LineData[:] = np.sum(PSNR_SSIM_data[epoch * len(self.eval_Dataset_ins):(epoch + 1) * len(self.eval_Dataset_ins), 3:6].astype('float32'),
                axis=0) / len(self.eval_Dataset_ins)
            print("Eval {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient_Avg:{:.4f}".format(
                    os.path.basename(os.path.normpath(self.eval_path)),
                    LineData[0, 1], LineData[0, 2], coefficient_avg.item()))
            psnr_avg = LineData[0, 1]

            if Hyperparams.PSNRSSIM:
                Hyperparams.PSNR.append(LineData[0, 1])
                Hyperparams.SSIM.append(LineData[0, 2])
        return psnr_avg


    def predict(self, img_path, depth_path, phs_path, rec_path, mul_rec_path):
        # 评估模式，弃用dropout/固定BN
        self.phs_net.eval()
        with torch.no_grad():
            # 验证文件是否存在
            if not os.path.exists(img_path) or not os.path.exists(depth_path):
                raise FileNotFoundError(f"Image {img_path} not found in specified path.")

            depth_num = self.optics_ins.layer_num
            # 蓝 绿 红 0 1 2
            img = cv2.imread(img_path)[:, :, self.optics_ins.channel]
            if Hyperparams.DIFF:
                diff = cv2.imread(f'diff/diff_{self.optics_ins.channel}.png')[:, :, self.optics_ins.channel].astype(np.float32) / 255 + 0.001
                img = img.astype(np.float32) / diff
                img = img ** (1/Hyperparams.GAMMA[self.optics_ins.channel])
                img = img / img.max() * 255

            image_depth = cv2.imread(depth_path, 0)

            if img.shape[0] > self.rgbd_res_h or img.shape[1] > self.rgbd_res_w:
                img = cv2.resize(img, (self.rgbd_res_w, self.rgbd_res_h))  # 调整大小
            if image_depth.shape[0] > self.rgbd_res_h or image_depth.shape[1] > self.rgbd_res_w:
                image_depth = cv2.resize(image_depth, (self.rgbd_res_w, self.rgbd_res_h))  # 调整大小

            image = img.astype(np.float32) / 255
            image_amp = np.sqrt(image)  # 相位为0的初始光场
            image_amp_slm = torch.Tensor(image_amp).unsqueeze(0).unsqueeze(0).to(self.device)  # 升高维度后放入显卡
            image_ints_ts = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(self.device)

            # 深度顺序问题，0黑色，255（1）白色，为了符合人眼观察，用1减，255（1）黑色，0白色
            image_depth = image_depth.astype(np.float32) / 255
            # 匹配SLM的深度，进行深度反转 
            if Hyperparams.DEPTH_INVERSION:
                image_depth = 1 - image_depth
            for dep in range(depth_num):  # 深度离散化
                image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
            image_depth_slm = torch.Tensor(image_depth).unsqueeze(0).unsqueeze(0).to(self.device)
            # 5-4,4-3,3-2,2-1,1-0;1-2,2-3,3-4,4-5,5-0
            u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
            _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
            for depth_index in range(depth_num):
                if depth_index != 0:
                    u[image_depth_slm == (depth_num - depth_index - 1) / depth_num] = _u[
                        image_depth_slm == (depth_num - depth_index - 1) / depth_num]
                if depth_index == depth_num - 1:
                    u = self.optics_ins.prop_asm(self.optics_ins.h_backward, u0=u)
                else:
                    u = self.optics_ins.prop_asm(self.optics_ins.h_backward_delta, u0=u)

            u = u / torch.max(torch.abs(u))

            if Hyperparams.TIME:
                total_time = 0

                # 定义 CUDA 事件
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                for i in range(10):
                    # 记录程序开始时间
                    start_event.record()

                    __u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs)
                    _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs)
                    
                    for depth_index in range(depth_num):
                        if depth_index != 0:
                            __u[image_depth_slm == (depth_num - depth_index - 1) / depth_num] = _u[
                                image_depth_slm == (depth_num - depth_index - 1) / depth_num
                            ]
                        if depth_index == depth_num - 1:
                            __u = self.optics_ins.prop_asm(self.optics_ins.h_backward, u0=__u)
                        else:
                            __u = self.optics_ins.prop_asm(self.optics_ins.h_backward_delta, u0=__u)

                    # 记录程序结束时间
                    end_event.record()

                    # 等待事件完成并计算时间
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒

                    if i >= 5:
                        total_time += elapsed_time

                # 计算平均时间
                average_time = total_time / 5
                Hyperparams.time1.append(average_time)
                print(f'Average elapsed time: {average_time:.4f} seconds')

            # 实部虚部
            # ri = torch.cat((torch.real(u), torch.imag(u)), -3)
            # 幅值相位
            # aa = torch.cat((torch.abs(u), torch.angle(u)), -3)
            # 输出实部虚部的强度表示
            Real_Imag = 1
            if Real_Imag == 1:
                # 实部
                real = torch.real(u)
                real_out = (real * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Real.png', real_out)
                # 虚部
                imag = torch.imag(u)
                imag_out = (imag * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Imag.png', imag_out)
                        
            Amp_Phs = 1
            if Amp_Phs == 1:
                # 实部
                amp = torch.abs(u)
                amp_out = (amp * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Amp.png', amp_out)
                # 虚部
                phs = torch.angle(u)
                phs_out = (phs * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Phs.png', phs_out)

            phs_slm = self.phs_net(u)

            if Hyperparams.TIME:
                total_time = 0
                for i in range(35):
                    # 记录程序开始时间
                    torch.cuda.synchronize()
                    start_time = time.time()

                    _ = self.phs_net(u)

                    # 记录程序结束时间
                    torch.cuda.synchronize()
                    end_time = time.time()
                    # 计算程序运行时间
                    elapsed_time = end_time - start_time
                    # print('elapsed_time: {:.4f} seconds'.format(elapsed_time))
                    if i >= 5:
                        total_time = total_time + elapsed_time
                # print('total_time: {:.4f} seconds'.format(total_time / 5))
                Hyperparams.time2.append(total_time / 30)

            if Hyperparams.SAVE:
                max_phs = 2 * np.pi
                output_phase = ((phs_slm - phs_slm.mean() + max_phs / 2) % max_phs) / max_phs
                phase_out_8bit = ((output_phase[0, 0, ...]) * 255).round().cpu().detach().squeeze().numpy().astype(
                    np.uint8)
                cv2.imwrite(phs_path, phase_out_8bit)

            phs_slm = phs_slm - self.optics_ins.phase_grating
            # 全息再现
            rec_amp_slm = torch.zeros_like(image_amp_slm)
            rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
            for dep in range(depth_num):
                if dep == 0:
                    rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward, phs_in=phs_slm)
                else:
                    rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward_delta, u0=rec_u)
                rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                    image_depth_slm == (depth_num - dep - 1) / depth_num]

                if Hyperparams.MUL_SAVE:
                    print('深度'+str(dep))
                    mul_rec_path_temp = mul_rec_path.replace('.png', f"{str(dep)}.png")
                    mul_rec_amp_slm = torch.abs(rec_u)
                    # 图像亮度、对比度线性变换
                    mul_rec_ints = mul_rec_amp_slm * mul_rec_amp_slm
                    mul_rec_ints *= torch.sum(image_ints_ts) / torch.sum(mul_rec_ints)
                    mul_rec_ints = (mul_rec_ints * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(
                        np.uint8)
                    # 保存再现图像
                    cv2.imwrite(mul_rec_path_temp, mul_rec_ints)

            # 图像亮度、对比度线性变换
            rec_ints = rec_amp_slm ** 2

            coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
            rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(
                0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

            if Hyperparams.SAVE:
                cv2.imwrite(rec_path, rec_img_ints)

            rec = rec_img_ints

            psnr_value2 = psnr(img, rec, data_range=255)
            # 计算Y通道分量的均方误差（MSE）
            mse = np.mean((img - rec) ** 2)
            # 计算PSNR（峰值信噪比）
            max_pixel_value = 255.0
            psnr_value3 = 20 * np.log10(max_pixel_value / np.sqrt(mse))
            # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
            ssim_value = ssim(img, rec, data_range=255)
            print("Test Image {} | Dataset {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient:{:.4f}".format(
                os.path.splitext(os.path.basename(img_path))[0],
                os.path.splitext(os.path.basename(self.root_path))[0],
                psnr_value3, ssim_value, coefficient_temp.item()))


    def phs_predict(self, rgb_path, depth_path, phs_path, img_rec_path='', phase_grating=True):
        # 验证文件是否存在
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"Image {rgb_path} not found in specified path.")

        depth_num = self.optics_ins.layer_num
        # 蓝 绿 红 0 1 2
        img = cv2.imread(rgb_path)[:, :, self.optics_ins.channel]
        image_depth = cv2.imread(depth_path, 0)

        image = img.astype(np.float32) / 255
        image_ints_ts = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        image_amp = np.sqrt(image)  # 相位为0的初始光场
        image_amp_slm = torch.Tensor(image_amp).unsqueeze(0).unsqueeze(0).to(self.device)  # 升高维度后放入显卡

        # 深度顺序问题，0黑色，255（1）白色，为了符合人眼观察，用1减，255（1）黑色，0白色
        image_depth = image_depth.astype(np.float32) / 255
        for dep in range(depth_num):  # 深度离散化
            image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
        image_depth_slm = torch.Tensor(image_depth).unsqueeze(0).unsqueeze(0).to(self.device)

        image_phs_slm = cv2.imread(phs_path)[:, :, self.optics_ins.channel].astype(np.float32) / 255 * 2 * np.pi
        phs_slm = torch.Tensor(image_phs_slm).unsqueeze(0).unsqueeze(0).to(self.device)  # 升高维度后放入显卡
        if phase_grating:
            phs_slm = phs_slm - self.optics_ins.phase_grating
        # 全息再现
        rec_amp_slm = torch.zeros_like(image_amp_slm)
        rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
        for dep in range(depth_num):
            if dep == 0:
                rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward, phs_in=phs_slm)
            else:
                rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward_delta, u0=rec_u)
            rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                image_depth_slm == (depth_num - dep - 1) / depth_num]

            if Hyperparams.MUL_SAVE:
                print('深度'+str(dep))
                img_mul_rec_path = r'mul_pred/' + os.path.splitext(os.path.basename(rgb_path))[0] + '_' + \
                            os.path.splitext(os.path.basename(self.root_path))[0] + '_' + \
                            str(self.optics_ins.channel) + '_' + \
                            'rec' + '_' + \
                            str(dep) + '.png'  # 存储路径
                mul_rec_amp_slm = torch.abs(rec_u)
                # 图像亮度、对比度线性变换
                mul_rec_ints = mul_rec_amp_slm * mul_rec_amp_slm
                mul_rec_ints *= torch.sum(image_ints_ts) / torch.sum(mul_rec_ints)
                mul_rec_ints = (mul_rec_ints * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(
                    np.uint8)
                # 保存再现图像
                cv2.imwrite(img_mul_rec_path, mul_rec_ints)

        # 图像亮度、对比度线性变换
        rec_ints = rec_amp_slm ** 2

        coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
        rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

        if Hyperparams.SAVE:
            if img_rec_path=='':
                img_rec_path = r'pred/' + os.path.splitext(os.path.basename(rgb_path))[0] + '_' + \
                            os.path.splitext(os.path.basename(self.root_path))[0] + '_' + \
                            str(self.optics_ins.channel) + '_' + \
                            'rec.png'  # 存储路径
            cv2.imwrite(img_rec_path, rec_img_ints)

        rec = rec_img_ints

        psnr_value2 = psnr(img, rec, data_range=255)
        # 计算Y通道分量的均方误差（MSE）
        mse = np.mean((img - rec) ** 2)
        # 计算PSNR（峰值信噪比）
        max_pixel_value = 255.0
        psnr_value3 = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
        ssim_value = ssim(img, rec, data_range=255)
        print("Test Image {} | Dataset {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient:{:.4f}".format(
            os.path.splitext(os.path.basename(rgb_path))[0],
            os.path.splitext(os.path.basename(self.root_path))[0],
            psnr_value3, ssim_value, coefficient_temp.item()))
            

    def phs_predict_movie(self, rgb_path, depth_path, phs_path, depth_frame, img_rec_path='', phase_grating=True):
        # 验证文件是否存在
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"Image {rgb_path} not found in specified path.")

        depth_num = self.optics_ins.layer_num
        # 蓝 绿 红 0 1 2
        img = cv2.imread(rgb_path)[:, :, self.optics_ins.channel]
        image_depth = cv2.imread(depth_path, 0)

        image = img.astype(np.float32) / 255
        image_ints_ts = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        image_amp = np.sqrt(image)  # 相位为0的初始光场
        image_amp_slm = torch.Tensor(image_amp).unsqueeze(0).unsqueeze(0).to(self.device)  # 升高维度后放入显卡

        # 深度顺序问题，0黑色，255（1）白色，为了符合人眼观察，用1减，255（1）黑色，0白色
        image_depth = image_depth.astype(np.float32) / 255
        # 匹配SLM的深度，进行深度反转 
        if Hyperparams.DEPTH_INVERSION:
            image_depth = 1 - image_depth
        for dep in range(depth_num):  # 深度离散化
            image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
        image_depth_slm = torch.Tensor(image_depth).unsqueeze(0).unsqueeze(0).to(self.device)
        # 5-4,4-3,3-2,2-1,1-0;1-2,2-3,3-4,4-5,5-0
        u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
        _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
        for depth_index in range(depth_num):
            if depth_index != 0:
                u[image_depth_slm == (depth_num - depth_index - 1) / depth_num] = _u[
                    image_depth_slm == (depth_num - depth_index - 1) / depth_num]
            if depth_index == depth_num - 1:
                u = self.optics_ins.prop_asm(self.optics_ins.h_backward, u0=u)
            else:
                u = self.optics_ins.prop_asm(self.optics_ins.h_backward_delta, u0=u)

        u = u / torch.max(torch.abs(u))
        phs_slm = self.phs_net(u)

        max_phs = 2 * np.pi
        output_phase = ((phs_slm - phs_slm.mean() + max_phs / 2) % max_phs) / max_phs
        phase_out_8bit = ((output_phase[0, 0, ...]) * 255).round().cpu().detach().squeeze().numpy().astype(
            np.uint8)
        if Hyperparams.SAVE:

            cv2.imwrite(phs_path, phase_out_8bit)

        if phase_grating:
            phs_slm = phs_slm - self.optics_ins.phase_grating
        # 全息再现
        rec_amp_slm = torch.zeros_like(image_amp_slm)
        rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
        for dep in range(depth_frame):
            if dep == 0:
                rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward, phs_in=phs_slm)
            else:
                rec_u = self.optics_ins.prop_asm(self.optics_ins.h_forward_delta, u0=rec_u)

        # 图像亮度、对比度线性变换
        rec_ints = torch.abs(rec_u) ** 2

        coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
        rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

        if Hyperparams.SAVE:
            if img_rec_path=='':
                img_rec_path = r'pred_movie/' + os.path.splitext(os.path.basename(rgb_path))[0] + '_' + \
                            os.path.splitext(os.path.basename(self.root_path))[0] + '_' + \
                            str(self.optics_ins.channel) + '_' + \
                            'rec.png'  # 存储路径
            cv2.imwrite(img_rec_path, rec_img_ints)

        # rec = rec_img_ints
        # psnr_value2 = psnr(img, rec, data_range=255)
        # # 计算Y通道分量的均方误差（MSE）
        # mse = np.mean((img - rec) ** 2)
        # # 计算PSNR（峰值信噪比）
        # max_pixel_value = 255.0
        # psnr_value3 = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        # # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
        # ssim_value = ssim(img, rec, data_range=255)
        # print("Test Image {} | Dataset {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient:{:.4f}".format(
        #     os.path.splitext(os.path.basename(rgb_path))[0],
        #     os.path.splitext(os.path.basename(self.root_path))[0],
        #     psnr_value3, ssim_value, coefficient_temp.item()))
        
        return phase_out_8bit, rec_img_ints
        

    # 定义一个函数，每次接受一个一维数组，然后将它保存到一个文件中
    @staticmethod
    def save_train_data(data_path, new_data):
        # 指定文件名
        filename = data_path
        # 使用numpy的save函数保存数据
        # 注意：如果文件已经存在，我们需要先加载原有的数据，然后在其基础上添加新的行
        try:
            # 加载原有数据
            train_data = np.load(filename, allow_pickle=True)
            # 添加新的行
            train_data = np.vstack((train_data, new_data))
        except FileNotFoundError:
            # 如果文件不存在，我们就创建一个新的二维数组
            train_data = np.array([new_data])
        # 保存更新后的数据
        np.save(filename, train_data)

    # 定义一个函数，从文件中读取数据，然后返回
    @staticmethod
    def load_train_data(data_path):
        # 指定文件名
        filename = data_path
        # 使用numpy的load函数读取数据
        return np.load(filename, allow_pickle=True)

    # 定义一个函数，用于删除data.npy文件
    @staticmethod
    def delete_train_data(data_path):
        filename = data_path
        if os.path.exists(filename):  # 检查文件是否存在
            os.remove(filename)  # 删除文件
            print(data_path + "Train data file removed successfully!")
        else:
            print(data_path + "Train data file does not exist")


# if __name__ == '__main__':
#     print('非主程序')
#     # LCoS_res_h 代表训练尺寸
#     optics_ins = Optics(2, 0.005, 0.005 * 2 ** 0, 2 ** 2, LCoS_res_h=216, LCoS_res_w=384)
#
#     TEST_MODE = True  # 测试模式
#
#     # Paper2: 'NYU4K_1000', 'MIT4K_1000'
#     work_paths = ['MIT4K_1000', 'NYU4K_1000']
#     work_paths = ['MIT4K_500']
#
#     if TEST_MODE:
#         pred_path = r'eval/'  # 测试集路径
#         eval_path = r'E:/dataset/Testset/MIT_testset/'  # 测试集路径
#         # 测试路径
#         base_path = r'D:/PROJECT/OOP_20241107/D_CMA_36SLM_test' \
#                     + '_CHANNEL_' + str(optics_ins.channel) \
#                     + '_Dn_' + str(optics_ins.layer_num)
#     else:
#         # 基础路径
#         base_path = r'D:/PROJECT/OOP_20241107/D_CMA_36SLM' \
#                     + '_CHANNEL_' + str(optics_ins.channel) \
#                     + '_Dn_' + str(optics_ins.layer_num)
#
#     for index, work_path in enumerate(work_paths):
#         # 路径参数
#         train_path = r'E:/Dataset/Paper_2/' + work_path  # 训练集路径
#         root_path = base_path + '/' + work_path  # 输出路径
#         # 如果目录已存在，则删除它
#         if TEST_MODE:
#             if os.path.exists(root_path):
#                 shutil.rmtree(root_path)
#             os.makedirs(root_path)
#             print('测试模式，已重建路径。')
#         elif os.path.exists(root_path) and not Hyperparams.ModelPrepared:
#             print('非测试模式，请勿删除路径。')
#             sys.exit()  # 停止程序运行
#         elif not os.path.exists(root_path):
#             os.makedirs(root_path)
#         trainer = Trainer(train_path, pred_path, pred_path, optics_ins, root_path)
#         trainer.train()

