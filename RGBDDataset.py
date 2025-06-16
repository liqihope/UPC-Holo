import os
import cv2
from torch.utils.data import Dataset, DataLoader
import random
from Optics import *
import re
from Hyperparams import *


class RGBDDataset(Dataset):
    def __init__(self, dataset_path, optics_ins, padding=False):
        # 初始化路径和图像分辨率
        self.device = Hyperparams.device  # GPU选择
        self.dataset_path = dataset_path
        self.rgbd_res_h = optics_ins.LCoS_res_h
        self.rgbd_res_w = optics_ins.LCoS_res_w
        self.optics_ins = optics_ins
        self.padding = padding
        self.ROTATE = True
        if re.search(r'ZS_CGH', self.dataset_path):
            # print("Using random data generation.")
            self.dataset_files = None
            self.random_data_size = Hyperparams.train_image_num  # 设置随机数据的大小
        elif re.search(r'FZF_CGH', self.dataset_path):
            # print("Using random data generation.")
            self.dataset_files = None
            self.random_data_size = Hyperparams.train_image_num  # 设置随机数据的大小
        else:
            self.dataset_files = self._load_files(self.dataset_path)

    def _load_files(self, path):
        # 从给定路径加载图像文件名列表
        # img_folder = os.path.join(path, 'img')
        # depth_folder = os.path.join(path, 'depth')

        pattern = re.compile(r'_depth\.png$')
        # 找到图像文件的基本名
        img_files = [f for f in os.listdir(path) if not pattern.search(f) and os.path.isfile(os.path.join(path, f))]

        # 返回 (RGB图像路径, 深度图像路径) 的列表
        return [(os.path.join(path, f), os.path.join(path, f.replace('.png', '_depth.png')))
                for f in img_files]

    def _generate_random_image(self):
        if re.search('ZS_CGH_LJQ', self.dataset_path):
            mean = int(re.search(r'\d+', self.dataset_path).group())
            std = int(mean / 2)
            maxRandomHeight = np.round(np.random.normal(mean, std, 3)).astype(int).clip(1, self.rgbd_res_h)
            maxRandomWidth = np.round(np.random.normal(mean * self.rgbd_res_w / self.rgbd_res_h, std * self.rgbd_res_w / self.rgbd_res_h, 3)).astype(int).clip(1, self.rgbd_res_w)
            img0 = np.random.randint(0, 256, (maxRandomHeight[0], maxRandomWidth[0]), dtype=np.uint8)
            img0 = cv2.resize(img0, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
            img1 = np.random.randint(0, 256, (maxRandomHeight[1], maxRandomWidth[1]), dtype=np.uint8)
            img1 = cv2.resize(img1, (self.rgbd_res_w, self.rgbd_res_h), interpolation=1)
            img2 = np.random.randint(0, 256, (maxRandomHeight[2], maxRandomWidth[2]), dtype=np.uint8)
            img2 = cv2.resize(img2, (self.rgbd_res_w, self.rgbd_res_h), interpolation=2)
            img_noise = np.random.randint(0, 256, (random.randint(1, 540), random.randint(1, 960)), dtype=np.uint8)
            img_noise = cv2.resize(img_noise, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
            img = img0 + img1 % random.randint(100, 256) + img2 % random.randint(100, 256) + img_noise * random.random() * 0.1
            if self.ROTATE:
                img = (img + random.randint(0, 256)) % 256  # 强度轮转，让训练无死角
            maxRandomHeight_depth = np.round(np.random.normal(mean / 2, std / 2, 1)).astype(int).clip(1, self.rgbd_res_h)
            maxRandomWidth_depth = np.round(np.random.normal(mean / 2 * self.rgbd_res_w / self.rgbd_res_h, std / 2 * self.rgbd_res_w / self.rgbd_res_h, 1)).astype(int).clip(1, self.rgbd_res_w)
            image_depth = np.random.randint(0, 256, (maxRandomHeight_depth[0], maxRandomWidth_depth[0]), dtype=np.uint8)
            image_depth = cv2.resize(image_depth, (self.rgbd_res_w, self.rgbd_res_h), interpolation=np.random.randint(1, 3))
            img_depth_noise = np.random.randint(0, 256, (random.randint(1, self.rgbd_res_h), random.randint(1, self.rgbd_res_w)), dtype=np.uint8)
            img_depth_noise = cv2.resize(img_depth_noise, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
            image_depth = image_depth + img_depth_noise * random.random() * 0.1
            if self.ROTATE:
                image_depth = (image_depth + random.randint(0, 256)) % 256  # 强度轮转，让训练无死角
            img = img.astype(np.uint8)
            image_depth = image_depth.astype(np.uint8)
        elif re.search('ZS_CGH_NNU', self.dataset_path):
            Height = int(re.search(r'\d+', self.dataset_path).group())
            Width = int(Height * self.rgbd_res_w / self.rgbd_res_h)
            img = np.random.randint(0, 256, (Height, Width), dtype=np.uint8)
            img = cv2.resize(img, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
            if self.ROTATE:
                img = (img + random.randint(0, 256)) % 256  # 强度轮转，让训练无死角
            image_depth = np.random.randint(0, 256, (np.random.randint(1, self.rgbd_res_h), np.random.randint(1, self.rgbd_res_w)), dtype=np.uint8)
            image_depth = cv2.resize(image_depth, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
            if self.ROTATE:
                image_depth = (image_depth + random.randint(0, 256)) % 256  # 强度轮转，让训练无死角
            img = img.astype(np.uint8)
            image_depth = image_depth.astype(np.uint8)
        elif re.search('ZS_CGH_PR', self.dataset_path):
            fac = int(re.search(r'\d+', self.dataset_path).group())
            Height = 2 ** np.random.randint(2, fac)
            Width = 2 ** np.random.randint(2, fac)
            img = np.random.randint(0, 256, (np.random.randint(1, Height), np.random.randint(1, Width)), dtype=np.uint8)
            img = cv2.resize(img, (self.rgbd_res_w, self.rgbd_res_h), interpolation=np.random.randint(0, 3))
            if self.ROTATE:
                img = (img + random.randint(0, 256)) % 256  # 强度轮转，让训练无死角
            image_depth = np.random.randint(0, 256, (np.random.randint(1, self.rgbd_res_h), np.random.randint(1, self.rgbd_res_w)), dtype=np.uint8)
            image_depth = cv2.resize(image_depth, (self.rgbd_res_w, self.rgbd_res_h), interpolation=np.random.randint(0, 3))
            if self.ROTATE:
                image_depth = (image_depth + random.randint(0, 256)) % 256  # 强度轮转，让训练无死角
            img = img.astype(np.uint8)
            image_depth = image_depth.astype(np.uint8)
        return img, image_depth


    def __len__(self):
        if self.dataset_files is None:
            return self.random_data_size
        return len(self.dataset_files)

    def __getitem__(self, idx):
        # 读取RGB图像和对应的深度图像
        depth_num = self.optics_ins.layer_num

        if re.search('FZF_CGH', self.dataset_path):
            real, imag = self._generate_fzf()
            ###################### 分布
            # img = (img/255)**2*255
            rgb_path = 'img/rand_fzf.png'

            real = real.astype(np.float32) / 255
            real_slm = torch.Tensor(real).unsqueeze(0)  # 升高维度

            imag = imag.astype(np.float32) / 255
            imag_slm = torch.Tensor(imag).unsqueeze(0)  # 升高维度
            return real_slm, imag_slm, rgb_path

        if re.search('ZS_CGH', self.dataset_path):
            img, image_depth = self._generate_random_image()
            ###################### 分布
            # img = (img/255)**2*255
            rgb_path = 'img/rand.png'
        else:
            rgb_path, depth_path = self.dataset_files[idx]
            # print(f"Loading RGB: {rgb_path}, Depth: {depth_path}")
            # 蓝 绿 红 0 1 2
            img = cv2.imread(rgb_path)[:, :, self.optics_ins.channel]
            image_depth = cv2.imread(depth_path, 0)

        if img.shape[0] > self.rgbd_res_h or img.shape[1] > self.rgbd_res_w:
            img = cv2.resize(img, (self.rgbd_res_w, self.rgbd_res_h))  # 调整大小
        if image_depth.shape[0] > self.rgbd_res_h or image_depth.shape[1] > self.rgbd_res_w:
            image_depth = cv2.resize(image_depth, (self.rgbd_res_w, self.rgbd_res_h))  # 调整大小

        image = img.astype(np.float32) / 255
        image_ints_ts = torch.Tensor(image).unsqueeze(0)

        image_amp = np.sqrt(image)  # 相位为0的初始光场
        image_amp_slm = torch.Tensor(image_amp).unsqueeze(0)  # 升高维度
        # 深度顺序问题，0黑色，255（1）白色，为了符合人眼观察，用1减，255（1）黑色，0白色
        image_depth = image_depth.astype(np.float32) / 255
        for dep in range(depth_num):  # 深度离散化
            image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
        image_depth_slm = torch.Tensor(image_depth).unsqueeze(0)
        return img, image_ints_ts, image_amp_slm, image_depth_slm, rgb_path


# if __name__ == '__main__':
#     # 测试用例
#     train_path = 0
#     eval_path = 'dataset'
#     # LCoS_res_h 代表训练尺寸
#     optics_ins = Optics(2, 0.005, 0.005 * 2 ** 0, 2 ** 2, LCoS_res_h=216, LCoS_res_w=384)
#     train_dataset = RGBDDataset(train_path, optics_ins)
#
#     # 读取一个训练样本,复振幅 ,实部虚部, 幅值相位
#     u, ri, aa, _, _, image_path = train_dataset[0]
#     print("训练样本尺寸:", u.shape)
#
#     # 测试加载特定图像
#     u2, ri2, aa2, _, _, image_path = train_dataset.load_test_image('img/bbb.png', 'img/bbb_depth.png')
#     print("测试样本尺寸:", u.shape)
#
#     eval_dataset = RGBDDataset(eval_path, optics_ins)
#     eval_loader = DataLoader(eval_dataset, batch_size=3, shuffle=True, drop_last=False)
#     # 使用DataLoader迭代批次数据
#     for batch_idx, (u, ri, aa, image_amp_slm, image_depth_slm, image_path) in enumerate(eval_loader):
#         # 此时u, ri, aa, image_amp_slm, image_depth_slm已经是一个批次的数据
#         print(f"Batch {batch_idx} shapes:")
#         print("u shape:", u.shape)  # [batch_size, channels, height, width]
#         print("ri shape:", ri.shape)  # [batch_size, channels, height, width]
#         print("aa shape:", aa.shape)  # [batch_size, channels, height, width]
#         print("image_amp_slm shape:", image_amp_slm.shape)  # [batch_size, 1, height, width]
#         print("image_depth_slm shape:", image_depth_slm.shape)  # [batch_size, 1, height, width]

