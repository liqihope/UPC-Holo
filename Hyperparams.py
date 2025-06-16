import torch

class Hyperparams:
    TEST_MODE = False
    batch_size = 1          # 批次大小
    TEST_batch = 50         # 最高批次数量
    SAVE = False
    MUL_SAVE = False
    TIME = False      # 计时，两阶段计时
    DIFF = False      # 拟合SLM强度不均匀情况DIFF系数
    DEPTH_INVERSION = False # 匹配SLM的深度，进行深度反转 
    GAMMA = [1, 1, 1]
    time1 = []
    time2 = []
    PSNRSSIM = False
    PSNR = []
    SSIM = []
    init_phs = 0/4
    ModelPrepared = False
    learning_rate = 5e-4 * batch_size * 2   # 学习率
    epochs = 50             # 训练轮数
    train_image_num = 200   # 训练图像数量
    optimizer = 'Adam'      # 优化器选择
    learning_rate_step_size = int(epochs * train_image_num / batch_size / 5)  # 每2000次训练
    learning_rate_gamma = 0.5  # 学习率下降自身的50%
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
