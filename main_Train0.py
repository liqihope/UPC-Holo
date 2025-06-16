from Trainer import *
from Hyperparams import *

if __name__ == '__main__':
    Hyperparams.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Hyperparams.TEST_MODE = True
    Hyperparams.SAVE = True
    # Paper2: 'NYU4K_500', 'MIT4K_500'
    # work_paths = ['MIT4K_500', 'NYU4K_500']
    work_paths = ['NYU_4K', '3D_train_xinlei', 'DIV2K_Avg_2']
    # work_paths = ['fruitandvegetable']
    # LCoS_res_h 代表训练尺寸
    optics_ins = Optics(2, 0.005, 0.005 * 2 ** 0, 2 ** 2, LCoS_res_h=1080, LCoS_res_w=1920, LCoS_pitch=3.6e-6)
    if Hyperparams.TEST_MODE:
        eval_path = r'eval/'  # 测试集路径
        # 测试路径
        base_path = r'models/CMA_ZYC_test' \
                    + '_CHANNEL_' + str(optics_ins.channel) \
                    + '_Dn_' + str(optics_ins.layer_num)
    else:
        eval_path = r'eval/'  # 测试集路径
        # 基础路径
        base_path = r'models/CMA_ZYC' \
                    + '_CHANNEL_' + str(optics_ins.channel) \
                    + '_Dn_' + str(optics_ins.layer_num)

    for index, work_path in enumerate(work_paths):
        phs_code_model = CNNMaxAvgSingle()
        # phs_code_model = FAN()
        # phs_code_model = FAN_cos()
        # 路径参数
        train_path = r'E:/Dataset/Paper_1/' + work_path  # 训练集路径
        # train_path = r'' + work_path  # 训练集路径
        root_path = base_path + '/' + work_path  # 输出路径
        # 如果目录已存在，则删除它
        if Hyperparams.TEST_MODE:
            if os.path.exists(root_path):
                shutil.rmtree(root_path)
            os.makedirs(root_path)
            print('测试模式，已重建路径。')
        elif os.path.exists(root_path) and not Hyperparams.ModelPrepared:
            print('非测试模式，请勿删除路径。')
            sys.exit()  # 停止程序运行
        elif not os.path.exists(root_path):
            os.makedirs(root_path)
        trainer = Trainer(optics_ins=optics_ins, phs_code_model=phs_code_model, root_path=root_path, train_path=train_path, eval_path=eval_path)
        trainer.train()
