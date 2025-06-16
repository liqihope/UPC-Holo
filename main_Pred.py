from Trainer import *
import time

if __name__ == '__main__':
    Hyperparams.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Hyperparams.TIME = False
    # Hyperparams.MUL_SAVE = True
    Hyperparams.SAVE = True
    Hyperparams.DEPTH_INVERSION = False

    # TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST
    # 'shibing1', 'shibing4', 'biaohao'
    # 1280*853, 1920*1080, 1071*940
    imgs_name = ['shibing1', 'shibing4', 'biaohao']
    work_paths = ['NYU_4K', '3D_train_xinlei', 'DIV2K_Avg_2']
    layer_num = 4
    CHANNELs = [0, 1, 2]
    Param_ID = '0609'
    for index, work_path in enumerate(work_paths):
        for ind, img_name in enumerate(imgs_name):
            # LCoS_res_h 代表训练尺寸
            img = cv2.imread(r'dataset/' + img_name + '.png')
            LCoS_res_h, LCoS_res_w = img.shape[:2]
            img_path = os.path.join(r'dataset/', f'{img_name}.png')
            depth_path = os.path.join(r'dataset/', f'{img_name}_depth.png')

            img = cv2.imread(img_path)
            LCoS_res_h, LCoS_res_w = img.shape[:2]
            for _, CHANNEL in enumerate(CHANNELs):
                optics_ins = Optics(CHANNEL, 0.005, 0.02/layer_num, layer_num, LCoS_res_h=LCoS_res_h, LCoS_res_w=LCoS_res_w, LCoS_pitch=3.6e-6)

                base_path = r'models/CMA_ZYC_CHANNEL_2_Dn_4'
                
                phs_path = os.path.join('pred', f'{Param_ID}_{img_name}_{work_path}_{CHANNEL}_phase.png')
                rec_path = os.path.join('pred', f'{Param_ID}_{img_name}_{work_path}_{CHANNEL}_rec.png')
                mul_rec_path = os.path.join('mul_pred', f'{Param_ID}_{img_name}_{work_path}_{CHANNEL}_rec.png')
                root_path = os.path.join(base_path, work_path)  # 输出路径
                if os.path.exists(root_path):
                    # 获取指定目录下所有符合模式的.pkl文件路径列表
                    model_file = root_path + '/' + 'model_best' + '.pkl'
                    phs_code_model = torch.load(model_file, map_location="cuda:0", weights_only=False)
                    # print("模型已成功加载")
                else:
                    print("文件不存在，请检查文件路径")
                    continue

                trainer = Trainer(optics_ins=optics_ins, phs_code_model=phs_code_model, root_path=root_path)
                trainer.predict(img_path, depth_path, phs_path, rec_path, mul_rec_path)

            image_B = os.path.join('pred', f'{Param_ID}_{img_name}_{work_path}_0_rec.png')
            image_G = os.path.join('pred', f'{Param_ID}_{img_name}_{work_path}_1_rec.png')
            image_R = os.path.join('pred', f'{Param_ID}_{img_name}_{work_path}_2_rec.png')
            output_path = os.path.join('pred', f'{Param_ID}_{img_name}_{work_path}_rec.png')
            # 读取图像
            img_R = cv2.imread(image_R)
            img_G = cv2.imread(image_G)
            img_B = cv2.imread(image_B)

            # 创建一个新图像，大小和原始图像相同
            rec = np.zeros_like(img_R)

            # 分别将三张图像的RGB通道赋值到新图像的RGB通道
            rec[:, :, 0] = img_B[:, :, 0]  # B通道
            rec[:, :, 1] = img_G[:, :, 1]  # G通道
            rec[:, :, 2] = img_R[:, :, 2]  # R通道

            # 保存新图像
            cv2.imwrite(output_path, rec)
            print(f"Saved new image to: {output_path}")

            psnr_value2 = psnr(img, rec, data_range=255)
            # 计算Y通道分量的均方误差（MSE）
            mse = np.mean((img - rec) ** 2)
            psnr_value3 = 0
            # 计算PSNR（峰值信噪比）
            if mse == 0:
                psnr_score = float('inf')
            else:
                max_pixel_value = 255.0
                psnr_value3 = 20 * np.log10(max_pixel_value / np.sqrt(mse))
            # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
            ssim_value = ssim(img, rec, data_range=255, channel_axis=2)

            print("Test Image {} : PSNR: {:.4f}, SSIM: {:.4f}".format(img_name, psnr_value3, ssim_value))

            if Hyperparams.MUL_SAVE:
                for dep in range(optics_ins.layer_num):
                    image_B = os.path.join('mul_pred', f'{Param_ID}_{img_name}_{work_path}_0_rec{dep}.png')
                    image_G = os.path.join('mul_pred', f'{Param_ID}_{img_name}_{work_path}_1_rec{dep}.png')
                    image_R = os.path.join('mul_pred', f'{Param_ID}_{img_name}_{work_path}_2_rec{dep}.png')
                    output_path = os.path.join('mul_pred', f'{Param_ID}_{img_name}_{work_path}_rec{dep}.png')
                    # 读取图像
                    img_R = cv2.imread(image_R)
                    img_G = cv2.imread(image_G)
                    img_B = cv2.imread(image_B)

                    # 创建一个新图像，大小和原始图像相同
                    rec = np.zeros_like(img_R)

                    # 分别将三张图像的RGB通道赋值到新图像的RGB通道
                    rec[:, :, 0] = img_B[:, :, 0]  # B通道
                    rec[:, :, 1] = img_G[:, :, 1]  # G通道
                    rec[:, :, 2] = img_R[:, :, 2]  # R通道

                    # 保存新图像
                    cv2.imwrite(output_path, rec)
                    print(f"Saved new image to: {output_path}")

    if Hyperparams.TIME:
        print(np.array(Hyperparams.time1))
        print(np.array(Hyperparams.time2))
        print(np.array(Hyperparams.time1) + np.array(Hyperparams.time2))
