import argparse
import os

import numpy as np

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type_IAM, generate_type_CVL, Random_WID_IAMDataset
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
import cv2


def read_and_group_file(file_path):
    # 使用 defaultdict 来存储每个 wid 对应的 label 列表
    wid_dict = defaultdict(list)
    # 打开文件并读取内容
    with open(file_path, 'r') as f:
        for line in f:
            # 分割每行内容，提取 wid 和 label
            parts = line.strip().split(',', 1)  # 只分割第一个逗号
            if len(parts) == 2:
                wid = parts[0]  # 将 wid 转换为整数

                # 提取后续的 label 部分
                label = parts[1].split(' ', 1)[-1]  # 从剩余的部分取最后一个部分作为 label
                wid_dict[wid].append(label)  # 将 label 添加到对应 wid 的列表中

    # 将字典转为一个按 wid 排序的列表
    sorted_wid_label_list = sorted(wid_dict.items())

    return dict(wid_dict)


def process_data(file_path):
    # 创建一个字典来存储结果
    result_dict = defaultdict(lambda: {"name": [], "label": []})
    with open(file_path, 'r') as f:
        for line in f:
            # 按逗号分割每一行，得到 wid 和剩余部分
            parts = line.strip().split(',', 1)

            if len(parts) == 2:
                # 获取 wid 和后面的数据
                wid = parts[0]  # 键
                rest = parts[1]  # 后续部分

                # 再分割后续部分，得到 name 和 label
                rest_parts = rest.split(' ', 1)
                if len(rest_parts) == 2:
                    name = rest_parts[0]
                    label = rest_parts[1]

                    # 将 name 和 label 添加到字典中对应 wid 的列表
                    result_dict[wid]["name"].append(name)
                    result_dict[wid]["label"].append(label)

    return dict(result_dict)


def split_data(data_val, x_text, x_name, max_size=224):
    data_loader = []
    total_len = len(data_val)

    # 按照最大长度拆分数据
    start = 0
    while start < total_len:
        end = min(start + max_size, total_len)  # 计算当前拆分的结束位置

        # 提取当前切片的数据并添加到 data_loader 中
        data_loader.append((
            data_val[start:end],
            x_text[start:end],
            x_name[start:end],
        ))

        start = end  # 更新起始位置，继续拆分下一段

    return data_loader


def main(opt):
    """ Load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """Fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """ Set device """
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(opt.local_rank)

    load_content = ContentData()

    epoch = opt.one_dm.split('/')[-1].split('-')[0]
    epoch_floder = "{}_model".format(epoch)



    """setup data_loader instances"""

    data_path = cfg.DATA_LOADER.IAM.IMAGE_PATH
    style_path = cfg.DATA_LOADER.IAM.STYLE_PATH
    generate_type = generate_type_IAM
    if cfg.DATA_LOADER.DATASET == "CVL":
        data_path = cfg.DATA_LOADER.CVL.IMAGE_PATH
        style_path = cfg.DATA_LOADER.CVL.STYLE_PATH
        generate_type = generate_type_CVL

    text_corpus = generate_type[opt.generate_type][1]

    wid_label_D1 = read_and_group_file(text_corpus)
    wid_label_D2 = process_data(text_corpus)


    print(style_path)
    style_dataset = Random_WID_IAMDataset(
        os.path.join(style_path, generate_type[opt.generate_type][0]),
    )

    print('this process handle characters: ', len(style_dataset))
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                               pin_memory=True
                                               )
    opt.save_dir = os.path.join(opt.save_dir, cfg.DATA_LOADER.DATASET)
    target_dir = os.path.join(opt.save_dir, epoch_floder, opt.generate_type)
    os.makedirs(target_dir, exist_ok=True)

    diffusion = Diffusion(device=device)

    """Build model architecture"""
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM
    ).to(device)

    """Load pretrained one_dm model"""
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location="cpu"))
        print('Loaded pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('Input the correct checkpoint path')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)

    # if opt.generate_type == 'train':
    #     style_loader = train_loader
    # else:
    #     style_loader = test_loader

    """Generate the handwriting datasets"""
    # loader_iter = iter(style_loader)
    # print(texts)
    for data in tqdm(style_loader, position=0, desc='batch_number'):

        data_vals, wid = data['style'][0], data['wid'][0]

        x_text1 = wid_label_D1[wid]
        x_texts = wid_label_D2[wid]['label']
        x_names = wid_label_D2[wid]['name']

        data_loader = split_data(data_vals, x_texts, x_names, max_size=1)

        for (data_val, x_text, x_name) in data_loader:
            # for (data_val, laplace, x_text, x_name) in zip(data_vals, laplaces, x_texts, x_names):
            #     print(x_text[0])
            style_input = data_val.to(device)

            text_ref = load_content.get_content(x_text[0]).to(device)

            # text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(device)
            # print(text_ref.shape)
            # print(style_input.shape)
            # print(x.shape)
            """
            torch.Size([100, 16, 16, 16])
            torch.Size([100, 1, 64, 320])
            torch.Size([100, 4, 8, 64])
            """

            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(
                    unet, vae, style_input.shape[0],
                    x, style_input, text_ref,
                    opt.sampling_timesteps, opt.eta
                )
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(
                    unet, vae, style_input.shape[0],
                    x, style_input, text_ref
                )
            else:
                raise ValueError('Sample method is not supported')

            for index in range(len(ema_sampled_images)):
                # print(x_text[index])
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                image = im.convert("L")
                # print(wid)
                out_path = os.path.join(target_dir, wid)
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, x_name[index] + ".png"))

            # for index in range(len(ema_sampled_images)):
            #     # 1. 从批次中获取单张图片张量，并移除批次维度
            #     #    [1, 3, 64, 192] -> [3, 64, 192]
            #     fake_tensor_chw = ema_sampled_images[index]
            #
            #     # 2. 手动反归一化: 将 [-1, 1] 范围映射到 [0, 255]
            #     image_normalized = fake_tensor_chw.cpu()* 255.0
            #
            #     # 3. [核心改动] 维度换位: 从 [C, H, W] 转换为 [H, W, C]
            #     image_hwc = image_normalized.permute(1, 2, 0).numpy().astype(np.uint8)
            #
            #     # 4. [核心改动] 颜色通道转换: 从 RGB 转换为 BGR
            #     image_bgr = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR)
            #
            #     # 5. 准备路径和文件名
            #     out_path = os.path.join(target_dir, wid)
            #     os.makedirs(out_path, exist_ok=True)
            #     file_path = os.path.join(out_path, x_name[index] + ".png")
            #
            #     # 6. 使用 cv2.imwrite 保存 BGR 格式的图像
            #     cv2.imwrite(file_path, image_bgr)


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='saved_images',
                        help='Target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm',
                        default='Saved/IAM/debug-20251105_180237/model/300-ckpt.pt'
                        , help='Pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', default='fid_test',
                        help='Four generation settings: train, test')
    parser.add_argument('--dataset', type=str, default='IAM', help='IAM for CVL')
    parser.add_argument('--device', type=str, default='cuda', help='Device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='Choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='Device for training')
    opt = parser.parse_args()
    main(opt)
