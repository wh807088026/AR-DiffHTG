
import argparse
import os

import torch
import torchvision
from PIL import Image
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 文件名: fid_generate.py
from data_loader.loader import FidGenerationDataset, fid_collate_fn, ContentData, generate_type
from models.diffusion import Diffusion
from models.unet import UNetModel
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed
from functools import partial

# from torch.utils.data.distributed import DistributedSampler

def main(opt):
    """ 加载配置文件 """
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """ 固定随机种子 """
    fix_seed(cfg.TRAIN.SEED)

    """ 设置设备 """
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # --- 1. 初始化模型 ---
    print("INFO: Building models...")
    diffusion = Diffusion(device=device)
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

    if os.path.exists(opt.one_dm):
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location="cpu"))
        print('INFO: Loaded pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError(f'Checkpoint path not found: {opt.one_dm}')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)

    # --- 2. 使用新的 FidGenerationDataset 和 DataLoader ---
    print("INFO: Initializing data loader...")
    load_content = ContentData()

    # 定义一个图像变换 (高度固定为64，宽度自适应)
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.resize((int(img.width * 64 / img.height), 64), Image.BICUBIC)),
        torchvision.transforms.ToTensor(),
        # [核心改动] 将均值和标准差都改为单个值的列表，以匹配单通道
        torchvision.transforms.Normalize([0.5], [0.5]),
    ])

    text_corpus_path = generate_type[opt.generate_type][1]
    image_root_path = os.path.join(cfg.DATA_LOADER.IAMGE_PATH, generate_type[opt.generate_type][0])

    dataset = FidGenerationDataset(
        image_path=image_root_path,
        text_path=text_corpus_path,
        content_loader=load_content,
        transform=img_transform
    )
    collate_with_args = partial(fid_collate_fn, content_loader=load_content, transform=img_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_with_args, # <-- 使用这个绑定了参数的新函数
        pin_memory=True
    )

    # --- 3. 准备输出目录 ---
    epoch = os.path.basename(opt.one_dm).split('-')[0]
    target_dir = os.path.join(opt.save_dir, f"{epoch}_model_fid", opt.generate_type)
    os.makedirs(target_dir, exist_ok=True)

    # --- 4. 执行高效的批处理生成循环 ---
    print(f"INFO: Starting batch generation with batch s16ize {opt.batch_size}...")
    for batch in tqdm(data_loader, desc='Generating Batches'):
        with torch.no_grad():
            style_input = batch['style'].to(device)
            text_ref = batch['content'].to(device)
            batch_size = style_input.shape[0]
            content_lengths = batch['content_lengths']

            # 准备初始噪声，宽度与内容序列长度相关
            # 注意: 这个尺寸可能需要根据您的U-Net结构进行微调
            latent_height = style_input.shape[3] // 8
            latent_width = text_ref.shape[1] * 2
            x = torch.randn((batch_size, 4, latent_height, latent_width)).to(device)

            # 批量执行 DDIM 采样
            ema_sampled_images = diffusion.ddim_sample(
                unet, vae, batch_size,
                x, style_input, text_ref,
                opt.sampling_timesteps, opt.eta
            )

            # 逐个裁剪和保存批次中的结果
            for i in range(batch_size):
                full_image = torchvision.transforms.ToPILImage()(ema_sampled_images[i].clamp(-1, 1) / 2. + 0.5)

                # 根据内容文本长度和每个字符32像素的宽度，进行精确裁剪
                target_width = content_lengths[i] * 32
                height = full_image.height

                final_width = min(target_width, full_image.width)
                cropped_image = full_image.crop((0, 0, final_width, height))

                image = cropped_image.convert("L")

                wid = batch['wid'][i]
                output_name = batch['output_name'][i]

                out_path = os.path.join(target_dir, wid)
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, output_name))


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml', help='Config file')
    parser.add_argument('--dir', dest='save_dir', default='saved_images/IAM', help='Target dir for storing images')
    parser.add_argument('--one_dm', dest='one_dm', default='Saved/IAM64_scratch/debug-20250905_104347/model/799-ckpt.pt', help='Pre-trained model checkpoint')
    parser.add_argument('--generate_type', dest='generate_type', default='fid_test', help='Generation setting')
    parser.add_argument('--device', type=str, default='cuda', help='Device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Path to Stable Diffusion VAE')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    opt = parser.parse_args()
    main(opt)