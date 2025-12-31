import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type_IAM, generate_type_CVL
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed


def split_data(data_val, wid, max_size=224):
    data_loader = []
    total_len = len(data_val)

    # 按照最大长度拆分数据
    start = 0
    while start < total_len:
        end = min(start + max_size, total_len)  # 计算当前拆分的结束位置

        # 提取当前切片的数据并添加到 data_loader 中
        data_loader.append((
            data_val[start:end],
            wid[start:end],
        ))

        start = end  # 更新起始位置，继续拆分下一段

    return data_loader

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """set device"""
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    load_content = ContentData()


    data_path = cfg.DATA_LOADER.IAM.IMAGE_PATH
    style_path = cfg.DATA_LOADER.IAM.STYLE_PATH
    generate_type = generate_type_IAM
    if cfg.DATA_LOADER.DATASET == "CVL":
        data_path = cfg.DATA_LOADER.CVL.IMAGE_PATH
        style_path = cfg.DATA_LOADER.CVL.STYLE_PATH
        generate_type = generate_type_CVL



    text_corpus = generate_type[opt.generate_type][1]
    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()

    """setup data_loader instances"""

    print(style_path)
    style_dataset = Random_StyleIAMDataset(
        os.path.join(style_path, generate_type[opt.generate_type][0]),
        len(texts)
    )
    print('this process handles characters: ', len(style_dataset))

    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )

    target_dir = os.path.join(opt.save_dir, opt.generate_type)

    diffusion = Diffusion(device=device)

    """build model architecture"""
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

    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=torch.device('cpu')))
        print('Loaded pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('Input the correct checkpoint path')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)  # Freeze vae

    """generate the handwriting datasets"""
    loader_iter = iter(style_loader)

    for x_text in tqdm(texts, position=0, desc='batch_number'):
        print(x_text)
        data = next(loader_iter)
        data_val, wid = data['style'][0], data['wid']

        data_loader = []
        data_loader = split_data(data_val, wid, max_size=50)

        for (data_val, wid) in data_loader:

            style_input = data_val.to(device)

            text_ref = load_content.get_content(x_text)

            # print(wid)

            text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(device)

            # Check if the images already exist for all WIDs
            all_images_exist = True
            for w in wid:

                out_path = os.path.join(target_dir, w[0])
                os.makedirs(out_path, exist_ok=True)
                image_path = os.path.join(out_path, f"{x_text}.png")

                # If any image does not exist, set all_images_exist to False
                if not os.path.exists(image_path):
                    all_images_exist = False
                    break  # No need to check further if one image is missing

            # Skip the generation process if all images already exist
            if all_images_exist:
                # print(f"All images for {x_text} in WIDs {wid} already exist. Skipping image generation.")
                continue

            # Generate the image only if any of the images do not exist
            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(
                    unet, vae, style_input.shape[0], x, style_input, text_ref,
                    opt.sampling_timesteps, opt.eta
                )
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(
                    unet, vae, style_input.shape[0], x, style_input, text_ref
                )
            else:
                raise ValueError('Sample method is not supported')

            # Iterate over sampled images and save them
            for index in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                image = im.convert("L")
                out_path = os.path.join(target_dir, wid[index][0])
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, x_text + ".png"))
                    # print(f"Saved image for {x_text} in {w} at {image_path}")


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='saved_images/IAM',
                        help='Target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm',
                        default='/mnt/ssd4T/Autoregressive/One-Auto_V3/Saved/IAM/debug-20251105_180237/model/650-ckpt.pt',
                        help='Pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True,
                        help='Four generation settings: iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cuda', help='Device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='Choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    opt = parser.parse_args()
    main(opt)
