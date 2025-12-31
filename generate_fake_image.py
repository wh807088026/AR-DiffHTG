import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type_IAM, generate_type_CVL, Some_StyleIAMDataset
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed
from datetime import datetime

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

    data_path = cfg.DATA_LOADER.IAM.IMAGE_PATH
    style_path = cfg.DATA_LOADER.IAM.STYLE_PATH
    generate_type = generate_type_IAM
    if cfg.DATA_LOADER.DATASET == "CVL":
        data_path = cfg.DATA_LOADER.CVL.IMAGE_PATH
        style_path = cfg.DATA_LOADER.CVL.STYLE_PATH
        generate_type = generate_type_CVL

    text_corpus = generate_type[opt.generate_type][1]
    with open(text_corpus, 'r') as _f:
        line = _f.readline().strip()  # 读取整行并去掉首尾空白
        texts = line.split()

    new_path = '/mnt/ssd4T/FID_KID_HWD_GS/IAM64-new/style_samples_new'

    """Setup data_loader instances"""

    style_dataset = Some_StyleIAMDataset(
        new_path,
        len(texts)
    )

    print('Handling characters: ', len(style_dataset))
    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )
    # return
    time_stamp = datetime.now().strftime("%Y%m%d") # e.g. "20251210_154530"
    target_dir = os.path.join(opt.save_dir, epoch_floder, opt.generate_type, line)

    # 如果目录不存在，就创建
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

    """Generate the handwriting datasets"""
    loader_iter = iter(style_loader)
    for x_text in tqdm(texts, position=0, desc='batch_number'):
        data = next(loader_iter)
        data_val, wid = data['style'][0], data['wid']

        data_loader = []
        if len(data_val) > 224:
            data_loader.append((data_val[:224], wid[:224]))
            data_loader.append((data_val[224:], wid[224:]))
        else:
            data_loader.append((data_val, wid))

        for (data_val, wid) in data_loader:
            style_input = data_val.to(device)
            text_ref = load_content.get_content(x_text)
            text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(device)
            # print(style_input.size())
            if opt.sample_method == 'ddim':
                # ema_sampled_images = diffusion.ddim_sample(
                #     unet, vae, style_input.shape[0],
                #     x, style_input, text_ref,
                #     opt.sampling_timesteps, opt.eta
                # )
                ema_sampled_images = diffusion.ddim_sample(
                    unet, vae, style_input.shape[0],
                    x, style_input, text_ref,
                    opt.sampling_timesteps, opt.eta,
                    save_intermediates=False,
                    save_dir=target_dir,
                )
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(
                    unet, vae, style_input.shape[0],
                    x, style_input, text_ref
                )
            else:
                raise ValueError('Sample method is not supported')

            for index in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                image = im.convert("L")
                # print(wid)
                out_path = os.path.join(target_dir, wid[index][0])
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, x_text + ".png"))

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='saved_images/IAM', help='Target dir for storing the '
                                                                                   'generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='Saved/IAM/debug-20251105_180237/model/650-ckpt.pt', help='Pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', default="test_fake", help='Four generation settings: iv_s, '
                                                                                     'iv_u, oov_s, oov_u, test_fake')
    parser.add_argument('--device', type=str, default='cuda', help='Device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='Choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='Device for training')
    opt = parser.parse_args()
    main(opt)
