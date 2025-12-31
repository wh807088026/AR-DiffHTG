import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader import IAMDataset
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from data_loader.loader import letters
from models.recognition import HTRNet
from diffusers import AutoencoderKL
# from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from models.loss import SupConLoss
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
"""
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py \
    --feat_model model_zoo/RN18_class_10400.pth \
    --log English
"""


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """

    if not opt.test:
        print("Running in TRAINING mode, creating log directories...")
        logs = set_log(cfg.OUTPUT_DIR, cfg.DATA_LOADER.DATASET, opt.log_name)
    else:
        logs = {'tboard': None, 'model': None, 'sample': None}

    """ set mulit-gpu """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ set dataset"""

    data_path = cfg.DATA_LOADER.IAM.IMAGE_PATH
    style_path = cfg.DATA_LOADER.IAM.STYLE_PATH

    if cfg.DATA_LOADER.DATASET == "CVL":
        data_path = cfg.DATA_LOADER.CVL.IMAGE_PATH
        style_path = cfg.DATA_LOADER.CVL.STYLE_PATH
    train_dataset = IAMDataset(
        data_path, style_path, cfg.DATA_LOADER.DATASET, cfg.TRAIN.TYPE)
    print('{} number of training images: {}'.format(cfg.DATA_LOADER.DATASET, len(train_dataset)))
    if opt.test:
        cfg.TRAIN.IMS_PER_BATCH = 8
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,  # 单卡时需启用打乱
        drop_last=False,
        collate_fn=train_dataset.collate_fn_,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )

    test_dataset = IAMDataset(
        data_path, style_path, cfg.DATA_LOADER.DATASET, cfg.TEST.TYPE)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,  # 测试集不打乱
        drop_last=False,
        collate_fn=test_dataset.collate_fn_,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM,
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                     attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS,
                     context_dim=cfg.MODEL.EMB_DIM).to(device)

    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cuda')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))

    """load pretrained resnet18 model"""

    """Initialize the U-Net model for parallel training on multiple GPUs"""
    unet = unet.to(device)

    """build criterion and optimizer"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)
    vae = AutoencoderKL.from_pretrained("/mnt/ssd4T/All-defffuser/pk/stable-diffusion-v1-5", subfolder="vae")
    # vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    """Freeze vae and text_encoder"""
    vae.requires_grad_(False)
    vae = vae.to(device)


    '''load pretrained ocr model'''
    if opt.ocr:
        ctc_loss = nn.CTCLoss()
        ocr_model = HTRNet(nclasses=len(letters), vae=True)

        if len(opt.ocr_model) > 0:
            print(f"Loading pretrained OCR model from {opt.ocr_model} ...")
            checkpoint = torch.load(opt.ocr_model, map_location=torch.device('cpu'))

            # 当前模型参数
            model_dict = ocr_model.state_dict()

            # 过滤掉形状不匹配的层（例如输出层）
            filtered_state_dict = {k: v for k, v in checkpoint.items()
                                   if k in model_dict and v.shape == model_dict[k].shape}

            # 加载过滤后的参数
            missing_keys, unexpected_keys = ocr_model.load_state_dict(filtered_state_dict, strict=False)

            print(f"Loaded {len(filtered_state_dict)}/{len(model_dict)} layers successfully.")
            if missing_keys:
                print(f"⚠️ Missing keys (not loaded): {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
        else:
            print('❌ Failed to load the pretrained OCR model: empty path.')
            exit()

        ocr_model.requires_grad_(False)
        ocr_model = ocr_model.to(device)

        trainer = Trainer(
            diffusion, unet, vae, criterion, optimizer,
            train_loader, logs, test_loader, device,
            ocr_model, ctc_loss
        )

    else:
        trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)

    if opt.test:
        # trainer.test()
        trainer.test(num_iters=10)
    else:
        trainer.train()


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64_scratch.yml',
                        help='Config file for training (and optionally testing)')

    parser.add_argument('--one_dm', dest='one_dm', default='', help='pre-trained one_dm model')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument('--ocr_model', dest='ocr_model', default='./model_zoo/vae_HTR138.pth',
                        help='pre-trained ocr model')
    parser.add_argument("--ocr", action='store_true', default=False)
    parser.add_argument("--wandb", action='store_true', default=True)
    parser.add_argument("--testFlops", action='store_true', default=False)
    opt = parser.parse_args()
    main(opt)