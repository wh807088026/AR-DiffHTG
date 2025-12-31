import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
import os
import sys
from PIL import Image
# from torch.amp import GradScaler, autocast
import torchvision
from tqdm import tqdm
from data_loader.loader import ContentData
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader,
                 logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        self.model = unet
        self.diffusion = diffusion
        self.vae = vae
        self.recon_criterion = criterion['recon']
        self.nce_criterion = criterion['nce']
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.ocr_model = ocr_model
        self.ctc_criterion = ctc_loss
        self.device = device
        # GradScaler 已在您原始代码的 __init__ 中初始化，无需改动
        # self.scaler = GradScaler()

    def _train_iter(self, data, step, pbar):
        """
        [已修改] 标准训练迭代，开启 AMP。
        """
        self.model.train()
        images, style_ref, content_ref, wid = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device)

        # vae encode 在 autocast 之外，因为它可能对 float16 不稳定
        images = self.vae.encode(images).latent_dist.sample()
        images = images * 0.18215

        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)

        self.optimizer.zero_grad()

        # --- 步骤 1: 使用 autocast 包裹前向传播和损失计算 ---
        # with autocast(device_type='cuda'):
        predicted_noise, low_nce_emb = self.model(x_t, t, style_ref, content_ref,
                                                  tag='train')
        recon_loss = self.recon_criterion(predicted_noise, noise)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + low_nce_loss

        # --- 步骤 2 & 3: 使用 scaler 进行反向传播和优化器更新 ---
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (如果需要)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # log loss
        loss_dict = {"reconstruct_loss": recon_loss.item()}
        self.tb_summary.add_scalars("loss", loss_dict, step)
        self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _finetune_iter(self, data, step, pbar):
        self.model.train()
        images, style_ref, content_ref, wid, target, target_lengths = \
            data['img'].to(self.device), data['style'].to(self.device), \
                data['content'].to(self.device), data['wid'].to(self.device), \
                data['target'].to(self.device), data['target_lengths'].to(self.device)

        latent_images = self.vae.encode(images).latent_dist.sample() * 0.18215
        t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True).to(self.device)
        x_t, noise = self.diffusion.noise_images(latent_images, t)

        self.optimizer.zero_grad()

        # --- diffusion 主训练路径 ---
        predicted_noise, low_nce_emb = self.model(x_t, t, style_ref, content_ref, tag='train')
        recon_loss = self.recon_criterion(predicted_noise, noise)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)

        # --- OCR 结构约束部分 ---
        with torch.no_grad():
            # 用少步采样近似生成结果，不参与反传
            x_start, _, _ = self.diffusion.train_ddim(
                self.model, x_t, style_ref, content_ref, t, sampling_timesteps=2
            )
            x_small = F.interpolate(x_start, scale_factor=0.5, mode='bilinear', align_corners=False)
            rec_out = self.ocr_model(x_small)
            input_lengths = torch.IntTensor(x_start.shape[0] * [rec_out.shape[0]]).to(self.device)
            ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target, input_lengths, target_lengths)

        loss = recon_loss + low_nce_loss + 0.1 * ctc_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        self.tb_summary.add_scalars("loss", {
            "reconstruct_loss": recon_loss.item(),
            "low_nce_loss": low_nce_loss.item(),
            "ctc_loss": ctc_loss.item()
        }, step)

        self._progress(recon_loss.item(), pbar)
        del data, loss
        torch.cuda.empty_cache()

    #
    # ... 您其他的 _save_images, _valid_iter, train, test 等方法无需任何改动 ...
    #
    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save(path)
        return im

    @torch.no_grad()
    def _valid_iter(self, epoch):
        print('loading test dataset, the number is', len(self.valid_data_loader))
        self.model.eval()
        # use the first batch of dataloader in all validations for better visualization comparisons
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)
        # prepare input
        images, style_ref, content_ref = test_data['img'].to(self.device), \
            test_data['style'].to(self.device), \
            test_data['content'].to(self.device)

        load_content = ContentData()
        # forward
        texts = ['getting', 'both', 'success']
        for text in texts:
            text_ref = load_content.get_content(text)
            text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(
                self.device)
            preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref,
                                               text_ref)
            out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}.png")
            self._save_images(preds, out_path)

    def train(self):
        """start training iterations"""
        for epoch in range(cfg.SOLVER.EPOCHS):
            print(f"Epoch: {epoch}")
            pbar = tqdm(self.data_loader, leave=False)

            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step + 1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        self._save_checkpoint(total_step)

                    if self.valid_data_loader is not None:
                        if (total_step + 1) > cfg.TRAIN.VALIDATE_BEGIN and (
                                total_step + 1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                            self._valid_iter(total_step)
                else:
                    self._train_iter(data, total_step, pbar)

            if epoch % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self._save_checkpoint(epoch)

            if self.valid_data_loader is not None:
                if epoch % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(epoch)

            pbar.close()

    # trainer.py (仅展示优化后的 test 方法)

    # trainer.py

    # ... Trainer 类的其他方法 ...

    def test(self, num_iters=None):
        """
        [最终优化版] 在训练集上执行指定数量或一整个周期的训练迭代测试。

        Args:
            num_iters (int, optional): 要执行的测试迭代次数。
                                       如果为 None 或不传入，则会遍历整个训练集数据加载器，
                                       运行一个完整的周期。默认为 None。
        """
        # 1. 决定迭代次数
        num_iters_to_run = 0
        if num_iters is None:
            num_iters_to_run = len(self.data_loader)
            print(f"INFO: `num_iters` not specified. Running a full epoch test for all {num_iters_to_run} batches.")
        else:
            num_iters_to_run = min(num_iters, len(self.data_loader))
            print(f"INFO: Running a test for {num_iters_to_run} batches on the training set.")

        if num_iters_to_run <= 0:
            print("INFO: Nothing to test. Exiting.")
            return

        # 2. 确保模型处于训练模式
        self.model.train()

        # 3. 创建进度条和数据迭代器
        pbar = tqdm(total=num_iters_to_run, desc=f"Testing {num_iters_to_run} iterations")
        data_iterator = iter(self.data_loader)

        try:
            # 4. 循环指定次数
            for i in range(num_iters_to_run):
                data = next(data_iterator)

                # 5. 调用核心训练迭代函数
                if self.ocr_model is not None:
                    self._finetune_iter(data, step=i, pbar=pbar)
                else:
                    self._train_iter(data, step=i, pbar=pbar)

                pbar.update(1)

            print("\n" + "=" * 50)
            print(f"Test PASSED for {i + 1} batches!")
            print("Successfully completed: forward, backward, and optimizer steps.")
            print("=" * 50)

        except Exception as e:
            print(f"\n" + "=" * 50)
            print(f"Test FAILED at iteration {i + 1} with an error.")
            print("=" * 50)
            raise e
        finally:
            pbar.close()

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch):
        save_path = os.path.join(self.save_model_dir, f"{epoch}-ckpt.pt")
        torch.save(self.model.state_dict(), save_path)
