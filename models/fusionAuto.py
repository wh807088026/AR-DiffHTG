# fusionAuto.py (自回归图像序列生成版本)

import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from typing import Tuple
from models.transformer import *
from torch.cuda.amp import autocast
from torch.nn import functional as F
from models.resnet_dilation import resnet18 as resnet18_dilation

# ==============================================================================
# 模块 1: 核心 Transformer 构建块 (与之前版本相同)
# ==============================================================================
# 为了代码完整性，这里保留了 TransformerBlock 及其依赖项
# 您可以从之前的文件中复制 SelfAttention_RoPE, FFN, precompute_freqs_cis

def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


class SelfAttention_RoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def _apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1, 2)
        x_out = torch.stack([
            x_[..., 0] * freqs_cis[..., 0] - x_[..., 1] * freqs_cis[..., 1],
            x_[..., 1] * freqs_cis[..., 0] + x_[..., 0] * freqs_cis[..., 1],
        ], -1)
        return x_out.flatten(-2).type_as(x)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim) for t in (q, k, v)]
        q, k = self._apply_rope(q, freqs_cis), self._apply_rope(k, freqs_cis)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v
        return self.proj(out.permute(0, 2, 1, 3).contiguous().view(B, L, C))


class FFN(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention_RoPE(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, mlp_ratio)
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 4 * embed_dim))

    def _modulate(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: torch.Tensor, style_vector: torch.Tensor, freqs_cis: torch.Tensor,
                attn_bias: torch.Tensor) -> torch.Tensor:
        mod_params = self.ada_lin(style_vector)
        scale1, shift1, scale2, shift2 = mod_params.chunk(4, dim=1)
        x = x + self.attn(self._modulate(self.norm1(x), scale1.unsqueeze(1), shift1.unsqueeze(1)), freqs_cis, attn_bias)
        x = x + self.ffn(self._modulate(self.norm2(x), scale2.unsqueeze(1), shift2.unsqueeze(1)))
        return x


class StyleEncoder(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, hidden_dim_ratio: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_chans, embed_dim * hidden_dim_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * hidden_dim_ratio, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(x).flatten(1))


# ==============================================================================
# 模块 2: 全新的手写字生成模型 (Autoregressive Handwriting Generator)
# ==============================================================================


class StyleProcessor(nn.Module):
    """
    [新] 强大的风格处理器。
    它借鉴了您原始代码的设计，使用 ResNet + Transformer Encoder 来提取丰富的风格特征，
    然后再将其提炼为全局风格向量。
    """

    def __init__(self, in_chans: int, embed_dim: int, nhead: int, num_encoder_layers: int):
        super().__init__()
        # 1. ResNet 主干网络，用于初步提取特征图
        self.feature_extractor = self.initialize_resnet_feature_extractor()

        self.channel_adapter = nn.Conv2d(512, 256, kernel_size=1)

        # 2. 空洞卷积层，用于扩大感受野
        self.dilation_layer = resnet18_dilation().conv5_x

        # 3. Transformer 编码器，用于深度处理风格特征序列
        encoder_layer = TransformerEncoderLayer(in_chans, nhead, in_chans * 4, 0.1, "relu", True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(in_chans))
        self.pos_embed = PositionalEncoding(dim=in_chans, dropout=0.1)

        # 4. 最终的全局风格向量提炼器
        self.final_encoder = StyleEncoder(in_chans=in_chans, embed_dim=embed_dim)

    def initialize_resnet_feature_extractor(self):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # 返回一个纯粹的特征图提取器，下采样率为 16x，输出通道为 512
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

    def forward(self, style_img: torch.Tensor) -> torch.Tensor:
        # a. 提取 2D 特征图: [B, 1, H_in, W_in] -> [B, 512, H_out, W_out]
        style_feat_map = self.feature_extractor(style_img)

        # b. [新增] 通过 1x1 卷积适配通道数: -> [B, 256, H_out, W_out]
        style_feat_map_adapted = self.channel_adapter(style_feat_map)

        # c. 通过空洞卷积层: -> [B, 512, H_out, W_out] (注意尺寸可能减半)
        style_feat_map_dilated = self.dilation_layer(style_feat_map_adapted)

        # d. 转换为序列并添加位置编码
        style_seq = rearrange(style_feat_map_dilated, 'b c h w -> (h w) b c')
        style_seq = self.pos_embed(style_seq)

        # e. 通过 Transformer Encoder 深度建模
        style_seq_encoded = self.transformer_encoder(style_seq)

        # f. 将处理后的序列转回 2D 特征图
        H, W = style_feat_map_dilated.shape[2:]
        style_feat_map_encoded = rearrange(style_seq_encoded, '(h w) b c -> b c h w', h=H, w=W)

        # g. 提炼为单一的全局风格向量
        style_vector = self.final_encoder(style_feat_map_encoded)
        return style_vector


class CharacterImageEncoder(nn.Module):
    """
    [新] 字符图像编码器。
    将一个 [B, N, 16, 16] 的字符图像序列，编码为 [B, N, D_model] 的特征序列。
    """

    def __init__(self, in_chans=1, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1),  # -> [B*N, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> [B*N, 128, 4, 4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> [B*N, 256, 2, 2]
            nn.ReLU(),
            # MODIFIED: 将 start_dim 从 2 改为 1
            # 这会将 [B*N, 256, 2, 2] 展平为 [B*N, 256*2*2] = [B*N, 1024]
            nn.Flatten(start_dim=1),
            # 现在 Linear 层的输入维度 (1024) 与展平后的特征维度完全匹配
            nn.Linear(256 * 2 * 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H, W = x.shape
        # 假设输入是灰度图，增加一个通道维度
        x = x.unsqueeze(2)  # -> [B, N, 1, H, W]
        x = x.view(B * N, 1, H, W)  # -> [B*N, 1, 16, 16]

        encoded = self.encoder(x)  # -> [B*N, output_dim]
        return encoded.view(B, N, -1)  # -> [B, N, output_dim]


class AutoregressiveHandwritingGenerator(nn.Module):
    """
    [最终方案] 风格条件的自回归手写字生成器
    """

    def __init__(self,
                 style_in_chans: int = 512,
                 content_char_dim: int = 512,
                 depth: int = 12,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 output_dim: int = 512,
                 contrastive_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # --- 1. 风格编码器 ---
        # 负责将 [B, 1, 64, W] 的风格图像编码为 [B, embed_dim] 的全局风格向量
        # (假设有一个预定义的 ResNet 特征提取器 self.style_feature_extractor)
        # self.style_feature_extractor = self.initialize_resnet_feature_extractor()
        # self.style_encoder = StyleEncoder(in_chans=style_in_chans, embed_dim=embed_dim)
        self.style_processor = StyleProcessor(in_chans=style_in_chans, embed_dim=embed_dim, nhead=8,
                                              num_encoder_layers=3)
        # --- 2. 内容编码器 ---
        # 负责将 [B, N, 16, 16] 的字符图像序列编码为 [B, N, content_char_dim]
        self.character_encoder = CharacterImageEncoder(in_chans=1, output_dim=content_char_dim)

        # --- 3. 自回归 Transformer 主体 ---
        self.content_proj_in = nn.Linear(content_char_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))  # 假设最大序列长度 4096
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])

        # --- 4. 输出处理 ---
        self.norm_out = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # --- 5. 对比学习投影头 ---
        self.contrastive_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, contrastive_dim)
        )
        self._initialize_weights()

        # --- 5. 字符级显式对齐层 ---
        self.char_queries = nn.Parameter(torch.randn(1, 4096, embed_dim))
        self.char_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        torch.nn.init.normal_(self.pos_embed, std=.02)

    def initialize_resnet_feature_extractor(self):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

    def forward(self, style: torch.Tensor, content: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            style (torch.Tensor): 风格图像对。Shape: `[B, 2, 64, W]`
            content (torch.Tensor): 内容字符图像序列。Shape: `[B, N, 16, 16]`
            anchor_style_feat: torch.Size([8, 512, 2, W])
            pos_style_feat: torch.Size([8, 512, 2, W])
            anchor_style_vector: torch.Size([8, 768])
            pos_style_vector: torch.Size([8, 768])
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - style_hs (torch.Tensor): 融合后的特征序列。Shape: `[B, N, 512]`
            - low_nce_emb (torch.Tensor): 用于对比学习的嵌入。Shape: `[B, 2, 256]`
        """
        B, N, H, W = content.shape

        # --- 1. 处理风格，生成全局风格向量和对比学习对 ---
        anchor_style_img = style[:, 0, :, :].clone().unsqueeze(1)  # -> [B, 1, 64, W]
        pos_style_img = style[:, 1, :, :].clone().unsqueeze(1)  # -> [B, 1, 64, W]


        anchor_style_vector = self.style_processor(anchor_style_img)
        pos_style_vector = self.style_processor(pos_style_img)
        # 计算对比学习嵌入
        low_nce_emb = torch.stack([
            self.contrastive_mlp(anchor_style_vector),
            self.contrastive_mlp(pos_style_vector)
        ], dim=1)  # -> [B, 2, 256]
        low_nce_emb = F.normalize(low_nce_emb, p=2, dim=2)

        # --- 2. 处理内容，生成内容特征序列 ---
        content_seq = self.character_encoder(content)  # -> [B, N, 512]
        content_seq = self.content_proj_in(content_seq)  # -> [B, N, 768]
        content_seq = content_seq + self.pos_embed[:, :N]

        # --- 3. 自回归处理 ---
        # 准备 RoPE 和因果掩码
        seq_len = content_seq.shape[1]
        freqs_cis = precompute_freqs_cis(self.embed_dim // self.num_heads, seq_len).to(content_seq.device)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=content_seq.device), diagonal=1)

        # 逐层通过 Transformer，注入风格
        x = content_seq
        for block in self.blocks:
            x = block(x, style_vector=anchor_style_vector, freqs_cis=freqs_cis, attn_bias=causal_mask)

        char_queries = self.char_queries[:, :N].expand(B, N, -1)  # [B, N, 768]
        aligned_chars, attn_weights = self.char_attn(
            query=char_queries, key=x, value=x
        )  # aligned_chars: [B, N, 768]
        x = x + aligned_chars

        # --- 4. 最终输出 ---
        style_hs = self.output_proj(self.norm_out(x))  # -> [B, N, 512]

        return style_hs, low_nce_emb

    def generate(self, style: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """
        [修正版] 推理阶段的生成函数。

        Args:
            style (torch.Tensor): 风格图像对。Shape: `[B, 2, 64, W]`
            content (torch.Tensor): 内容字符图像序列。Shape: `[B, N, 16, 16]`

        Returns:
            torch.Tensor: 融合后的特征序列。Shape: `[B, N, 512]`
        """
        # 1. 准备风格和内容输入
        # 在推理时，我们只使用第一张图（锚点）作为风格参考
        anchor_style_img = style[:, 0, :, :].clone().unsqueeze(1)
        B, N, H, W = content.shape

        # --- 2. [核心改动] 使用 self.style_processor 提取风格向量 ---
        #    不再调用不存在的 self.style_feature_extractor
        anchor_style_vector = self.style_processor(anchor_style_img)

        # --- 3. 处理内容，生成内容特征序列 (与 forward 方法一致) ---
        content_seq = self.character_encoder(content)
        content_seq = self.content_proj_in(content_seq)
        content_seq = content_seq + self.pos_embed[:, :N]

        # --- 4. 自回归处理 (与 forward 方法一致) ---
        seq_len = content_seq.shape[1]
        freqs_cis = precompute_freqs_cis(self.embed_dim // self.num_heads, seq_len).to(content_seq.device)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=content_seq.device), diagonal=1)

        x = content_seq
        for block in self.blocks:
            x = block(x, style_vector=anchor_style_vector, freqs_cis=freqs_cis, attn_bias=causal_mask)

        char_queries = self.char_queries[:, :N].expand(B, N, -1)
        aligned_chars, _ = self.char_attn(query=char_queries, key=x, value=x)
        x = x + aligned_chars
        # --- 5. 最终输出 ---
        style_hs = self.output_proj(self.norm_out(x))

        return style_hs