from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Type
from segment_anything.modeling import ImageEncoderViT
from segment_anything.modeling.common import LayerNorm2d


class SamFeatSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        seg_decoder,
        img_size=1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder

    def forward(self,
                x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x) #[B, 256, 64, 64]
        out = self.mask_decoder(image_embedding)

        if out.shape[-1] != original_size:
            out = F.interpolate(
                out,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        return out

    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out


class SegDecoderLinear(nn.Module):
    def __init__(self,
                 num_classes=14,
                 emb_dim=256,
                 activation: Type[nn.Module] = nn.GELU,
                 ):
        super().__init__()

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(emb_dim // 4),
            activation(),
            nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(emb_dim // 8, emb_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 8, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.output_upscaling(x)
        return self.final(out)


class SegDecoderCNN(nn.Module):
    def __init__(self,
                 num_classes=14,
                 embed_dim=256,
                 num_depth=2,
                 top_channel=64,
                 ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(embed_dim, top_channel*2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),

        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2,
                                       stride=2)
                )

            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)

