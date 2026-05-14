from torch import nn
import torch

from .modules.diffusionmodules.model import Encoder, Decoder
from .modules.vqvae.quantize import EMAVectorQuantizer


DDCONFIG_REG = {
    'default': {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 256,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 4,
        "attn_resolutions": [16],
        "dropout": 0.0,
    },
    'small': {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 4,
        "attn_resolutions": [16],
        "dropout": 0.0,
    }
}


class InsightTok(nn.Module):
    def __init__(
        self,
        dd_config_key='default',
        codebook_size=16384,
        codebook_embed_dim=256,
        commit_loss_beta=0.02,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        use_checkpoint=False,
        quant_disable_autocast=False,
        frozen_encoder=False
    ):
        super().__init__()
        assert dd_config_key in DDCONFIG_REG, f"Invalid dd_config_key: {dd_config_key}. Must be one of {list(DDCONFIG_REG.keys())}"
        ddconfig = DDCONFIG_REG[dd_config_key]

        self.quant_disable_autocast = quant_disable_autocast
        self.encoder = Encoder(**ddconfig, gradient_checkpointing=use_checkpoint)
        self.decoder = Decoder(**ddconfig, gradient_checkpointing=use_checkpoint)
        self.quantize = EMAVectorQuantizer(codebook_size, codebook_embed_dim, 
                                                    commit_loss_beta,
                                                    codebook_l2_norm, codebook_show_usage,
                                                    restart_mode='random')

        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(codebook_embed_dim, ddconfig["z_channels"], 1)
        self.frozen_encoder = frozen_encoder

    def encode(self, x):
        if self.frozen_encoder:
            with torch.no_grad():
                h = self.encoder(x)
        else:
            h = self.encoder(x)
        h = self.quant_conv(h)
        if self.quant_disable_autocast:
            with torch.autocast(device_type=h.device.type, enabled=False):
                quant, emb_loss, info = self.quantize(h)
        else:
            quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        if self.quant_disable_autocast:
            with torch.autocast(device_type=code_b.device.type, enabled=False):
                quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        else:
            quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
