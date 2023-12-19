from pathlib import Path
import math
import torch
import torch.nn as nn
from timm.layers import DropPath


def get_model_from_cfg(cfg, resume_path=None):
    if cfg.model.arch == "transformer":
        model = RNAModel(cfg)
    else:
        raise NotImplementedError(f"unknown model arch {cfg.model.arch}")

    if resume_path:
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}

        if model.out_features != state_dict["proj_out.weight"].shape[0]:
            print(f"remove proj_out from checkpoint: {resume_path}")
            del state_dict["proj_out.weight"]
            del state_dict["proj_out.bias"]

        model.load_state_dict(state_dict, strict=False)

    return model


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MyEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, norm_first=True, drop_path=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        dim_feedforward = d_model * 4
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Conv1d(d_model, dim_feedforward, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv1d(dim_feedforward, d_model, kernel_size=5, padding=2)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv2d1 = nn.Sequential(
            nn.BatchNorm2d(nhead),
            nn.ReLU(),
            nn.Conv2d(nhead, nhead, kernel_size=7, padding=3),
        )

    def forward(self, src, bpp, src_key_padding_mask=None):
        x = src
        attn_mask = self.conv2d1(bpp)

        if self.norm_first:
            x = x + self.drop_path(self._sa_block(self.norm1(x), attn_mask, src_key_padding_mask))
            x = x + self.drop_path(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop_path(self._sa_block(x, attn_mask, src_key_padding_mask)))
            x = self.norm2(x + self.drop_path(self._ff_block(x)))

        bpp = bpp + attn_mask
        return x, bpp

    def _sa_block(self, x, attn_mask, key_padding_mask):
        attn_mask = attn_mask.reshape(-1, attn_mask.shape[-2], attn_mask.shape[-1])
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = x.transpose(1, 2)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x.transpose(1, 2)
        return self.dropout2(x)


class RNAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.model.hidden_dim
        self.dim = dim
        self.num_layers = cfg.model.num_layers
        self.nhead = cfg.model.nhead
        self.emb = nn.Embedding(5 ** cfg.task.ngram, dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        dpr = [x.item() for x in torch.linspace(0, cfg.model.drop_path_rate, self.num_layers)]
        self.encoders = nn.ModuleList(
            [MyEncoderLayer(d_model=dim, nhead=self.nhead, dropout=0.1, batch_first=True,
                            norm_first=cfg.model.norm_first, drop_path=dpr[i]) for i in range(self.num_layers)])
        self.out_features = 2 if cfg.task.mode == "train" else 4
        self.proj_out = nn.Linear(dim, self.out_features)

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0['seq'][:, :Lmax]
        x = self.emb(x)
        bpp = x0["bpp"][:, :Lmax, :Lmax].unsqueeze(1)
        bpp = bpp.tile(1, self.nhead, 1, 1)

        for i in range(self.num_layers):
            x, bpp = self.encoders[i](x, bpp, src_key_padding_mask=~mask)

        x = self.proj_out(x)
        return x


class EnsembleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if Path(cfg.test.resume_path).is_dir():
            resume_paths = Path(cfg.test.resume_path).glob("*.ckpt")
        else:
            resume_paths = [Path(cfg.test.resume_path)]

        self.models = nn.ModuleList()

        for resume_path in resume_paths:
            model = get_model_from_cfg(cfg, resume_path)
            self.models.append(model)

    def __call__(self, x):
        outputs = [model(x) for model in self.models]
        x = torch.mean(torch.stack(outputs), dim=0)
        return x
