import os
import urllib.request

import torch

from common import HERE, MEAN, STD
from model import UNet

# python make_init_weights.py   -> init/carvana_unet.pth and init/vgg13bn_encoder.pth: pretrained starting weights
# for train.py --init (allowed by the TA on the forum, 10/4). Both are plain state dicts of the official UNet.
#   carvana_unet.pth   the whole UNet from milesial/Pytorch-UNet trained on Carvana (GPL-3.0), same design as
#                      model.py; 2-class head turned into 1 logit, input scale changed to ImageNet normalisation
#   vgg13bn_encoder.pth  torchvision VGG13-BN (ImageNet) in inc, down1, down2, down3; down4 and decoder random (seed 0)

OUT = os.path.join(HERE, "init")
CARVANA_URL = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth"
MEAN_T, STD_T = torch.tensor(MEAN), torch.tensor(STD)


def carvana_to_official(path):
    """milesial Carvana UNet (2 classes, input in [0, 1]) -> state_dict of the official UNet."""
    src = torch.load(path, map_location="cpu")
    dst = UNet(in_channels=3, out_channels=1).state_dict()
    body = [k for k in src if not k.startswith("outc")]
    out = {d: src[s].clone() for s, d in zip(body, [k for k in dst if not k.startswith("outc")])}
    # two classes -> one logit: p(pet) = sigmoid(z1 - z0)
    out["outc.weight"] = (src["outc.conv.weight"][1] - src["outc.conv.weight"][0])[None]
    out["outc.bias"] = (src["outc.conv.bias"][1] - src["outc.conv.bias"][0])[None]
    # Carvana saw x in [0, 1]; the official pipeline gives (x - mean) / std: undo it inside the first conv
    w = out["inc.net.0.weight"]  # (64, 3, 3, 3)
    out["inc.net.1.running_mean"] -= (w * MEAN_T.view(1, 3, 1, 1)).sum(dim=(1, 2, 3))
    out["inc.net.0.weight"] = w * STD_T.view(1, 3, 1, 1)
    return out


def vgg13bn_encoder_into(unet_sd, vgg):
    """Copy VGG13-BN stages 1-4 (8 conv + BN) into inc, down1, down2, down3."""
    convs = [m for m in vgg.features if isinstance(m, torch.nn.Conv2d)][:8]
    bns = [m for m in vgg.features if isinstance(m, torch.nn.BatchNorm2d)][:8]
    prefixes = ["inc.net", "down1.net.1.net", "down2.net.1.net", "down3.net.1.net"]
    for i, (conv, bn) in enumerate(zip(convs, bns)):
        p = f"{prefixes[i // 2]}.{0 if i % 2 == 0 else 3}"  # conv inside DoubleConv
        q = f"{prefixes[i // 2]}.{1 if i % 2 == 0 else 4}"  # its BatchNorm
        unet_sd[f"{p}.weight"] = conv.weight.detach().clone()
        unet_sd[f"{q}.weight"] = bn.weight.detach().clone()
        unet_sd[f"{q}.bias"] = bn.bias.detach().clone()
        unet_sd[f"{q}.running_var"] = bn.running_var.clone()
        # no conv bias in the UNet: a BatchNorm follows and removes any constant, so move it into the running mean
        unet_sd[f"{q}.running_mean"] = bn.running_mean.clone() - conv.bias.detach()
    return unet_sd


def make_carvana():
    path = os.path.join(OUT, "carvana_unet.pth")
    if not os.path.exists(path):
        raw = os.path.join(OUT, os.path.basename(CARVANA_URL))
        if not os.path.exists(raw):
            urllib.request.urlretrieve(CARVANA_URL, raw)
        sd = carvana_to_official(raw)
        UNet(in_channels=3, out_channels=1).load_state_dict(sd, strict=True)  # same check as the TA's loader
        torch.save(sd, path)
    return path


def make_vgg13bn():
    from torchvision.models import VGG13_BN_Weights, vgg13_bn

    path = os.path.join(OUT, "vgg13bn_encoder.pth")
    if not os.path.exists(path):
        torch.manual_seed(0)
        sd = vgg13bn_encoder_into(UNet(in_channels=3, out_channels=1).state_dict(),
                                  vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1))
        UNet(in_channels=3, out_channels=1).load_state_dict(sd, strict=True)
        torch.save(sd, path)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("wrote", make_carvana())
    print("wrote", make_vgg13bn())
