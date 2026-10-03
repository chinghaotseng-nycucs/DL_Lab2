import sys

import torch
from model import UNet

# python check_weights.py DL_Lab2_315551118_曾敬豪.pth
# The TA's loading snippet, verbatim, in a fresh process. If this fails, the private score is 0.

weight_path = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1).to(device)
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()

assert isinstance(state_dict, dict) and all(torch.is_tensor(v) for v in state_dict.values()), "not a plain state dict"
bad = [k for k in state_dict if k.startswith(("_orig_mod.", "module."))]
assert not bad, f"wrapper prefixes in keys: {bad[:3]}"
floats = {v.dtype for v in state_dict.values() if v.is_floating_point()}
assert floats == {torch.float32}, f"float dtypes {floats}, expected float32 only"
with torch.no_grad():
    out = model(torch.randn(2, 3, 256, 256, device=device))
assert out.shape == (2, 1, 256, 256), out.shape
print(f"OK: strict load passed, {len(state_dict)} tensors, output {tuple(out.shape)}")
