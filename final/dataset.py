import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from common import MEAN, SIZE, STD, load_image, load_mask

# aug name -> list of v2 steps that run BEFORE ToDtype/Normalize.
# Pass the image as tv_tensors.Image and the mask as tv_tensors.Mask: geometric steps then
# move both with the same random parameters (mask with nearest), colour steps skip the mask.
AUGS = {
    # TODO(Stage 1) "none": squash to SIZE x SIZE like the official pipeline, nothing else
    # TODO(Stage 4) "geo": Run A, geometric
    # TODO(Stage 4) "geo_color": Run B, Run A + photometric
}


def build_transform(aug):
    """AUGS[aug] + uint8 -> float [0, 1] (images only) + ImageNet Normalize, as one v2.Compose."""
    # TODO(Stage 1)
    raise NotImplementedError


class PetDataset(Dataset):
    def __init__(self, ids, aug="none"):
        self.ids = list(ids)
        self.transform = build_transform(aug)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """-> img (3, 256, 256) float32 normalised, mask (1, 256, 256) float32 of 0/1."""
        # TODO(Stage 1): load_image + load_mask, wrap as tv_tensors, run self.transform on both together
        raise NotImplementedError
