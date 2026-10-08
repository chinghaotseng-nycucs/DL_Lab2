import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from common import MEAN, SIZE, STD, load_image, load_mask


class EraseImageOnly:
    """Random erasing (a black patch) on the image only. The mask keeps the pet under the patch,
    so the model must recognise a pet that is partly covered instead of relying on one texture."""

    def __init__(self, p=0.5):
        self.erase = v2.RandomErasing(p=p, scale=(0.02, 0.15), value=0)

    def __call__(self, img, mask):
        return self.erase(img), mask


# aug name -> list of extra steps before ToDtype/Normalize.
# Geometric steps hit image and mask with the same parameters (mask uses nearest);
# colour steps skip the mask automatically because it is a tv_tensors.Mask.
AUGS = {
    # Baseline: squash to 256x256 like the official pipeline, nothing else
    "none": [v2.Resize((SIZE, SIZE), antialias=True)],
    # Run A: geometric
    "geo": [
        v2.RandomResizedCrop((SIZE, SIZE), scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation((-15, 15)),
    ],
    # Run B: Run A + photometric, for other cameras and lighting
    "geo_color": [
        v2.RandomResizedCrop((SIZE, SIZE), scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation((-15, 15)),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomGrayscale(p=0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
    ],
    # geo + lighting only: for backlit and dark photos, without geo_color's hue/saturation/grayscale/blur
    "geo_light": [
        v2.RandomResizedCrop((SIZE, SIZE), scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation((-15, 15)),
        v2.ColorJitter(brightness=0.4, contrast=0.4),
    ],
    # geo + random erasing: against relying on fur texture (blankets) and for occlusion (people)
    "geo_erase": [
        v2.RandomResizedCrop((SIZE, SIZE), scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation((-15, 15)),
        EraseImageOnly(p=0.5),
    ],
}


def build_transform(aug):
    return v2.Compose(
        AUGS[aug]
        + [
            v2.ToDtype(torch.float32, scale=True),  # images only; the mask stays uint8
            v2.Normalize(MEAN, STD),
        ]
    )


class PetDataset(Dataset):
    def __init__(self, ids, aug="none"):
        self.ids = list(ids)
        self.transform = build_transform(aug)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = tv_tensors.Image(v2.functional.pil_to_tensor(load_image(image_id)))  # (3, h, w) uint8
        mask = tv_tensors.Mask(torch.from_numpy(load_mask(image_id))[None])  # (1, h, w) uint8
        img, mask = self.transform(img, mask)
        return img, mask.float()
