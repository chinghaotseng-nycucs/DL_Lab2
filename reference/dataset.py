import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from common import MEAN, SIZE, STD, load_image, load_mask, load_trimap


class EraseImageOnly:
    """Random erasing (a black patch) on the image only. The mask keeps the pet under the patch,
    so the model must recognise a pet that is partly covered instead of relying on one texture."""

    def __init__(self, p=0.5):
        self.erase = v2.RandomErasing(p=p, scale=(0.02, 0.15), value=0)

    def __call__(self, img, mask):
        return self.erase(img), mask


class NoiseImageOnly:
    """Gaussian sensor noise on the uint8 image only; the mask is untouched."""

    def __init__(self, p=0.2, sigma=(3.0, 20.0)):
        self.p, self.sigma = p, sigma

    def __call__(self, img, mask):
        if torch.rand(1).item() >= self.p:
            return img, mask
        s = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * torch.rand(1).item()
        noisy = (img.float() + s * torch.randn(img.shape)).round().clamp(0, 255).to(torch.uint8)
        return tv_tensors.wrap(noisy, like=img), mask


class ExactMaskResizedCrop(v2.RandomResizedCrop):
    """RandomResizedCrop that resizes the mask with NEAREST_EXACT (pixel centres) instead of NEAREST (pixel corners).
    The photo is resized around pixel centres, so plain NEAREST leaves the mask about half a pixel down-right of it.
    Measured on val: 256 x 256 masks made with NEAREST reach at most Dice 0.980 after the official resize back,
    NEAREST_EXACT ones 0.991. Same random crops as RandomResizedCrop; only the mask's resampling changes."""

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.Mask):
            m = v2.functional.crop(inpt, params["top"], params["left"], params["height"], params["width"])
            return v2.functional.resize(m, self.size, interpolation=v2.InterpolationMode.NEAREST_EXACT)
        return super().transform(inpt, params)


class ExactMaskResize(v2.Resize):
    """Resize that uses NEAREST_EXACT for the mask (see ExactMaskResizedCrop)."""

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.Mask):
            return v2.functional.resize(inpt, self.size, interpolation=v2.InterpolationMode.NEAREST_EXACT)
        return super().transform(inpt, params)


def copy_paste(img, mask, src_img, src_trimap, scale=(0.3, 1.0)):
    """Paste the pet of another training image onto img, before the usual augmentation.
    img (3, H, W) uint8, mask (1, H, W) uint8: the target. src_img (3, h, w) uint8, src_trimap (h, w): the source.
    Copies pet + border band (trimap 1 or 3) so fur edges come along; only trimap 1 becomes pet, as in every label."""
    H, W = img.shape[1:]
    region = torch.from_numpy(np.isin(src_trimap, (1, 3)))  # pixels that get copied
    pet = torch.from_numpy(src_trimap == 1)  # pixels labelled pet
    if not region.any():  # a few Oxford trimaps are empty: nothing to paste
        return img, mask
    ys, xs = torch.nonzero(region, as_tuple=True)
    y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
    src_img, region, pet = src_img[:, y0:y1, x0:x1], region[y0:y1, x0:x1], pet[y0:y1, x0:x1]
    if torch.rand(1).item() < 0.5:  # random horizontal flip
        src_img, region, pet = src_img.flip(-1), region.flip(-1), pet.flip(-1)
    # resize the pet's box to s x the largest size that fits in the target
    s = scale[0] + (scale[1] - scale[0]) * torch.rand(1).item()
    f = s * min(H / region.shape[0], W / region.shape[1])
    nh, nw = max(1, round(region.shape[0] * f)), max(1, round(region.shape[1] * f))
    src_img = F.interpolate(src_img[None].float(), size=(nh, nw), mode="bilinear", antialias=True)[0]
    src_img = src_img.round().clamp(0, 255).to(torch.uint8)
    region = F.interpolate(region[None, None].float(), size=(nh, nw), mode="nearest")[0, 0].bool()
    pet = F.interpolate(pet[None, None].float(), size=(nh, nw), mode="nearest")[0, 0].bool()
    # random position, then paste: the source covers the target inside region
    top = torch.randint(0, H - nh + 1, (1,)).item()
    left = torch.randint(0, W - nw + 1, (1,)).item()
    img, mask = img.clone(), mask.clone()
    patch, m = img[:, top : top + nh, left : left + nw], mask[0, top : top + nh, left : left + nw]
    patch[:, region] = src_img[:, region]
    m[region] = pet[region].to(mask.dtype)
    return img, mask


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
    # geo_light + camera effects, each with p 0.2: run 14's weak spots on the stress val set were blur, low
    # resolution and noise, where geo_color (which blurs) was much more robust
    "geo_light_cam": [
        v2.RandomResizedCrop((SIZE, SIZE), scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation((-15, 15)),
        v2.ColorJitter(brightness=0.4, contrast=0.4),
        v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 3.0))], p=0.2),  # out of focus, low resolution
        v2.RandomApply([v2.JPEG(quality=(15, 90))], p=0.2),  # messaging-app compression
        NoiseImageOnly(p=0.2),  # dark-room sensor noise
    ],
}


def exact_masks(step):
    """The same step, but resizing masks around pixel centres (see ExactMaskResizedCrop)."""
    if isinstance(step, v2.RandomResizedCrop):
        return ExactMaskResizedCrop(step.size, scale=step.scale, ratio=step.ratio, interpolation=step.interpolation,
                                    antialias=step.antialias)
    if isinstance(step, v2.Resize):
        return ExactMaskResize(step.size, interpolation=step.interpolation, antialias=step.antialias)
    return step


def build_transform(aug, exact=False):
    return v2.Compose(
        [exact_masks(s) if exact else s for s in AUGS[aug]]
        + [
            v2.ToDtype(torch.float32, scale=True),  # images only; the mask stays uint8
            v2.Normalize(MEAN, STD),
        ]
    )


class PetDataset(Dataset):
    def __init__(self, ids, aug="none", copy_paste_p=0.0, exact_masks=False):
        self.ids = list(ids)
        self.transform = build_transform(aug, exact_masks)
        self.copy_paste_p = copy_paste_p  # chance of pasting another image's pet; sources come from self.ids only

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = v2.functional.pil_to_tensor(load_image(image_id))  # (3, h, w) uint8
        mask = torch.from_numpy(load_mask(image_id))[None]  # (1, h, w) uint8
        if self.copy_paste_p and torch.rand(1).item() < self.copy_paste_p:
            other = self.ids[torch.randint(len(self.ids), (1,)).item()]
            img, mask = copy_paste(img, mask, v2.functional.pil_to_tensor(load_image(other)), load_trimap(other))
        img, mask = self.transform(tv_tensors.Image(img), tv_tensors.Mask(mask))
        return img, mask.float()
