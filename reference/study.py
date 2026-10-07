from torch.utils.data import DataLoader
from common import get_split
from dataset import PetDataset

train_ids, _ = get_split()
ds = PetDataset(train_ids, "none")
print("len(ds)", len(ds))
img, mask = ds[0]
print(
    "img.shape, img.dtype, mask.shape, mask.dtype, mask.unique() ",
    img.shape,
    img.dtype,
    mask.shape,
    mask.dtype,
    mask.unique(),
)
loader = DataLoader(ds, batch_size=4, shuffle=True)
imgs, masks = next(iter(loader))
print("imgs.shape, masks.shape", imgs.shape, masks.shape)
print("len(loader)", len(loader))
