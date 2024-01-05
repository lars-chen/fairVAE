import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import transforms

from torchvision.datasets import CelebA
from torchvision.utils import save_image

# project modules
from vae import VAE
from utils import read_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(
    data_dir,
    train_batch_size,
    image_size,
    _,
    _,
    beta,
    latent_dim,
    num_samples,
    _,
    label_idxs,
    t_idx,
) = read_config()

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


test_set = dataset = CelebA(
    root=data_dir, split="test", transform=transform, download=False
)

test_loader = DataLoader(
    dataset=test_set,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

# load model and state dict....

# test stuff...
