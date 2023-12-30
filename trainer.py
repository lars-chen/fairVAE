import yaml
import torch
import os
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import transforms

from torchvision.datasets import CelebA
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

# project modules
from vae import VAE
from utils import read_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(
    data_dir,
    train_batch_size,
    image_size,
    epochs,
    lr,
    beta,
    latent_dim,
    num_samples,
    checkpoints,
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

# first 15 images are missing in CelebA dataset
dataset = CelebA(root=data_dir, split="train", transform=transform, download=False)
dataset = Subset(dataset=dataset, indices=np.arange(16, 162770, 1))

# test_set = dataset = CelebA(
#     root=data_dir, split="test", transform=transform, download=False
# )

train_loader = DataLoader(
    dataset=dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)


model = VAE(image_size, beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(epoch):
    nsamples = 500
    model.train()
    train_loss = 0

    for batch_idx, (xs, labels) in enumerate(train_loader):
        if batch_idx > nsamples:
            break

        torch.cuda.empty_cache()
        xs = xs.to(device)
        ys = labels[:, label_idxs]
        t = labels[:, t_idx]

        optimizer.zero_grad()
        recon_batch, mu, log_var, prior_mu, prior_log_var = model(xs, ys, t)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = model.loss_function(
            recon_batch, xs, mu, log_var, prior_mu, prior_log_var
        )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(xs),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item() / len(xs),
            )
        )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch):
    pass


if __name__ == "__main__":
    print(f"epochs: {epochs}")
    for epoch in range(1, epochs + 1):
        train(epoch)

        if ~(epoch % checkpoints):
            torch.save(model, f"models/vae_model_{epoch}.pth")
            # test(epoch)
            with torch.no_grad():
                sample = torch.randn(num_samples, latent_dim).to(device)
                sample = model.decode(sample).cpu()
                save_image(
                    sample.view(num_samples, 3, model.image_size, model.image_size),
                    f"models/sample_{str(epoch)}.png",
                )
