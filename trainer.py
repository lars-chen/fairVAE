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
from utils import read_config, count_parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(
    validate,
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

model = VAE(image_size, beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# first 15 images are missing in CelebA dataset
dataset = CelebA(root=data_dir, split="train", transform=transform, download=False)
dataset = Subset(dataset=dataset, indices=np.arange(16, 162770, 1))

train_loader = DataLoader(
    dataset=dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

if validate:
    validation_set = dataset = CelebA(
        root=data_dir, split="valid", transform=transform, download=False
    )

    validation_loader = DataLoader(
        dataset=validation_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx > 200:
            break

        torch.cuda.empty_cache()
        images = images.to(device)
        xs = torch.tensor(labels[:, label_idxs], dtype=torch.float32)
        ts = torch.tensor(labels[:, t_idx].unsqueeze(1), dtype=torch.float32)

        optimizer.zero_grad()
        recon_batch, mu, log_var = model(images, xs, ts)
        loss = model.loss_function(recon_batch, images, xs, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(images),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item() / len(images),
            )
        )

    avg_loss = train_loss / len(train_loader.dataset)
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_loss))

    return avg_loss


# def validate(epoch):
#     print("Testing...")
#     test_loss = 0
#     for batch_idx, (images, labels) in enumerate(validation_loader):
#         if batch_idx > 100:
#             break
#         torch.cuda.empty_cache()
#         images = images.to(device)
#         xs = torch.tensor(labels[:, label_idxs], dtype=torch.float32)
#         ts = torch.tensor(labels[:, t_idx].unsqueeze(1), dtype=torch.float32)
#         recon_batch, mu, log_var = model(images, xs, ts)
#         loss = model.loss_function(recon_batch, images, xs, mu, log_var)
#         test_loss += loss.item()

#         print(
#             "Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 epoch,
#                 batch_idx * len(images),
#                 len(validation_loader.dataset),
#                 100.0 * batch_idx / len(validation_loader),
#                 loss.item() / len(images),
#             )
#         )

#     avg_loss = test_loss / len(validation_loader.dataset)
#     print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_loss))
#     writer.add_scalar("Loss/test", avg_loss, epoch)


if __name__ == "__main__":
    writer = SummaryWriter()
    print(model)
    print(f"Epochs: {epochs}")
    print(f"Model parameters: {np.round(count_parameters(model) / 1e6, 1)} million")
    print(f"Running model on device: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train(epoch)
        writer.add_scalar("Loss/train", train_loss, global_step=epoch)

        if epoch % checkpoints == 0:
            epoch_dir = writer.log_dir.replace("\\", "/") + f"/epoch_{epoch}/"
            os.makedirs(epoch_dir)
            torch.save(model, epoch_dir + "vae_model.pth")

            model.eval()
            with torch.no_grad():
                sample = torch.randn(num_samples, latent_dim).to(device)
                sample_0 = model.decode(sample, torch.zeros((num_samples, 1))).cpu()
                sample_1 = model.decode(sample, torch.ones((num_samples, 1))).cpu()

                save_image(
                    sample_0.view(num_samples, 3, model.image_size, model.image_size),
                    epoch_dir + "sample_0.png",
                )

                save_image(
                    sample_1.view(num_samples, 3, model.image_size, model.image_size),
                    epoch_dir + "sample_1.png",
                )

                # if validate:
                #     validate(epoch)

    writer.flush()
    writer.close()
