import yaml
import torch

from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from torchvision.datasets import CelebA, ImageFolder
from torchvision.utils import save_image

# project modules
from vae import VAE

# load hyperparameters from config.yml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

data_dir = config["data_dir"]

train_batch_size = config["train_batch_size"]
image_size = config["image_size"]
epochs = config["epochs"]
lr = float(config["lr"])
beta = config["beta"]
latent_dim = config["latent_dim"]
num_samples = config["num_samples"]
checkpoints = config["checkpoints"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load CelebA data from data directory
# train_dataset = CelebA(
#     root=data_dir, split="train", target_type="identity", download=False
# )
# train_loader = DataLoader(
#     dataset=train_dataset, batch_size=train_batch_size, shuffle=True
# )


transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

dataset = ImageFolder(data_dir, transform)

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
    nsamples = 100
    model.train()
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx > nsamples:
            break

        torch.cuda.empty_cache()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = model.loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item() / len(data),
            )
        )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


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
