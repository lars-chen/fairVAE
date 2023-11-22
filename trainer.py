import yaml
import torch

from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.nn import functional as F

# load hyperparameters from config.yml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

data_dir = config["data_dir"]
train_batch_size = config["train_batch_size"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CelebA(
    root=data_dir, split="train", target_type="identity", download=False
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, shuffle=True
)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x.view(-1, image_dim))
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD
    return loss
