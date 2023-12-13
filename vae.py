import yaml
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

# load hyperparameters from config.yml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

celeb_path = config["data_dir"]
image_size = config["image_size"]
latent_dim = config["latent_dim"]


celeb_transform = transforms.Compose(
    [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)  # used when transforming image to tensor

celeb_transform1 = transforms.Compose(
    [transforms.Resize(image_size, antialias=True), transforms.CenterCrop(image_size)]
)  # used by decode method to transform final output


class VAE(nn.Module):
    def __init__(self, image_size, beta):
        super(VAE, self).__init__()

        self.image_size = image_size
        self.image_dim = 3 * image_size * image_size
        self.beta = beta

        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, image_size, image_size))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1] * self.size * self.size
        )
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.size, self.size)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = celeb_transform1(result)
        result = torch.flatten(result, start_dim=1)
        result = torch.nan_to_num(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x.view(-1, self.image_dim))
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kld_weight = 0.00025
        loss = MSE + kld_weight * KLD
        return loss
