import yaml
import torch
import warnings
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, kl

warnings.filterwarnings("ignore")

# load hyperparameters from config.yml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

celeb_path = config["data_dir"]
image_size = config["image_size"]
latent_dim = config["latent_dim"]
label_dim = config["label_dim"]


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

        # Build Prior Network
        self.prior_nn = nn.Sequential(
            nn.Linear(label_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )

        self.prior_mu = nn.Linear(100, latent_dim)
        self.prior_var = nn.Linear(100, latent_dim)

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

        self.cnn = nn.Sequential(*modules)
        self.size = self.cnn(torch.zeros(1, 3, image_size, image_size)).shape[2]

        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dims[-1] * self.size * self.size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        self.context_nn = nn.Linear(256 + label_dim + 1, 200)

        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_var = nn.Linear(200, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(
            latent_dim + 1, hidden_dims[-1] * self.size * self.size
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

    def encode(self, image, x, t):
        result = self.cnn(image)  # extract image features
        result = torch.flatten(result, start_dim=1)  # map to cnn output to vector
        result = self.encoder_fc(result)

        # condition on labels and treatment
        result = torch.cat((result, x, t), dim=1)
        result = self.context_nn(result)

        # define variational posterior distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = torch.clamp_(log_var, -10, 10)
        return mu, log_var

    def prior_distribution(self, x):
        result = self.prior_nn(x)
        prior_mu = self.prior_mu(result)
        prior_log_var = self.prior_var(result)
        prior_log_var = torch.clamp_(prior_log_var, -10, 10)
        return MultivariateNormal(
            loc=prior_mu,
            covariance_matrix=torch.diag_embed(torch.exp(0.5 * prior_log_var)),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, t):
        result = self.decoder_input(torch.cat((z, t), dim=1))
        result = result.view(-1, self.final_dim, self.size, self.size)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = celeb_transform1(result)
        result = torch.flatten(result, start_dim=1)
        result = torch.nan_to_num(result)
        return result

    def forward(self, image, x, t):
        mu, log_var = self.encode(image, x, t)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, t), mu, log_var

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_image, image, x, mu, log_var):
        MSE = F.mse_loss(recon_image, image.view(-1, self.image_dim))

        posterior = MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(0.5 * log_var))
        )
        prior = self.prior_distribution(x)

        KLD = torch.mean(kl.kl_divergence(posterior, prior))
        kld_weight = 0.0025
        loss = MSE + kld_weight * KLD
        return loss
