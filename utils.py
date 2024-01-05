import yaml
import numpy as np


def count_parameters(model):
    """returns the total number of model parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def read_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    data_dir = config["data_dir"]

    validate = config["validate"]
    train_batch_size = config["train_batch_size"]
    image_size = config["image_size"]
    epochs = config["epochs"]
    lr = float(config["lr"])
    beta = config["beta"]
    latent_dim = config["latent_dim"]
    num_samples = config["num_samples"]
    checkpoints = config["checkpoints"]
    label_idxs = [
        config["labels"]["Young"],
        config["labels"]["Smiling"],
        config["labels"]["Mouth_Slightly_Open"],
        config["labels"]["No_Beard"],
        config["labels"]["Bald"],
        config["labels"]["Pale_Skin"],
    ]
    t_idx = config["labels"]["Male"]
    return (
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
    )


def kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    """KL divergence between two multivarariate Gaussian distributions

    Args:
        mu_p (_type_): _description_
        sigma_p (_type_): _description_
        mu_q (_type_): _description_
        sigma_q (_type_): _description_

    Returns:
        _type_: _description_
    """
    kl_divergence = (
        (mu_q - mu_p).T @ np.linalg.inv(sigma_q) @ (mu_q - mu_p)
        + np.trace(np.linalg.inv(sigma_q) @ sigma_p)
        - np.log(np.linalg.det(sigma_p) / np.linalg.det(sigma_q))
    )
    return kl_divergence
