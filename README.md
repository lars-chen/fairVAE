# fairVAE

1. Treatment condition:
    - Male
2. Conditioned labels:
    - Young
    - Smiling
    - Mouth_Slightly_Open
    - No_Beard
    - Bald
    - Pale_Skin

Next time:
    -. Write testing 
    
    

sample = (
    model.prior_distribution(
        torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(
            0
        )
    )
    .sample((num_samples, 1))
    .squeeze()
)

