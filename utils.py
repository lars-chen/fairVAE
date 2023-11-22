def count_parameters(model):
    """returns the total number of model parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
