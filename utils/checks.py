def get_weight_norm(model, norm_type=2):
    """Get the norm of the model weights"""
    norm = 0
    for p in model.parameters():
        norm += p.norm(norm_type).item()
    return norm
