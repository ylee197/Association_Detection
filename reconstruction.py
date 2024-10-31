import numpy as np
import torch

def spatial_reconstruction(x, x_pred, feature_mask):
    """
    For each feature channel we compute the relative error (using L2 norm): ||x - x_pred|| / ||x||. 
    Then, we take the mean over channels.
    """
    channel_relative_errors = []
    n_features = x.shape[1]
    
    for i in range(n_features):
        channel_feature_mask = feature_mask[:, i]
        y = x[:, i][~channel_feature_mask] 
        y_pred = x_pred[:, i][~channel_feature_mask]
        if torch.norm(y).item() != 0.0:
            channel_relative_error = (torch.norm(y - y_pred) / torch.norm(y)).item()
            channel_relative_errors.append(channel_relative_error)
    
    return np.mean(channel_relative_errors)