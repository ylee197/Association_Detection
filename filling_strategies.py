import math

import torch
from torch_scatter import scatter
import torch_sparse

from feature_propagation import FeaturePropagation

def feature_propagation(edge_index,X, feature_mask, num_iterations):
  propagation_model = FeaturePropagation(num_iterations=num_iterations)

  return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)

def filling(edge_index, X, feature_mask, num_iterations = None):
    X_reconstructed = feature_propagation(edge_index, X, feature_mask, num_iterations)

    return X_reconstructed
