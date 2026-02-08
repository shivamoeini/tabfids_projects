import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class FederatedServer:
    """Federated Learning Server"""
    
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_models = []
        self.client_data_sizes = []
    
    def aggregate_models(self, client_models, client_data_sizes):
        """
        Aggregate client models using FedAvg
        
        Args:
            client_models: List of client model state_dicts
            client_data_sizes: List of data sizes for each client
            
        Returns:
            Updated global model state_dict
        """
        total_data_size = sum(client_data_sizes)
        
        # Initialize averaged parameters
        avg_state_dict = OrderedDict()
        
        # Get parameter names from first client
        param_names = client_models[0].keys()
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(client_models[0][param_name])
            
            for client_idx, client_state_dict in enumerate(client_models):
                weight = client_data_sizes[client_idx] / total_data_size
                weighted_sum += weight * client_state_dict[param_name]
            
            avg_state_dict[param_name] = weighted_sum
        
        return avg_state_dict
    
    def update_global_model(self, avg_state_dict):
        """Update global model with averaged parameters"""
        self.global_model.load_state_dict(avg_state_dict)
    
    def distribute_model(self):
        """Distribute global model to clients"""
        return self.global_model.state_dict()