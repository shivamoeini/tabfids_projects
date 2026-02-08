import numpy as np
import torch

class TemporalAveraging:
    """Implement temporal averaging for data preprocessing"""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
    
    def apply(self, data):
        """
        Apply temporal averaging to data
        
        Args:
            data: numpy array or torch tensor of shape (samples, features)
            
        Returns:
            Averaged data
        """
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
        else:
            data_np = data
        
        # Create sliding window
        n_samples = len(data_np)
        averaged_data = []
        
        for i in range(self.window_size - 1, n_samples):
            window = data_np[i - self.window_size + 1: i + 1]
            avg = np.mean(window, axis=0)
            averaged_data.append(avg)
        
        if isinstance(data, torch.Tensor):
            return torch.FloatTensor(np.array(averaged_data))
        else:
            return np.array(averaged_data)
    
    def apply_to_dataset(self, dataset, labels):
        """Apply temporal averaging to entire dataset"""
        averaged_data = self.apply(dataset)
        
        # Adjust labels to match new data length
        # Keep labels for the last element of each window
        averaged_labels = labels[self.window_size - 1:]
        
        return averaged_data, averaged_labels