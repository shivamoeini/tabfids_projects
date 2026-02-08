import torch
import torch.nn as nn
import numpy as np

class DataDrivenFeatureElimination:
    """Implement DDFE algorithm from the paper"""
    
    def __init__(self, model, threshold=0.01):
        self.model = model
        self.threshold = threshold  # Performance degradation threshold
    
    def evaluate_feature_importance(self, data_loader):
        """
        Evaluate importance of each feature
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Feature importance scores
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # Get baseline performance
        baseline_accuracy = self._evaluate_accuracy(data_loader)
        
        # Get number of features
        sample_features, _ = next(iter(data_loader))
        num_features = sample_features.shape[1]
        
        importance_scores = []
        
        # Evaluate each feature
        for feature_idx in range(num_features):
            modified_accuracy = self._evaluate_accuracy_with_feature_removed(
                data_loader, feature_idx
            )
            
            # Calculate performance drop
            performance_drop = baseline_accuracy - modified_accuracy
            importance_scores.append(performance_drop)
        
        return np.array(importance_scores)
    
    def _evaluate_accuracy(self, data_loader):
        """Evaluate model accuracy"""
        device = next(self.model.parameters()).device
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _evaluate_accuracy_with_feature_removed(self, data_loader, feature_idx):
        """Evaluate accuracy with specific feature removed (set to 0)"""
        device = next(self.model.parameters()).device
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Set feature to 0
                features_modified = features.clone()
                features_modified[:, feature_idx] = 0
                
                outputs = self.model(features_modified)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def select_features(self, importance_scores):
        """Select features based on importance scores"""
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        # Select top features (keep 60% as in the paper)
        num_features_to_keep = int(len(importance_scores) * 0.6)
        selected_indices = sorted_indices[:num_features_to_keep]
        
        return selected_indices
    
    def create_feature_mask(self, selected_indices, num_features):
        """Create mask for selected features"""
        mask = torch.zeros(num_features)
        mask[selected_indices] = 1
        return mask