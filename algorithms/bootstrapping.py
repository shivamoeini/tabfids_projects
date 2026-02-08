import numpy as np
from sklearn.utils import resample

class Bootstrapping:
    """Implement bootstrapping for handling imbalanced datasets"""
    
    def __init__(self, target_ratio=0.5):
        self.target_ratio = target_ratio
    
    def balance_dataset(self, X, y):
        """
        Balance dataset using bootstrapping
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Balanced X and y
        """
        # Separate classes
        unique_classes = np.unique(y)
        
        if len(unique_classes) != 2:
            raise ValueError("Bootstrapping currently supports binary classification")
        
        # Identify minority and majority classes
        class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        # Separate data
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        # Calculate target number of samples for minority class
        n_majority = len(X_majority)
        n_minority_target = int(n_majority * self.target_ratio / (1 - self.target_ratio))
        
        # Bootstrap minority class
        X_minority_resampled, y_minority_resampled = resample(
            X_minority, y_minority,
            n_samples=n_minority_target,
            random_state=42,
            replace=True
        )
        
        # Combine datasets
        X_balanced = np.vstack([X_majority, X_minority_resampled])
        y_balanced = np.hstack([y_majority, y_minority_resampled])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        return X_balanced, y_balanced