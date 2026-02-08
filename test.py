import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from config import Config
from models.base_cnn import TabFIDSModel
from train import evaluate_transferability

def calculate_attack_accuracy(y_true, y_pred, benign_class=0):
    """
    Calculate attack accuracy as defined in the paper
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        benign_class: Label for benign class
        
    Returns:
        Attack accuracy score
    """
    # Separate benign and attack
    benign_mask = y_true == benign_class
    attack_mask = y_true != benign_class
    
    # Calculate accuracy for each
    benign_accuracy = accuracy_score(y_true[benign_mask], y_pred[benign_mask])
    attack_accuracy = accuracy_score(y_true[attack_mask], y_pred[attack_mask])
    
    # Combined attack accuracy (equal weight)
    attack_acc = (benign_accuracy + attack_accuracy) / 2
    
    return attack_acc

def test_model(model, X_test, y_test):
    """Test model performance"""
    device = torch.device(Config.DEVICE)
    model = model.to(device)
    model.eval()
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    attack_acc = calculate_attack_accuracy(y_test, predictions)
    
    print("\n=== Test Results ===")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Attack Accuracy: {attack_acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'attack_accuracy': attack_acc
    }

def plot_transferability_matrix(transfer_results, attack_classes):
    """Plot transferability matrix"""
    num_classes = len(attack_classes)
    matrix = np.zeros((num_classes, num_classes))
    
    for train_idx, test_idx, accuracy in transfer_results:
        matrix[train_idx, test_idx] = accuracy
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    plt.xlabel('Test Attack Class')
    plt.ylabel('Train Attack Class')
    plt.title('Transferability Matrix')
    
    # Set tick labels
    plt.xticks(range(num_classes), attack_classes, rotation=45, ha='right')
    plt.yticks(range(num_classes), attack_classes)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            if matrix[i, j] > 0:
                plt.text(j, i, f'{matrix[i, j]:.2f}',
                        ha='center', va='center',
                        color='black' if matrix[i, j] < 0.7 else 'white')
    
    plt.tight_layout()
    plt.savefig(f"{Config.RESULTS_DIR}/transferability_matrix.png")
    plt.show()

def main():
    """Main testing function"""
    print("Testing TabFIDS Model")
    
    # Load test data (synthetic for demonstration)
    np.random.seed(42)
    num_samples = 2000
    num_features = Config.NUM_FEATURES
    num_classes = len(Config.ATTACK_CLASSES)
    
    X_test = np.random.randn(num_samples, num_features)
    y_test = np.random.randint(0, num_classes, num_samples)
    
    # Load model
    model = TabFIDSModel(
        backbone_type=Config.MODEL_TYPE,
        input_dim=num_features,
        num_classes=num_classes
    )
    
    try:
        model.load_state_dict(torch.load(f"{Config.RESULTS_DIR}/tabfids_model.pth"))
        print("Model loaded successfully")
    except:
        print("No saved model found, using random initialization")
    
    # Test overall performance
    print("\n1. Overall Performance Test")
    test_results = test_model(model, X_test, y_test)
    
    # Test transferability
    print("\n2. Transferability Analysis")
    transfer_results = []
    
    for train_attack in range(1, min(6, num_classes)):
        for test_attack in range(1, min(6, num_classes)):
            if train_attack != test_attack:
                accuracy = evaluate_transferability(
                    model, train_attack, test_attack, X_test, y_test
                )
                transfer_results.append((train_attack, test_attack, accuracy))
                
                if accuracy >= 0.7:
                    print(f"Train: {Config.ATTACK_CLASSES[train_attack]}, "
                          f"Test: {Config.ATTACK_CLASSES[test_attack]}, "
                          f"Accuracy: {accuracy:.4f}")
    
    # Plot transferability matrix
    plot_transferability_matrix(transfer_results, Config.ATTACK_CLASSES[:6])
    
    # Count transferable pairs
    transferable_pairs = [(t, test, acc) for t, test, acc in transfer_results if acc >= 0.7]
    print(f"\n3. Transferability Summary")
    print(f"Total transferable pairs: {len(transferable_pairs)}")
    print(f"Total possible pairs: {len(transfer_results)}")
    
    return test_results, transfer_results

if __name__ == "__main__":
    main()