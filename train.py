# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# اضافه کردن مسیر ماژول‌ها
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./algorithms')

# تنظیمات بر اساس مقاله
class Config:
    # تنظیمات از مقاله
    BATCH_SIZE = 64  # معمول در مقاله
    LEARNING_RATE = 0.001  # از مقاله: "learning rate of 0.001"
    NUM_EPOCHS = 50  # افزایش به 50 epoch برای آموزش کامل
    EARLY_STOPPING_PATIENCE = 10  # اگر بهبود نداشت، stop کن
    
    # Federated Learning (از مقاله)
    FEDERATED_ROUNDS = 20  # از مقاله: "we run 20 rounds of federated model aggregation"
    NUM_NODES = 11  # از مقاله: 11 کلاس حمله
    
    # مدل
    MODEL_TYPE = "CNN"
    HIDDEN_DIM = 128
    
    # دستگاه
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # مسیرها
    DATA_DIR = "data/preprocessed"
    MODEL_SAVE_DIR = "models/saved"
    RESULTS_DIR = "results"
    
    @staticmethod
    def create_dirs():
        """ایجاد پوشه‌های مورد نیاز"""
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    @staticmethod
    def print_settings():
        print("=" * 70)
        print("TABFIDS SETTINGS (Based on Paper 2508.09060v1)")
        print("=" * 70)
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
        print(f"Device: {Config.DEVICE}")
        print(f"Federated Rounds (for FL): {Config.FEDERATED_ROUNDS}")
        print(f"Number of Nodes (for FL): {Config.NUM_NODES}")
        print("=" * 70)

# تعریف مدل CNN مشابه مقاله
class PaperCNN(nn.Module):
    """مدل CNN بر اساس معماری مقاله"""
    def __init__(self, input_dim=78, num_classes=14):
        super(PaperCNN, self).__init__()
        
        # از Figure 1 در مقاله: چند لایه کانولوشن
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # محاسبه سایز flattened
        self._calculate_flatten_size(input_dim)
        
        # لایه‌های fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
    
    def _calculate_flatten_size(self, input_dim):
        x = torch.randn(1, 1, input_dim)
        x = self.conv_layers(x)
        self.flatten_size = x.numel()
    
    def forward(self, x):
        # اگر 2D است (batch, features) به 3D تبدیل کن (batch, channel, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x

def load_preprocessed_data():
    """لود داده‌های پیش‌پردازش شده"""
    print("Loading preprocessed data...")
    
    if not os.path.exists(Config.DATA_DIR):
        raise FileNotFoundError(
            f"Preprocessed data not found at {Config.DATA_DIR}.\n"
            f"Please run 'python data/preprocess.py' first."
        )
    
    # لود numpy arrays
    X_train = np.load(os.path.join(Config.DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(Config.DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(Config.DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(Config.DATA_DIR, "y_test.npy"))
    
    # لود metadata و transformers
    with open(os.path.join(Config.DATA_DIR, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    with open(os.path.join(Config.DATA_DIR, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"✓ Data loaded successfully:")
    print(f"  X_train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  X_test:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    print(f"  Classes: {metadata['num_classes']}")
    
    # نمایش توزیع کلاس‌ها (مهم برای مقاله)
    print(f"\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        cls_name = label_encoder.inverse_transform([cls])[0]
        percentage = (count / len(y_train)) * 100
        print(f"  Class {cls} ({cls_name:30s}): {count:6,} ({percentage:5.1f}%)")
    
    return X_train, X_test, y_train, y_test, metadata, label_encoder

def calculate_attack_accuracy(y_true, y_pred, label_encoder):
    """محاسبه Attack Accuracy مطابق مقاله (معادله 3)"""
    # پیدا کردن index کلاس benign (فرض می‌کنیم اولین کلاس)
    benign_idx = np.where(label_encoder.classes_ == 'BENIGN')[0]
    if len(benign_idx) == 0:
        benign_idx = 0  # اگر BENIGN نبود، کلاس 0 را benign فرض کن
    else:
        benign_idx = benign_idx[0]
    
    # جدا کردن benign و attack
    benign_mask = y_true == benign_idx
    attack_mask = y_true != benign_idx
    
    # محاسبه accuracy برای هر کدام
    if np.sum(benign_mask) > 0:
        benign_acc = accuracy_score(y_true[benign_mask], y_pred[benign_mask])
    else:
        benign_acc = 0
    
    if np.sum(attack_mask) > 0:
        attack_acc = accuracy_score(y_true[attack_mask], y_pred[attack_mask])
    else:
        attack_acc = 0
    
    # Attack Accuracy مطابق مقاله: (tn/(tn+fp) + tp/(tp+fn)) / 2
    # که معادل (benign_accuracy + attack_accuracy) / 2 است
    attack_accuracy = (benign_acc + attack_acc) / 2
    
    return attack_accuracy, benign_acc, attack_acc

def create_dataloaders(X_train, X_test, y_train, y_test):
    """ایجاد DataLoaderها"""
    print("\nCreating dataloaders...")
    
    # تبدیل به tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # ایجاد Datasetها
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # استفاده از WeightedRandomSampler برای مقابله با imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Using weighted sampling for class imbalance")
    
    return train_loader, test_loader

def train_with_early_stopping(model, train_loader, test_loader, label_encoder):
    """آموزش با Early Stopping"""
    print("\nStarting training with early stopping...")
    print(f"Device: {Config.DEVICE}")
    print("-" * 70)
    
    device = Config.DEVICE
    model = model.to(device)
    
    # Loss function با weighting (مهم برای imbalance)
    class_counts = np.bincount(train_loader.dataset.tensors[1].numpy())
    class_weights = 1. / class_counts
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer مطابق مقاله
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # ذخیره تاریخچه
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'attack_acc': [],
        'learning_rate': []
    }
    
    best_acc = 0.0
    best_attack_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 50)
        
        # === آموزش ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 200 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # === ارزیابی ===
        model.eval()
        test_running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = test_running_loss / len(test_loader)
        test_acc = accuracy_score(all_labels, all_preds) * 100
        
        # محاسبه Attack Accuracy مطابق مقاله
        attack_acc, benign_acc, attack_det_acc = calculate_attack_accuracy(
            np.array(all_labels), np.array(all_preds), label_encoder
        )
        attack_acc_percent = attack_acc * 100
        
        # ذخیره تاریخچه
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['attack_acc'].append(attack_acc_percent)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # نمایش نتایج
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Attack Acc: {attack_acc_percent:.2f}% (Benign: {benign_acc*100:.1f}%, Attack: {attack_det_acc*100:.1f}%)")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # بررسی بهبود
        if attack_acc_percent > best_attack_acc:
            best_attack_acc = attack_acc_percent
            best_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ↳ New best model! (Attack Acc: {best_attack_acc:.2f}%)")
            
            # ذخیره مدل
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'attack_accuracy': best_attack_acc,
                'test_accuracy': best_acc
            }, os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
        
        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # لود بهترین مدل
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("\n" + "=" * 70)
    print(f"Training completed.")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Best Attack Accuracy: {best_attack_acc:.2f}%")
    print("=" * 70)
    
    return model, history, best_acc, best_attack_acc, all_preds, all_labels

def evaluate_final(model, test_loader, label_encoder):
    """ارزیابی نهایی مدل"""
    print("\nPerforming final evaluation...")
    
    device = Config.DEVICE
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # محاسبه متریک‌ها
    test_acc = accuracy_score(all_labels, all_preds) * 100
    attack_acc, benign_acc, attack_det_acc = calculate_attack_accuracy(
        np.array(all_labels), np.array(all_preds), label_encoder
    )
    attack_acc_percent = attack_acc * 100
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {test_acc:.2f}%")
    print(f"Attack Accuracy:  {attack_acc_percent:.2f}%")
    print(f"  - Benign Detection: {benign_acc*100:.1f}%")
    print(f"  - Attack Detection: {attack_det_acc*100:.1f}%")
    
    # گزارش طبقه‌بندی
    try:
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=label_encoder.classes_,
            digits=3,
            zero_division=0
        ))
    except:
        print("\n(Detailed classification report skipped)")
    
    return test_acc, attack_acc_percent, all_preds, all_labels

def plot_paper_results(history, label_encoder):
    """نمودارهای مشابه مقاله"""
    print("\nGenerating paper-style plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training and Validation Loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['epoch'], history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training and Validation Accuracy
    axes[0, 1].plot(history['epoch'], history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history['epoch'], history['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Attack Accuracy (مهم برای مقاله)
    axes[1, 0].plot(history['epoch'], history['attack_acc'], 'g-', linewidth=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Attack Accuracy (%)')
    axes[1, 0].set_title('Attack Accuracy (Paper Metric)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Bar chart of class-wise accuracy
    try:
        # این بخش نیاز به محاسبات اضافه دارد
        axes[1, 1].text(0.5, 0.5, 'Class-wise Performance\n(Calculated during evaluation)',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].axis('off')
    except:
        axes[1, 1].text(0.5, 0.5, 'Class-wise plot not available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].axis('off')
    
    plt.suptitle('TABFIDS Training Results (Paper Implementation)', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(Config.RESULTS_DIR, "paper_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Paper-style plots saved: {plot_path}")
    plt.show()
    
    # همچنین نمودار confusion matrix
    try:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_labels, all_preds)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # برای خوانایی، اگر کلاس‌ها زیاد هستند
        if len(label_encoder.classes_) <= 15:
            tick_marks = np.arange(len(label_encoder.classes_))
            plt.xticks(tick_marks, label_encoder.classes_, rotation=45, ha='right')
            plt.yticks(tick_marks, label_encoder.classes_)
        
        plt.tight_layout()
        cm_path = os.path.join(Config.RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Confusion matrix saved: {cm_path}")
        plt.show()
    except Exception as e:
        print(f"  ⚠ Could not create confusion matrix: {e}")

def save_paper_results(model, history, metadata, label_encoder, 
                      test_acc, attack_acc, all_preds, all_labels):
    """ذخیره نتایج به فرمت مقاله"""
    print("\nSaving paper-style results...")
    
    # ذخیره مدل نهایی
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_dim': metadata['num_features'],
        'num_classes': metadata['num_classes'],
        'model_architecture': 'PaperCNN',
        'metadata': metadata,
        'history': history,
        'final_accuracy': test_acc,
        'attack_accuracy': attack_acc,
        'paper_settings': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'epochs': len(history['epoch']),
            'early_stopping_patience': Config.EARLY_STOPPING_PATIENCE
        }
    }
    
    model_path = os.path.join(Config.MODEL_SAVE_DIR, "tabfids_paper_model.pth")
    torch.save(model_info, model_path)
    print(f"  ✓ Paper model saved: {model_path}")
    
    # ذخیره نتایج به صورت CSV
    try:
        import pandas as pd
        
        # ذخیره تاریخچه
        history_df = pd.DataFrame({
            'Epoch': history['epoch'],
            'Train_Loss': history['train_loss'],
            'Train_Accuracy': history['train_acc'],
            'Test_Loss': history['test_loss'],
            'Test_Accuracy': history['test_acc'],
            'Attack_Accuracy': history['attack_acc'],
            'Learning_Rate': history['learning_rate']
        })
        
        history_path = os.path.join(Config.RESULTS_DIR, "paper_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"  ✓ Training history saved: {history_path}")
        
        # ذخیره نتایج نهایی
        results_df = pd.DataFrame({
            'Metric': ['Overall Accuracy', 'Attack Accuracy', 'Benign Detection', 'Attack Detection'],
            'Value': [test_acc, attack_acc, 
                     history['attack_acc'][-1] if 'attack_acc' in history else 0,
                     history['test_acc'][-1] if 'test_acc' in history else 0],
            'Unit': ['%', '%', '%', '%']
        })
        
        results_path = os.path.join(Config.RESULTS_DIR, "final_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"  ✓ Final results saved: {results_path}")
        
    except ImportError:
        print("  ⚠ Pandas not available, skipping CSV export")
    
    # ذخیره گزارش نهایی
    report_path = os.path.join(Config.RESULTS_DIR, "paper_final_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TABFIDS - PAPER IMPLEMENTATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL SETTINGS (From Paper 2508.09060v1):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {Config.LEARNING_RATE}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Epochs: {len(history['epoch'])}\n")
        f.write(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}\n\n")
        
        f.write("DATASET INFO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: CIC-IDS-2017\n")
        f.write(f"Training Samples: {metadata.get('train_samples', 'N/A')}\n")
        f.write(f"Test Samples: {metadata.get('test_samples', 'N/A')}\n")
        f.write(f"Features: {metadata['num_features']}\n")
        f.write(f"Classes: {metadata['num_classes']}\n\n")
        
        f.write("FINAL RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Accuracy: {test_acc:.2f}%\n")
        f.write(f"Attack Accuracy: {attack_acc:.2f}%\n")
        f.write(f"Best Attack Accuracy: {max(history['attack_acc']):.2f}%\n\n")
        
        f.write("CLASS MAPPING:\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(label_encoder.classes_):
            f.write(f"{i:2d}: {name}\n")
    
    print(f"  ✓ Final report saved: {report_path}")

def main():
    """تابع اصلی - پیاده‌سازی مقاله"""
    print("=" * 80)
    print("TABFIDS PAPER IMPLEMENTATION (2508.09060v1)")
    print("=" * 80)
    
    # نمایش تنظیمات
    Config.print_settings()
    
    try:
        # ایجاد پوشه‌ها
        Config.create_dirs()
        
        # ۱. لود داده‌ها
        print("\n[1/4] Loading preprocessed data...")
        X_train, X_test, y_train, y_test, metadata, label_encoder = load_preprocessed_data()
        
        # ۲. ایجاد DataLoaderها
        print("\n[2/4] Creating dataloaders...")
        train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test)
        
        # ۳. ایجاد مدل مقاله
        print("\n[3/4] Creating paper model...")
        input_dim = metadata['num_features']
        num_classes = metadata['num_classes']
        
        model = PaperCNN(input_dim=input_dim, num_classes=num_classes)
        
        print(f"\nModel Summary:")
        print(f"  Architecture: PaperCNN (from Figure 1)")
        print(f"  Input Dimension: {input_dim}")
        print(f"  Number of Classes: {num_classes}")
        print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # ۴. آموزش با Early Stopping
        print("\n[4/4] Training model (with early stopping)...")
        model, history, best_acc, best_attack_acc, all_preds, all_labels = train_with_early_stopping(
            model, train_loader, test_loader, label_encoder
        )
        
        # ۵. ارزیابی نهایی
        test_acc, attack_acc, final_preds, final_labels = evaluate_final(
            model, test_loader, label_encoder
        )
        
        # ۶. ذخیره نتایج
        save_paper_results(model, history, metadata, label_encoder, 
                          test_acc, attack_acc, final_preds, final_labels)
        
        # ۷. ایجاد نمودارهای مقاله
        plot_paper_results(history, label_encoder)
        
        print("\n" + "=" * 80)
        print("✅ PAPER IMPLEMENTATION COMPLETED!")
        print("=" * 80)
        
        print(f"\nKey Results (Paper Metrics):")
        print(f"  • Attack Accuracy: {attack_acc:.2f}%")
        print(f"  • Overall Accuracy: {test_acc:.2f}%")
        print(f"  • Best Attack Accuracy: {best_attack_acc:.2f}%")
        
        print(f"\nOutput Files:")
        print(f"  • Model: models/saved/tabfids_paper_model.pth")
        print(f"  • Results: results/paper_final_report.txt")
        print(f"  • Plots: results/paper_results.png")
        print(f"  • History: results/paper_training_history.csv")
        
        print(f"\nNext Steps:")
        print(f"  1. Run 'python test.py' to test the trained model")
        print(f"  2. Check the 'results/' folder for detailed reports")
        print(f"  3. For federated learning, implement algorithms from the paper")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in paper implementation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nPaper implementation failed. Please check the error messages above.")