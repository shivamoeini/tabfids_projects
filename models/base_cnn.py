import torch
import torch.nn as nn
import torch.nn.functional as F

class IDS_CNN(nn.Module):
    """Convolutional Neural Network for IDS"""
    
    def __init__(self, input_dim=78, num_classes=11, hidden_dim=128):
        super(IDS_CNN, self).__init__()
        
        # Reshape input to 2D for CNN (assuming 1D time series-like data)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size(input_dim)
        
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def _get_flattened_size(self, input_dim):
        # Simulate forward pass to get flattened size
        x = torch.randn(1, 1, input_dim)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.numel()
    
    def forward(self, x):
        # Reshape to (batch, channels, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TabFIDSModel(nn.Module):
    """Main TabFIDS model with configurable backbone"""
    
    def __init__(self, backbone_type="CNN", input_dim=78, num_classes=11):
        super(TabFIDSModel, self).__init__()
        
        self.backbone_type = backbone_type
        
        if backbone_type == "CNN":
            self.backbone = IDS_CNN(input_dim, num_classes)
        elif backbone_type == "ResNet":
            # Implement ResNet backbone
            self.backbone = self._build_resnet(input_dim, num_classes)
        elif backbone_type == "Autoencoder":
            # Implement Autoencoder backbone
            self.backbone = self._build_autoencoder(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    def _build_resnet(self, input_dim, num_classes):
        """Build ResNet backbone"""
        # Simplified ResNet implementation
        class ResBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(ResBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
                self.bn2 = nn.BatchNorm1d(out_channels)
                self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
                
            def forward(self, x):
                identity = self.shortcut(x)
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
                return F.relu(out)
        
        class ResNetBackbone(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(ResNetBackbone, self).__init__()
                self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
                self.bn1 = nn.BatchNorm1d(32)
                self.pool = nn.MaxPool1d(2)
                
                self.res1 = ResBlock(32, 64)
                self.res2 = ResBlock(64, 128)
                self.res3 = ResBlock(128, 256)
                
                # Calculate flattened size
                self.flattened_size = 256 * (input_dim // 8)
                
                self.fc1 = nn.Linear(self.flattened_size, 128)
                self.fc2 = nn.Linear(128, num_classes)
                
            def forward(self, x):
                x = x.unsqueeze(1)
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.res1(x)
                x = self.pool(x)
                x = self.res2(x)
                x = self.pool(x)
                x = self.res3(x)
                
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return ResNetBackbone(input_dim, num_classes)
    
    def _build_autoencoder(self, input_dim, num_classes):
        """Build Autoencoder backbone"""
        class AutoencoderBackbone(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(AutoencoderBackbone, self).__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU()
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim),
                    nn.Sigmoid()
                )
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, num_classes)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                classified = self.classifier(encoded)
                return classified, decoded
        
        return AutoencoderBackbone(input_dim, num_classes)
    
    def forward(self, x):
        if self.backbone_type == "Autoencoder":
            classification_output, reconstruction = self.backbone(x)
            return classification_output, reconstruction
        else:
            return self.backbone(x)