import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for Handwritten Digit Recognition (MNIST).
    
    Architecture:
      Conv1 (32 filters, 3x3) -> BatchNorm -> ReLU
      Conv2 (32 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
      Conv3 (64 filters, 3x3) -> BatchNorm -> ReLU
      Conv4 (64 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
      Fully Connected 1 (3136 -> 128) -> BatchNorm -> ReLU -> Dropout(0.4)
      Fully Connected 2 (128 -> 10) -> Output Logits
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        
        # Block 1: 1x28x28 -> 32x14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Block 2: 32x14x14 -> 64x7x7
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Dense classification head: 64*7*7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = DigitCNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Model instantiated successfully.")
    print(f"Output shape: {output.shape}")
    print(f"Total trainable parameters: {model.count_parameters():,}")
