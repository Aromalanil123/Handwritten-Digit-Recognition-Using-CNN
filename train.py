import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DigitCNN

def get_data_loaders(batch_size=64, data_dir='./data'):
    # Data Augmentation to enhance real handwritten generalization
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Step [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%")
            
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            
    test_loss /= total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc

def train(epochs=3, batch_size=64, lr=0.001, save_path='models/cnn_digit_model.pt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training CNN on MNIST using device: {device} ---")
    
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': epochs,
        'model_parameters': model.count_parameters(),
        'best_test_acc': 0.0
    }
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_acc)
        
        history['train_loss'].append(round(train_loss, 4))
        history['train_acc'].append(round(train_acc, 2))
        history['test_loss'].append(round(test_loss, 4))
        history['test_acc'].append(round(test_acc, 2))
        
        print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint to {save_path} (Test Acc: {best_acc:.2f}%)")
            
    history['best_test_acc'] = round(best_acc, 2)
    history['training_time_sec'] = round(time.time() - start_time, 1)
    
    history_path = os.path.join(os.path.dirname(save_path), 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"\n[DONE] Training finished in {history['training_time_sec']}s. Final Best Accuracy: {best_acc:.2f}%\n")
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN on MNIST")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--save-path', type=str, default='models/cnn_digit_model.pt', help="Path to save model weights")
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
