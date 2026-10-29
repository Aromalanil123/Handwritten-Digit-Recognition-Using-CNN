import os
import torch
from torchvision import datasets, transforms
from model import DigitCNN
import numpy as np

def run_benchmark(model_path='models/cnn_digit_model.pt'):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    confusion_matrix = np.zeros((10, 10), dtype=int)

    print(f"\n--- Benchmarking CNN Model on 10,000 MNIST Test Images ({device}) ---")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for p, l in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                confusion_matrix[l][p] += 1
                class_total[l] += 1
                if p == l:
                    class_correct[l] += 1

    overall_acc = 100.0 * correct / total
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}% ({correct}/{total})\n")
    print(f"{'Digit':<8}{'Accuracy':<12}{'Correct':<10}{'Total':<10}")
    print("-" * 40)
    for i in range(10):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{i:<8}{acc:.2f}%{'':<6}{class_correct[i]:<10}{class_total[i]:<10}")

    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    header = "    " + " ".join([f"{i:>4}" for i in range(10)])
    print(header)
    print("    " + "-" * 50)
    for i in range(10):
        row_str = f"{i} | " + " ".join([f"{confusion_matrix[i][j]:>4}" for j in range(10)])
        print(row_str)
    print()

if __name__ == '__main__':
    run_benchmark()
