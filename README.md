# Handwritten Digit Recognition Using CNN (PyTorch)

A production-grade Deep Learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with **PyTorch**, featuring an interactive web canvas and real-time inference.

---

## 🚀 Key Features

- **Dual-Block Deep CNN**: Built in PyTorch with Conv2D, Batch Normalization, Max Pooling, and Dropout to achieve **>99% test accuracy** on MNIST while avoiding overfitting.
- **Center-of-Mass Preprocessing**: Human-drawn digits have varying stroke thicknesses, scales, and offsets. The preprocessing engine automatically centers by center-of-mass, scales preserving aspect ratio, and normalizes according to MNIST dataset parameters.
- **Real-Time Interactive Canvas**: Smooth HTML5 canvas with mouse and touch/stylus support, adjustable brush sizes, undo history, and instant digit classification.
- **Live Softmax Probabilities**: Visualizes the top prediction and probability distribution bars across all 10 digit classes (0–9).
- **Interactive Preset Test Digits**: Quick-test buttons (0–9) that programmatically draw strokes on the canvas for instant verification.
- **What the CNN Sees (28×28 Inspector)**: Live grayscale preview displaying the exact normalized tensor fed into the neural network.

---

## 🧠 CNN Architecture

```
Input: [Batch, 1, 28, 28]
  │
  ├── [Conv2D 1->32, 3x3, pad=1] ──> BatchNorm2D ──> ReLU
  ├── [Conv2D 32->32, 3x3, pad=1] ──> BatchNorm2D ──> ReLU
  ├── [MaxPool2D 2x2, stride=2] ──> [Batch, 32, 14, 14]
  ├── [Dropout2D 0.25]
  │
  ├── [Conv2D 32->64, 3x3, pad=1] ──> BatchNorm2D ──> ReLU
  ├── [Conv2D 64->64, 3x3, pad=1] ──> BatchNorm2D ──> ReLU
  ├── [MaxPool2D 2x2, stride=2] ──> [Batch, 64, 7, 7]
  ├── [Dropout2D 0.25]
  │
  ├── Flatten (64 * 7 * 7 = 3,136 features)
  ├── [Linear 3136 -> 128] ──> BatchNorm1D ──> ReLU ──> [Dropout 0.4]
  └── [Linear 128 -> 10] ──> Output Logits (Softmax)
```

**Total Trainable Parameters:** 430,730

---

## 📁 Project Structure

```
digit-recognizer-cnn/
├── app.py                     # Flask web server and REST API
├── model.py                   # PyTorch CNN architecture definition
├── train.py                   # Training pipeline with augmentation & evaluation
├── predict.py                 # Standalone inference class
├── utils.py                   # Image center-of-mass normalization pipeline
├── requirements.txt           # Python dependencies
├── models/
│   ├── cnn_digit_model.pt     # Trained PyTorch model weights
│   └── training_history.json  # Loss & accuracy metrics per epoch
├── static/
│   ├── css/style.css          # Glassmorphic dark UI stylesheet
│   └── js/canvas.js           # Interactive canvas drawing & API client
└── templates/
    └── index.html             # Web dashboard
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the CNN (Optional - Pre-trained weights included)
```bash
python train.py --epochs 3 --batch-size 64
```

### 3. Launch the Interactive Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.

---

## 🎯 How Inference Works

1. **User draws** on the 280×280 interactive canvas.
2. The drawing is transmitted as a base64 PNG to `/api/predict`.
3. `utils.preprocess_canvas_image()`:
   - Inverts colors if drawn on light background (MNIST expects white digit on black background).
   - Crops the bounding box around active strokes.
   - Resizes into a 20×20 bounding box with preserved aspect ratio.
   - Shifts the 20×20 digit onto a 28×28 canvas centered at the **center-of-mass**.
   - Normalizes pixels using `(x - 0.1307) / 0.3081`.
4. PyTorch performs a forward pass through `DigitCNN` and applies `torch.nn.functional.softmax`.
5. The result, confidence percentage, full probability vector, and 28×28 processed preview return to update the dashboard.
