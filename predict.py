import os
import torch
import torch.nn.functional as F
from model import DigitCNN
from utils import preprocess_canvas_image
from PIL import Image

class DigitPredictor:
    def __init__(self, model_path='models/cnn_digit_model.pt', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitCNN().to(self.device)
        self.model_path = model_path
        self.loaded = False
        self.load_weights()
        
    def load_weights(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.loaded = True
            print(f"Loaded weights from {self.model_path}")
        else:
            print(f"Warning: Model weights not found at {self.model_path}. Model will need to be trained first.")
            self.model.eval()

    def predict(self, image_data):
        """
        Accepts base64 or PIL Image, preprocesses it, and runs inference.
        Returns dict with:
          - prediction: int (0-9)
          - confidence: float (0.0 to 100.0)
          - probabilities: list of floats for digits 0-9
          - preview: base64 of centered 28x28 normalized image
          - is_empty: bool (if canvas was empty)
        """
        tensor, preview_base64, has_content = preprocess_canvas_image(image_data)
        
        if not has_content:
            return {
                'prediction': None,
                'confidence': 0.0,
                'probabilities': [0.0] * 10,
                'preview': preview_base64,
                'is_empty': True
            }
            
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        pred_digit = int(probs.argmax())
        confidence = float(probs[pred_digit] * 100.0)
        prob_list = [float(round(p * 100.0, 2)) for p in probs]
        
        return {
            'prediction': pred_digit,
            'confidence': round(confidence, 2),
            'probabilities': prob_list,
            'preview': preview_base64,
            'is_empty': False
        }

if __name__ == '__main__':
    predictor = DigitPredictor()
    print("DigitPredictor initialized.")
