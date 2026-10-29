import base64
import io
import math
import numpy as np
from PIL import Image, ImageOps
import torch

def preprocess_canvas_image(image_data):
    """
    Takes a base64 string or PIL Image from the user's canvas drawing
    and applies standard MNIST-style preprocessing:
      1. Grayscale conversion and contrast check
      2. Invert colors if drawn black-on-white (MNIST expects white digit on black background)
      3. Bounding box cropping around the drawn digit
      4. Aspect-ratio preserving resize to fit inside 20x20 box
      5. Placement on 28x28 canvas centered by center-of-mass (moment of inertia)
      6. PyTorch normalization matching MNIST dataset (mean=0.1307, std=0.3081)
    
    Returns:
      tensor: torch.FloatTensor of shape (1, 1, 28, 28)
      preview_base64: data URL string of the 28x28 preprocessed image for UI debugging
    """
    # 1. Parse image input
    if isinstance(image_data, str):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_data, Image.Image):
        img = image_data
    else:
        raise ValueError("Unsupported image data type")

    # Convert to grayscale
    img = img.convert('L')
    
    # Check if the canvas background is predominantly white (e.g., black strokes on white canvas)
    # MNIST requires white digit (255) on black background (0).
    np_img = np.array(img, dtype=np.float32)
    if np.mean(np_img) > 128:
        # Background is light, invert so digit is light
        img = ImageOps.invert(img)
        np_img = 255.0 - np_img

    # 2. Check if the image has any drawing
    if np_img.max() < 25:
        # Canvas is essentially blank
        blank_tensor = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        blank_tensor = (blank_tensor - 0.1307) / 0.3081
        blank_preview = _image_to_base64(Image.new('L', (28, 28), color=0))
        return blank_tensor, blank_preview, False

    # 3. Find bounding box around digit strokes
    non_empty_y, non_empty_x = np.where(np_img > 30)
    if len(non_empty_y) == 0:
        blank_tensor = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        blank_preview = _image_to_base64(Image.new('L', (28, 28), color=0))
        return blank_tensor, blank_preview, False

    ymin, ymax = non_empty_y.min(), non_empty_y.max()
    xmin, xmax = non_empty_x.min(), non_empty_x.max()
    
    # Crop to bounding box
    cropped = img.crop((xmin, ymin, xmax + 1, ymax + 1))
    
    # 4. Resize to fit inside 20x20 preserving aspect ratio
    w, h = cropped.size
    if w > h:
        scale = 20.0 / w
        new_w = 20
        new_h = max(1, int(round(h * scale)))
    else:
        scale = 20.0 / h
        new_h = 20
        new_w = max(1, int(round(w * scale)))
        
    resized_digit = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 5. Paste onto blank 28x28 canvas centered initially
    canvas28 = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas28.paste(resized_digit, (paste_x, paste_y))
    
    # 6. Center of Mass adjustment (the MNIST standard)
    np_canvas = np.array(canvas28, dtype=np.float32)
    total_mass = np_canvas.sum()
    if total_mass > 0:
        # Calculate center of mass
        y_indices, x_indices = np.indices((28, 28))
        cy = (y_indices * np_canvas).sum() / total_mass
        cx = (x_indices * np_canvas).sum() / total_mass
        
        # Calculate shift required to place center of mass at (13.5, 13.5)
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))
        
        # Limit shift to stay within bounds
        shift_x = max(-4, min(4, shift_x))
        shift_y = max(-4, min(4, shift_y))
        
        if shift_x != 0 or shift_y != 0:
            shifted = np.roll(np_canvas, shift_y, axis=0)
            if shift_y > 0:
                shifted[:shift_y, :] = 0
            elif shift_y < 0:
                shifted[shift_y:, :] = 0
                
            shifted = np.roll(shifted, shift_x, axis=1)
            if shift_x > 0:
                shifted[:, :shift_x] = 0
            elif shift_x < 0:
                shifted[:, shift_x:] = 0
                
            np_canvas = shifted
            canvas28 = Image.fromarray(np_canvas.astype(np.uint8), mode='L')

    # Convert to preview base64
    preview_base64 = _image_to_base64(canvas28)

    # 7. Convert to PyTorch Tensor and normalize
    # Scale pixel intensities to [0, 1]
    tensor = torch.from_numpy(np_canvas).float() / 255.0
    # Standard MNIST Normalization
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
    
    return tensor, preview_base64, True

def _image_to_base64(img):
    buffered = io.BytesIO()
    # Upscale slightly for crisp display in browser
    display_img = img.resize((140, 140), Image.Resampling.NEAREST)
    display_img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
