from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Character_Only import CNN, utils
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
original_labels_map = None

def load_model_and_labels():
    global model, original_labels_map
    print("Loading training labels for mapping...")
    _, all_training_labels = utils.load_train_dataset()
    unique_labels = sorted(list(set(all_training_labels)))
    original_labels_map = {i: label for i, label in enumerate(unique_labels)}
    print(f"Loaded {len(unique_labels)} unique labels.")

    num_unique_labels = len(unique_labels)
    model_path = 'Character_Only/best_cnn_model.pth'

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    model = CNN.CNN(num_unique_labels).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model state dict: {e}")
        model = None

with app.app_context():
    load_model_and_labels()
    if model is None:
        print("Application will not run as model failed to load.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    img_data_b64 = request.json['image'].split(',')[1]
    img_data = base64.b64decode(img_data_b64)

    img = Image.open(io.BytesIO(img_data))

    print(f"DEBUG: Original image mode: {img.mode}, size: {img.size}")

    # --- FINALIZED IMAGE PROCESSING CONFIGURATION ---
    # This now precisely matches your training pipeline:
    # 1. Convert to Grayscale (L mode).
    # 2. Resize to (50, 50).
    # 3. Convert to Tensor (which will then be Grayscale to 3 channels via num_output_channels).
    # Note: We'll put Grayscale(num_output_channels=3) directly into the Compose.

    target_image_size = (50, 50) # Confirmed correct

    transform_list = [
        # Step 1: Convert to Grayscale, then output 3 identical channels.
        # This is crucial for matching your model's input expectations.
        transforms.Grayscale(num_output_channels=3),
        # Step 2: Resize the image.
        transforms.Resize(target_image_size),
        # Step 3: Convert to a PyTorch Tensor (scales pixels to [0.0, 1.0]).
        transforms.ToTensor(),
        # No inversion needed (confirmed dark strokes on light background).
        # No Normalization needed (confirmed not used in training pipeline).
    ]

    transform = transforms.Compose(transform_list)
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension and move to device

    print(f"DEBUG: Tensor shape before model: {img_tensor.shape}")
    print(f"DEBUG: Example pixel value (top-left, channel 0): {img_tensor[0, 0, 0, 0].item()}")

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)

    top_n = 5
    if probabilities.numel() == 0:
        print("DEBUG: Probabilities tensor is empty. No predictions can be made.")
        return jsonify([])

    actual_top_n = min(top_n, probabilities.size(0))
    if actual_top_n <= 0:
        return jsonify([])

    top_probabilities, top_indices = torch.topk(probabilities, actual_top_n)

    predictions_list = []
    for i in range(actual_top_n):
        label_idx = top_indices[i].item()
        confidence = top_probabilities[i].item()
        predicted_label = original_labels_map.get(label_idx, "Unknown")
        predictions_list.append({
            'label': predicted_label,
            'confidence': confidence
        })

    print(f"DEBUG: Returning predictions: {predictions_list}")
    return jsonify(predictions_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)