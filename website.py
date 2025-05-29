from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Character_Only import CNN, utils

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
original_labels_map = None

def load_model_and_labels():
    global model, original_labels_map

    _, all_training_labels = utils.load_train_dataset()
    unique_labels = sorted(list(set(all_training_labels)))
    original_labels_map = {i: label for i, label in enumerate(unique_labels)}

    num_unique_labels = len(unique_labels)

    model = CNN.CNN(num_unique_labels).to(device)
    model.load_state_dict(torch.load('Character_Only/best_cnn_model.pth', map_location=device))
    model.eval()

with app.app_context():
    load_model_and_labels()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    img_data_b64 = request.json['image'].split(',')[1]
    img_data = base64.b64decode(img_data_b64)

    img = Image.open(io.BytesIO(img_data))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 請再次確認這個尺寸 (48, 48) 是否與您模型訓練時的輸入尺寸一致
    # 這非常重要，如果尺寸不符，模型會給出無效的預測
    target_image_size = (50, 50) # <--- YOU MUST VERIFY THIS WITH YOUR TRAINING!

    transform = transforms.Compose([
        transforms.Resize(target_image_size),
        transforms.ToTensor(),
        # Add normalization if your CNN was trained with it
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        # 這裡的 .squeeze(0) 是必要的，因為 softmax 後會多一個 batch 維度
        probabilities = torch.softmax(output, dim=1).squeeze(0)

    top_n = 5 # 顯示前 5 個預測
    # 檢查 probabilities 是否為空或有效
    if probabilities.numel() == 0: # 如果 tensor 為空，直接返回空列表
        print("DEBUG: Probabilities tensor is empty. No predictions can be made.")
        return jsonify([])

    # 確保 top_n 不會大於實際的類別數量
    actual_top_n = min(top_n, probabilities.size(0))
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

    # Debugging: 打印即將返回的數據
    print(f"DEBUG: Returning predictions: {predictions_list}")

    return jsonify(predictions_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)