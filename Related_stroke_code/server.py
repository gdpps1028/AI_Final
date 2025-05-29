# app.py
from flask import Flask, request, jsonify, send_from_directory
import base64, io, os
from PIL import Image
import torch
from torchvision import transforms
from model import StrokeCNN  # 請確保 model.py 在同一目錄
# 請先在同目錄下放好 best_model.pt 與 label_map.json

app = Flask(__name__, static_folder='static')

# 1. 載入 label_map（把 index 轉回中文字）
import json
with open('label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)       # e.g. ["丁","大",...]
label_list = label_map            # index → 字

# 2. 載入模型
device = torch.device("cpu")
model = StrokeCNN(num_classes=len(label_list))
state = torch.load('best_model.pt', weights_only=True, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# 3. 圖片前處理
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4. 提供前端靜態檔案 (index.html 放在 static/)
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# 5. 推論 API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('image', None)
    if data is None:
        return jsonify({'error': 'No image provided'}), 400

    # Base64 解碼
    header, b64 = data.split(',', 1)
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # 轉灰階

    # 推論
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()
        confidence = probs[idx].item()

    label = label_list[idx]
    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    # 在 http://localhost:5000 啟動
    app.run(host='0.0.0.0', port=5000, debug=True)