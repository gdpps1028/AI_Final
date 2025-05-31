import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify,  send_from_directory
from PIL import Image
from io import BytesIO
import base64
import json
from model import StrokeCNN  # 確保你這邊的模型架構和訓練時一致
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from importlib import import_module

def center_pad(image, size=(64, 64)):
    w, h = image.size
    scale = min(size[0]/w, size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new("L", size, 255)
    paste_x = (size[0] - new_w) // 2
    paste_y = (size[1] - new_h) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img

# 載入 label_map
with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
label_list = [char for char in label_map]

# 模型初始化
model = StrokeCNN(num_classes=len(label_list))
model.load_state_dict(torch.load("best_model.pt", map_location="cpu", weights_only=True))
model.eval()

# 前處理步驟（與訓練一致）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ← 這行確保輸出是 3 channels (因為已設定模型要3 channels)
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Flask app 初始化
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

def crop_union_bb(pil_img, thresh=180):
    # 裁減圖片
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binar = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    num, _, stats, _ = cv2.connectedComponentsWithStats(binar, connectivity=8)
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    x0, y0 = xs.min(), ys.min()
    x1 = (xs + ws).max()
    y1 = (ys + hs).max()
    return pil_img.crop((x0, y0, x1, y1))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_b64 = data["image"].split(",")[1]  
    image_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_data)).convert("L")
    img = crop_union_bb(img)
    img = center_pad(img)              
    img = transform(img).unsqueeze(0)  # 變成 [1, C, H, W]
    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)[0]  # shape: [num_classes]
        top5 = torch.topk(probs, k=5)
        indices = top5.indices.tolist()
        values = top5.values.tolist()
        result = [(label_list[i], v) for i, v in zip(indices, values)]

    return jsonify({
        "top": result[0][0],
        "all": result
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)