# project_root/scripts/cam_vis.py
import argparse, os, cv2, torch, numpy as np
from torchvision import transforms
from torchcam.methods import GradCAM

# ------------------ CLI 參數 ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True)
parser.add_argument('--base_ckpt', required=True)
parser.add_argument('--att_ckpt', required=True)
parser.add_argument('--out_dir', default='output_cam')
parser.add_argument('--label',   default='sample')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ------------------ 載入模型 ------------------
from CNN   import CNN        as BaseNet
from model import StrokeCNN  as AttNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_base = BaseNet(num_classes=4803).to(device)
model_att  = AttNet (num_classes=4803).to(device)
model_base.load_state_dict(torch.load(args.base_ckpt, map_location=device))
model_att .load_state_dict(torch.load(args.att_ckpt , map_location=device))
model_base.eval(); model_att.eval()

# ------------------ 前處理 ------------------
img_bgr = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)            # → 3-channel
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # 三通道
])
inp = to_tensor(img_rgb).unsqueeze(0).to(device)

# ------------------ 工具：overlay ------------------
def overlay(img_gray, cam, save_path):
    cam_uint8 = (cam * 255).astype(np.uint8)
    if cam_uint8.ndim == 3:          # (H,W,C) → squeeze 單通道
        cam_uint8 = cam_uint8.squeeze()
    heat = cv2.applyColorMap(cam_uint8.copy(), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                            0.5, heat, 0.5, 0)
    cv2.imwrite(save_path, blend)

# ------------------ 取得 CAM ------------------
def get_cam(model, x, layer='conv2'):
    cam_extractor = GradCAM(model, target_layer=layer)
    logits = model(x)['char'] if isinstance(model(x), dict) else model(x)
    pred   = logits.argmax(1).item()
    cam    = cam_extractor(pred, scores=logits)[0].squeeze().cpu().numpy()
    cam    = cv2.resize(cam, (x.shape[3], x.shape[2]))
    cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
    return cam, pred

cam_base, _ = get_cam(model_base, inp, 'conv2')
cam_att , _ = get_cam(model_att , inp, 'conv2')

# ------------------ 存原始兩張熱圖 ------------------
overlay(img_bgr, cam_base, os.path.join(args.out_dir, f'baseline_{args.label}.png'))
overlay(img_bgr, cam_att , os.path.join(args.out_dir, f'att_{args.label}.png'))

# ------------------ Attention - Baseline 差異熱圖 ------------------
diff = np.clip(cam_att - cam_base, 0, None)
if diff.max() > 0:
    diff = diff / diff.max()
overlay(img_bgr, diff, os.path.join(args.out_dir, f'diff_{args.label}.png'))

print('Grad-CAM 與差異圖已存至', args.out_dir)
# ------------------ 反向 ------------------
diff_neg = np.clip(cam_base - cam_att, 0, None)
if diff_neg.max() > 0:
    diff_neg /= diff_neg.max()
overlay(img_bgr, diff_neg,
        os.path.join(args.out_dir, f'diff_neg_{args.label}.png'))