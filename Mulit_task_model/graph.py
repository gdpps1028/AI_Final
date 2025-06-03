import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MutDataSet
from model import MultiTaskCNN
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 中文顯示用
matplotlib.rcParams['axes.unicode_minus'] = False

## global par
I_want_to_test_how_many_word = 4803
Batch_size = 32
epoch_times = 10
FileTag = 1
# gloabl
seed = 777
random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

def load_dataset(data_root, items = 4803):
    label_map = {}
    samples = []
    label_id = 0

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path): continue
        char = folder.split('_')[0]
        if char not in label_map:
            if len(label_map) >= items:
                break
            label_map[char] = label_id
            label_id += 1
        for fname in os.listdir(folder_path):
            samples.append((os.path.join(folder_path, fname), label_map[char]))
    return samples, label_map

def main():
    data_root = "../data_new"
    all_samples, label_map = load_dataset(data_root, items=I_want_to_test_how_many_word)
    index_to_char = { idx: char for char, idx in label_map.items() }

    radicals = set()
    for img_path, _ in all_samples:
        folder = os.path.basename(os.path.dirname(img_path))  # e.g. "好_女_6"
        _, radical_s, _ = folder.split('_')
        radicals.add(radical_s)

    radical_to_index = {rad: idx for idx, rad in enumerate(sorted(radicals))}
    num_radicals     = len(radical_to_index)
    num_radicals = len(radicals)

    print(f"使用 {len(label_map)} 個類別，共 {len(all_samples)} 張圖片")
    images = [p for p, l in all_samples]
    labels = [l for p, l in all_samples]

    # 80/20切
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=seed,  
        stratify=labels
    )

    # 20再切10/10
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels
    )
    train_samples = list(zip(train_imgs, train_labels))
    val_samples   = list(zip(val_imgs,   val_labels))
    test_samples  = list(zip(test_imgs,  test_labels))
    print(f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(5),
        transforms.RandomApply(
            [
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ],
            p = 0.1
        ),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = MutDataSet(train_samples, index_to_char, 
                                radical_to_index, transform=transform,
                               train=True)
    val_dataset   = MutDataSet(val_samples,   index_to_char,
                               radical_to_index, transform=val_transform, 
                               train=False)
    test_dataset  = MutDataSet(test_samples,  index_to_char,
                               radical_to_index, transform=val_transform, 
                               train=False)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=Batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=Batch_size, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    print(num_radicals)
    model = MultiTaskCNN(num_classes=len(label_map), num_radicals=num_radicals).to(device)

    best_model_path = "best_model.pt"
    # 訓練結束後，載入最佳模型，對 Test 集做評估
    print(f"\n>>> 載入最佳模型 ({best_model_path})，在 Test 集上評估")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct = total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, label_char, _, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            c = label_char.to(device)
            logits_c, _, _ = model(images)
            preds = logits_c.argmax(dim=1)

            correct += (preds == c).sum().item()
            total += c.size(0)

            all_true.extend(c.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # 2. 取得 index_to_char 的反查字典
    index_to_char = {idx: char for char, idx in label_map.items()}

    # 3. 統計錯誤的字對
    error_pairs = []
    for t, p in zip(all_true, all_pred):
        if t != p:
            error_pairs.append((index_to_char[t], index_to_char[p]))

    # 4. 畫圖
    pair_counts = Counter(error_pairs)
    top_n = 20
    pairs = [f"{true}→{pred}" for (true, pred), _ in pair_counts.most_common(top_n)]
    counts = [count for _, count in pair_counts.most_common(top_n)]

    plt.figure(figsize=(14, 6))
    plt.bar(pairs, counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("真實 → 預測")
    plt.ylabel("出現次數")
    plt.title(f"Top {top_n} 最常錯誤的字對")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    print("Multi_task test in main()")
    main()
