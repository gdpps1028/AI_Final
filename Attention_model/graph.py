import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import StrokeCharDataset
from model import StrokeCNN
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 中文顯示用
matplotlib.rcParams['axes.unicode_minus'] = False
## global par
I_want_to_test_how_many_word = 4803
Batch_size = 32
epoch_times = 10

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
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = StrokeCharDataset(train_samples, transform=transform, train=True)
    val_dataset   = StrokeCharDataset(val_samples,   transform=transform, train=False)
    test_dataset  = StrokeCharDataset(test_samples,  transform=transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=Batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=Batch_size, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    model = StrokeCNN(num_classes=len(label_map)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter("runs/stroke_cnn")

    best_val_acc = 0
    best_model_path = "best_model.pt"

    # # 訓練 + 驗證
    # for epoch in range(epoch_times):
    #     model.train()
    #     total_loss = 0
    #     for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
    #         x, y = x.to(device), y.to(device)
    #         out = model(x)
    #         loss = loss_fn(out, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     torch.cuda.empty_cache()

    #     model.eval()
    #     correct = total = 0
    #     with torch.no_grad():
    #         for x, y in tqdm(val_loader, desc="Validating"):
    #             x, y = x.to(device), y.to(device)
    #             preds = model(x).argmax(dim=1)
    #             correct += (preds == y).sum().item()
    #             total += y.size(0)
    #     val_acc = correct / total

    #     writer.add_scalar("Loss/train", total_loss, epoch)
    #     writer.add_scalar("Accuracy/val", val_acc, epoch)
    #     print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    #     # 保存最佳模型
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), best_model_path)

    # 訓練結束後，載入最佳模型，對 Test 集做評估
    print(f"\n>>> 載入最佳模型 ({best_model_path})，在 Test 集上評估")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct = total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, label_char in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            c = label_char.to(device)
            logits_c = model(images)
            preds = logits_c.argmax(dim=1)

            correct += (preds == c).sum().item()
            total += c.size(0)

            all_true.extend(c.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    index_to_char = {idx: char for char, idx in label_map.items()}

    # 3. 統計錯誤的字對
    error_pairs = []
    for t, p in zip(all_true, all_pred):
        if t != p:
            error_pairs.append((index_to_char[t], index_to_char[p]))

    # 4. 畫圖
    pair_counts = Counter(error_pairs)
    top_n = 50
    pairs = [f"{true}→{pred}" for (true, pred), _ in pair_counts.most_common(top_n)]
    counts = [count for _, count in pair_counts.most_common(top_n)]
    print(pairs, counts)
    # plt.figure(figsize=(14, 6))
    # plt.bar(pairs, counts)
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel("真實 → 預測")
    # plt.ylabel("出現次數")
    # plt.title(f"Top {top_n} 最常錯誤的字對")
    # plt.tight_layout()
    # plt.show()
if __name__ == '__main__':
    print("in main()")
    main()
