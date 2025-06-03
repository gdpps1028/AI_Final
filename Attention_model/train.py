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

## global par
I_want_to_test_how_many_word = 4803
Batch_size = 32
epoch_times = 10

# global seed
seed = 777
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def load_dataset(data_root, items=4803):
    label_map = {}
    samples = []
    label_id = 0

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
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

    # 80/20 切
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels
    )

    # 20 再切 10/10
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

    # 訓練 + 驗證
    for epoch in range(epoch_times):
        # ---- 1. train 一整個 epoch + 計算 train_loss ----
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        # 計算這個 epoch 的 train_loss 平均值
        avg_train_loss = train_loss_sum / train_steps

        # ---- 2. validation: 計算 val_loss 以及 val_acc ----
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        correct = total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validating]"):
                x, y = x.to(device), y.to(device)
                out = model(x)

                # 計算這個 batch 的 loss
                loss_val = loss_fn(out, y)
                val_loss_sum += loss_val.item()
                val_steps += 1

                # 計算這個 batch 的正確率
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        # 平均 validation loss
        avg_val_loss = val_loss_sum / val_steps
        # validation accuracy
        val_acc = correct / total

        # ---- 3. TensorBoard & printing ----
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val",   avg_val_loss,   epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"[Epoch {epoch+1:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # 保存最佳模型（依 val_acc 判定）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # 訓練結束後，載入最佳模型，對 Test 集做評估
    print(f"\n>>> 載入最佳模型 ({best_model_path})，在 Test 集上評估")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    print("Attention model in main")
    main()
