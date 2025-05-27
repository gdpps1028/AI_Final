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
Batch_size = 8

def set_seed(seed=42):
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
    set_seed()
    data_root = "../data_new"
    all_samples, label_map = load_dataset(data_root, items = I_want_to_test_how_many_word)
    print(f"使用{len(label_map)}個類別. 共{len(all_samples)}張圖片")
    # exit()
    train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42, stratify=[s[1] for s in all_samples])
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    ])
    train_dataset = StrokeCharDataset(train_samples, transform=transform, train=True)
    val_dataset = StrokeCharDataset(val_samples, transform=transform, train=False)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置:{device}")
    model = StrokeCNN(num_classes=len(label_map)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter("runs/stroke_cnn")

    best_val_acc = 0

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        torch.cuda.empty_cache()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Validating"):
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

if __name__ == '__main__':
    print("in main()")
    main()