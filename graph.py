import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from loguru import logger
from . import CNN, utils
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1. 載入資料並切割
    images, labels = utils.load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)

    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.2, random_state=777, stratify=labels
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=777, stratify=temp_labels
    )

    # 2. 還原 label 對照表
    dummy_ds = utils.TrainDataset(train_imgs + val_imgs, train_labels + val_labels, transform=None)
    index_to_label = dummy_ds.index_to_label

    # # 3. 建立 transform（與訓練時的 val_transform 相同）
    # val_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.Resize((48, 48)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    # ])

    # # 4. 準備 test dataset & dataloader
    # test_ds = utils.TestDataset(test_imgs, transform=val_transform)
    # test_ld = DataLoader(test_ds, batch_size=32, shuffle=False)

    # # 5. 載入模型並推論
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNN.CNN(len(index_to_label)).to(device)
    # model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
    # model.eval()

    # # 6. 推論與記錄結果
    # CNN.test_character(
    #     model,
    #     test_ld,
    #     criterion=nn.CrossEntropyLoss(),
    #     device=device,
    #     original_labels=index_to_label,
    #     save_path="CNN.csv"
    # )

    # 7. 計算測試集準確率
    df = pd.read_csv("CNN.csv")
    df['true'] = test_labels
    df['correct'] = df['true'] == df['char']
    total = len(df)
    corrects = df['correct'].sum()
    acc = corrects / total
    print(f"Test size={total}, correct={corrects}, accuracy={acc:.4%}")
    pairs = [f"{true}→{pred}" for true, pred in pair_counts.keys()]
    counts = list(pair_counts.values())

    plt.figure(figsize=(12, 6))
    plt.bar(pairs, counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("True → Predicted")
    plt.ylabel("Count")
    plt.title("Most Common Misclassifications")
    plt.tight_layout()
    plt.show()