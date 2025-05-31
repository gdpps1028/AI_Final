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

EPOCHS  = 10
BASE_DIR = os.path.dirname(__file__)  
CSV_PATH   = os.path.join(BASE_DIR, 'CNN.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'best_cnn_model.pth')
Batch_size = 32
def train_main(train_imgs, train_labels, val_imgs, val_labels):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(5),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))],
            p=0.1
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
    train_ds = utils.TrainDataset(train_imgs, train_labels, transform=transform)
    val_ds   = utils.TrainDataset(val_imgs,   val_labels, transform=val_transform)
    train_ld  = DataLoader(train_ds, batch_size=Batch_size, shuffle=True)
    val_ld    = DataLoader(val_ds,   batch_size=Batch_size, shuffle=False)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_classes= len(train_ds.unique_labels)
    model      = CNN.CNN(num_classes).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=1e-4)

    max_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss = CNN.train(model, train_ld, criterion, optimizer, device)
        val_loss, val_acc = CNN.validate(model, val_ld, criterion, device)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), os.path.join(BASE_DIR, 'best_cnn_model.pth'))
    logger.info(f"Training done, best val_acc={max_acc:.4f}")
    return train_ds.index_to_label  

def test_main(test_imgs, test_labels, index_to_label):
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    test_ds = utils.TestDataset(test_imgs, transform=val_transform)
    test_ld    = DataLoader(test_ds, batch_size=Batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNN.CNN(len(index_to_label)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # 生成 CSV
    CNN.test_character(
        model,
        test_ld,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        original_labels=index_to_label,
        save_path=CSV_PATH
    )

    df = pd.read_csv(CSV_PATH)
    df['true']    = test_labels
    df['correct'] = df['true'] == df['char']
    total    = len(df)
    corrects = df['correct'].sum()
    acc      = corrects / total
    print(f"Test size={total}, correct={corrects}, accuracy={acc:.4%}")

if __name__ == '__main__':
    # 做切分
    images, labels = utils.load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    # 80% train, 20% temp
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.2, random_state=777, stratify=labels
    )
    # temp 再切 50/50 -> val/test 各 10%
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=777, stratify=temp_labels
    )

    # 傳 train, val 給test_main
    index_to_label = train_main(train_imgs, train_labels, val_imgs, val_labels)
    # 傳test_case給 test_main 
    test_main(test_imgs, test_labels, index_to_label)
