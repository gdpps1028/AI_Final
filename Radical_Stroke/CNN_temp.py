import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from loguru import logger
from sklearn.metrics import accuracy_score

from Radical_Stroke import CNN, utils

def train_main_rs():
    """
    load data
    """
    logger.info("Start loading data")
    images, labels = utils.load_train_dataset()
    unique_labels = set(labels)
    num_unique_labels = len(unique_labels)
    images, labels = shuffle(images, labels, random_state=777)

    # New 80/10/10 split
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.2, random_state=777, stratify=labels
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=777, stratify=temp_labels
    )

    train_dataset = utils.TrainDataset(train_imgs, train_labels)
    val_dataset = utils.TrainDataset(val_imgs, val_labels)

    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN.CNN(num_unique_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []
    max_acc = 0

    EPOCHS = 10
    for epoch in range(EPOCHS):
        train_loss = CNN.train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = CNN.validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > max_acc:
            max_acc = val_acc
            print("Saved better model")
            torch.save(model.state_dict(), 'Radical_Stroke/best_cnn_model.pth')

    logger.info(f"Best Accuracy: {max_acc:.4f}")
    

def train_main_character():
    images, labels = utils.load_train_dataset()
    rss = set(labels)
    for i, rs in enumerate(rss):
        logger.info("Start loading data")
        images, labels = utils.load_train_dataset(rs)
        unique_labels = set(labels)
        num_unique_labels = len(unique_labels)
        images, labels = shuffle(images, labels, random_state=777)

        # New 80/10/10 split
        if len(images) >= 10:
            train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
                images, labels, test_size=0.2, random_state=777, stratify=labels
            )
            val_imgs, test_imgs, val_labels, test_labels = train_test_split(
                temp_imgs, temp_labels, test_size=0.5, random_state=777, stratify=temp_labels
            )
        else:
            train_imgs, val_imgs, train_labels, val_labels = train_test_split(
                images, labels, test_size=0.3, random_state=777, stratify=labels
            )

        train_dataset = utils.TrainDataset(train_imgs, train_labels)
        val_dataset = utils.TrainDataset(val_imgs, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN.CNN(num_unique_labels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_losses = []
        val_losses = []
        max_acc = 0

        EPOCHS = 10
        for epoch in range(EPOCHS):
            train_loss = CNN.train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = CNN.validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(f"{i}, {rs}, Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > max_acc:
                max_acc = val_acc
                path = f'Radical_Stroke/char_model/{rs}.pth'
                print("Saved better model")
                torch.save(model.state_dict(), path)
