import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score

from CNN import CNN, train, validate, test_character
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset


def train_main():
    """
    load data
    """
    logger.info("Start loading data")
    images, labels = load_train_dataset()
    unique_labels = set(labels) 
    num_unique_labels = len(unique_labels)
    images, labels = shuffle(images, labels, random_state=777)
    train_len = int(0.8 * len(images))

    train_images, val_images = images[:train_len], images[train_len:]
    train_labels, val_labels = labels[:train_len], labels[train_len:]

    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = TrainDataset(val_images, val_labels)
    
    """
    CNN - train and validate
    """
    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    model = CNN(num_unique_labels).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)

    train_losses = []
    val_losses = []
    max_acc = 0

    EPOCHS = 10
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn_model.pth') # Save the best model

    logger.info(f"Best Accuracy: {max_acc:.4f}")

def test_main(count):
    """
    load data
    """
    logger.info("Start loading data")
    train_dataset = TrainDataset(load_train_dataset()[0], load_train_dataset()[1])
    original_labels = train_dataset.index_to_label
    unique_labels = set(original_labels) 
    num_unique_labels = len(unique_labels)
    test_images = load_test_dataset(count)
    test_dataset = TestDataset(test_images)
    """
    CNN - test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    criterion = nn.CrossEntropyLoss()
    model = CNN(num_unique_labels).to(device)
    model.load_state_dict(torch.load('best_cnn_model.pth', weights_only=True))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_character(model, test_loader, criterion, device, original_labels)

if __name__ == '__main__':
    train_main()
    test_main(100)
