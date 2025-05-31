import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score

from Stroke_Character import CNN, utils
from sklearn.model_selection import train_test_split

def train_main_stroke():
    """
    load data
    """
    logger.info("Start loading data")
    images, labels = utils.load_train_dataset()
    unique_labels = set(labels) 
    num_unique_labels = len(unique_labels)
    images, labels = shuffle(images, labels, random_state=777)

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.2, random_state=777, stratify=labels
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=777, stratify=temp_labels
    )

    train_dataset = utils.TrainDataset(train_images, train_labels)
    val_dataset   = utils.TrainDataset(val_images, val_labels)
    test_dataset  = utils.TrainDataset(test_images, test_labels)  # 可以回傳用於 test_main
    
    """
    CNN - train and validate
    """
    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    model = CNN.CNN(num_unique_labels).to(device)
    criterion = nn.CrossEntropyLoss()

    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)

    train_losses = []
    val_losses = []
    max_acc = 0

    EPOCHS = 10
    for epoch in range(EPOCHS): #epoch
        train_loss = CNN.train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = CNN.validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > max_acc:
            max_acc = val_acc
            print("Saved better model")
            torch.save(model.state_dict(), 'Stroke_Character/best_cnn_model.pth') # Save the best model

    logger.info(f"Best Accuracy: {max_acc:.4f}")
    return test_dataset, num_unique_labels, train_dataset.index_to_label

def train_main_character():
    images, labels = utils.load_train_dataset()
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.2, random_state=777, stratify=labels
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=777, stratify=temp_labels
    )
    strokes = set(labels) 
    """
    load data
    """
    for i, stroke in enumerate(strokes):
        logger.info("Start loading data")
        images, labels = utils.load_train_dataset(stroke)
        unique_labels = set(labels) 
        num_unique_labels = len(unique_labels)
        images, labels = shuffle(images, labels, random_state=777)
        train_len = int(0.7 * len(images))

        train_images, val_images = images[:train_len], images[train_len:]
        train_labels, val_labels = labels[:train_len], labels[train_len:]

        train_dataset = utils.TrainDataset(train_images, train_labels)
        val_dataset = utils.TrainDataset(val_images, val_labels)
        
        """
        CNN - train and validate
        """
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
        model = CNN.CNN(num_unique_labels).to(device)
        criterion = nn.CrossEntropyLoss()

        # Optimizer configuration
        base_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optimizer = optim.Adam(base_params, lr=1e-4)

        train_losses = []
        val_losses = []
        max_acc = 0

        EPOCHS = 10
        for epoch in range(EPOCHS): #epoch
            train_loss = CNN.train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = CNN.validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(f"{i}, {stroke}, Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > max_acc:
                max_acc = val_acc
                path = 'Stroke_Character/char_model/' + stroke + '.pth'
                print("Saved better model")
                torch.save(model.state_dict(), path) # Save the best model

def test_main(test_dataset, num_classes, original_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = CNN(num_classes).to(device)
    model.load_state_dict(torch.load('Stroke_Character/best_cnn_model.pth', weights_only=True))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    CNN.test_stroke(model, test_loader, criterion, device, original_labels)

def main(train, test_count):
    if train:
        test_dataset, num_classes, index_to_label = train_main_stroke()
        train_main_character()
    test_main(test_dataset, num_classes, index_to_label)
    
    # 評估
    import csv
    with open('Stroke_Character/CNN.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        total = 0
        correct = 0
        stroke_correct = 0
        for row in rows:
            total += 1
            if row[0] == row[2] and row[1] == row[3]:
                correct += 1
            elif row[0] == row[2]:
                stroke_correct += 1
                print(row[1]+"-> "+row[3])
            else:
                print(row[1]+"-> "+row[3]+" | "+row[0]+"-> "+row[2])
        print("Total entries    :", total)
        print("Correct entries  :", correct)
        print("Partial entries  :", stroke_correct)
        print("Accuracy         : {:.4f}".format(correct/total))
        print("Partial Accuracy : {:.4f}".format(stroke_correct/total))


if __name__ == '__main__':
    main(train=True, test_count=100)