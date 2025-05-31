import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd
from Radical_Character import utils

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 12 * 12, num_classes)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.maxpool1(self.dropout1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.maxpool2(self.dropout2(self.relu2(self.bn2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def test_radical(model: CNN, test_loader: DataLoader, criterion, device, original_labels):
    results = []
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for i, name in enumerate(image_names):
                import os
                radical_name = original_labels[predictions[i].item()]
                labels = []
                correct_radical = ""
                for count_label in os.listdir('data_new/'):
                    if(count_label.split("_")[1] == radical_name):
                        labels.append(count_label[0])
                    if(count_label[0] == name[0]):
                        correct_radical = count_label.split("_")[1]
                model_name = 'Radical_Character/char_model/' + radical_name + '.pth'
                model_character = CNN(len(set(labels))).to(device)
                model_character.load_state_dict(torch.load(model_name, weights_only=True))
                guess = test_character(model_character, radical_name, test_loader, criterion, device, images)
                results.append({'id1': correct_radical, 'id2': image_names[i][0], 'radical': original_labels[predictions[i].item()], 'char': guess}) # Store the original label
    df = pd.DataFrame(results)
    df.to_csv('Radical_Character/CNN.csv', index=False)
    print(f"Predictions saved to 'CNN.csv'")
    return

def test_character(model: CNN, radical_name: str, test_loader: DataLoader, criterion, device, image):
    model.eval()
    dataset = utils.TrainDataset(utils.load_train_dataset(radical_name)[0], utils.load_train_dataset(radical_name)[1])
    original_labels = dataset.index_to_label
    with torch.no_grad():
        outputs = model(image)
        _, predictions = torch.max(outputs, 1)
    return original_labels[predictions[0].item()]