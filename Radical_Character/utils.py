import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple

class TrainDataset(Dataset):

    index_to_label = {}
    label_to_index = {}

    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])
        self.images = images
        self.string_labels = labels  # Store original string labels
        self.unique_labels = sorted(list(set(labels)))  # Get unique labels
        self.label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in self.label_to_index.items()} # Reverse mapping here

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        string_label = self.string_labels[idx]
        numerical_label = self.label_to_index[string_label]  # Convert to numerical index
        label_tensor = torch.tensor(numerical_label, dtype=torch.long)
        return image, label_tensor

def index_to_radical(dataset : TrainDataset, index):
    return dataset.index_to_label[index]

def radical_to_index(dataset : TrainDataset, radical):
    return dataset.label_to_index[radical]

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
def load_train_dataset(labelBy: str='radical', path: str='data_new/')->Tuple[List, List]:
    images = []
    labels = []
                        
    for folder in os.listdir(path):
        if labelBy != 'radical':
            if folder.split("_")[1] == labelBy:
                for image in os.listdir(path+folder):
                    images.append(path+folder+"/"+image)
                    labels.append(folder[0])  # labels based on character
        else:
            for image in os.listdir(path+folder):
                images.append(path+folder+"/"+image)
                labels.append(folder.split("_")[1])      # labels based on radical
        
    return images, labels

def load_test_dataset(count = 100)->List:
    images = []
    import random as rd
    for root, _, files in os.walk('data_new/'):
        for file in files:
            if file.lower().endswith(".png"):
                full_path = os.path.join(root, file)
                images.append(full_path)
    return rd.sample(images, count)

    
