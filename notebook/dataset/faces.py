import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

DATA_FOLDER = os.path.join("..", "data")
ORIGINAL_VIDEOS_FOLDER = os.path.join(DATA_FOLDER, "original_sequences", "actors", "c23", "videos")
FAKE_VIDEOS_FOLDER = os.path.join(DATA_FOLDER, "manipulated_sequences", "DeepFakeDetection", "c23", "videos" )
FACES_FOLDER = os.path.join(DATA_FOLDER, "faces")
FACES_REAL = os.path.join(FACES_FOLDER, "real")
FACES_FAKE = os.path.join(FACES_FOLDER, "fake")
FACES_CSV = os.path.join(FACES_FOLDER, "faces.csv")

class Faces(data.Dataset):
    def __init__(self, root:str, csv:str, split:str="training", transform:bool=None):
        self.root = root
        self.csv = pd.read_csv(csv)
        self.transform = transform
        self.split = split
        self.labels = {"real": 0, "fake": 1}

    def __len__(self):
        return len(self.csv)

    @staticmethod
    def get_train_transformations():
        return A.Compose([
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            ToTensorV2(p=1.0)
        ])

    @staticmethod
    def get_val_transformations():
        return A.Compose([
            A.Resize(256, 256),
            ToTensorV2(p=1.0)
        ])
    
    def _read_image(self, image_path:str, label:str):
        image = ""
        if label == "real":
            image = os.path.join(self.root, "real", image_path)
        else:
            image = os.path.join(self.root, "fake", image_path)
        
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = image.astype(np.float32)

        return image

    def __getitem__(self, idx:int):
        idx = int(idx)
        row = self.csv.iloc[idx]
        image = row["name"]
        label = row["label"]

        image = self._read_image(image, label)

        if self.transform and self.split == "training":
            transformation = self.get_train_transformations()
        else:
            transformation = self.get_val_transformations()
        
        image = transformation(image=image)["image"]

        return image, torch.tensor(self.labels[label]).float()

if __name__ == "__main__":
    dataset = Faces(root=FACES_FOLDER, csv=FACES_CSV, split="training", transform=True)
    labels = {0: "real", 1: "fake"}

    image, label = dataset[0]
    label = label.item()
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image * 255.0
    image = image.astype(np.uint8)

    plt.title(labels[int(label)])
    plt.imshow(image)
    plt.show()

