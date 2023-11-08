import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import cv2

class BikeBicycleDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, val_split=0.2):
        
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.images = []
        self.labels = []

        # Load all the images and labels
        for label, folder in enumerate(['bike', 'motorbike']):
            folder_path = os.path.join(self.root_dir, folder)
            for image_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, image_name))
                self.labels.append(label)

        # Split the data into training and validation sets
        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(
            self.images, self.labels, test_size=val_split, stratify=self.labels)

        # Use the correct set for this instance
        self.data = self.train_images if self.train else self.val_images
        self.labels = self.train_labels if self.train else self.val_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        if image.mode == 'P':
            image = image.convert('RGBA')

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    


