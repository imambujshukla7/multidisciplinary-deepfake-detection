import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

class CustomDataset(Dataset):
    def __init__(self, data_csv, image_dir, transform=None):
        """
        Initializing CustomDataset.
        :param data_csv: Path to the CSV file containing image paths and labels
        :param image_dir: Directory where the images are stored
        :param transform: Transformations to apply to the images
        """
        self.logger = logging.getLogger('data_loader_logger')
        self.data_csv = data_csv
        self.image_dir = image_dir
        self.transform = transform

        try:
            self.data = pd.read_csv(data_csv)
            self.logger.info(f"Loaded data from {data_csv}")
        except Exception as e:
            self.logger.error(f"Error loading data from {data_csv}: {e}", exc_info=True)
            raise

    def __len__(self):
        """
        Getting the number of samples in dataset.
        :return: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        To get sample from dataset.
        :param idx: Index of the sample
        :return: Tuple containing the image and its label
        """
        try:
            img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.data.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)

            self.logger.debug(f"Loaded image {img_name} with label {label}")

            return image, label
        except Exception as e:
            self.logger.error(f"Error loading image {img_name}: {e}", exc_info=True)
            raise

def create_dataloader(data_csv, image_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Creating DataLoader.
    :param data_csv: Path to the CSV file containing image paths and labels
    :param image_dir: Directory where the images are stored
    :param batch_size: Number of samples per batch
    :param shuffle: Whether to shuffle the data
    :param num_workers: Number of subprocesses to use for data loading
    :return: DataLoader
    """
    logger = logging.getLogger('data_loader_logger')
    logger.info(f"Creating DataLoader with data_csv={data_csv}, image_dir={image_dir}, batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")

    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = CustomDataset(data_csv=data_csv, image_dir=image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        logger.info(f"DataLoader created successfully")

        return dataloader
    except Exception as e:
        logger.error(f"Error creating DataLoader: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('data_loader_logger')
    
    # usage
    data_csv = 'path/to/data.csv'
    image_dir = 'path/to/images'
    dataloader = create_dataloader(data_csv, image_dir)

    # Displaying the shape of the first batch of images and labels
    for images, labels in dataloader:
        print(f'Image batch shape: {images.size()}')
        print(f'Label batch shape: {labels.size()}')
        break
