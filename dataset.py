import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RetinaDataset(Dataset):
    def __init__(self, csv_path, images_dir, train=True):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir

        # Images arrive in different sizes/qualities - resize everything to
        # a consistent 224x224 (standard input size for most pretrained CNNs).
        # For training data, we also add random flips/rotations - this is
        # called "data augmentation": it artificially creates variety so the
        # model doesn't just memorize exact image orientations, which helps
        # it generalize better to new images.
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            # No augmentation for validation - we want to evaluate on the
            # image as-is, consistently, every time.
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, f"{row['id_code']}.png")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row["diagnosis"])

        return image, label


# Quick sanity check when running this file directly
if __name__ == "__main__":
    dataset = RetinaDataset(
        csv_path="data/train_split.csv",
        images_dir="data/train_images",
        train=True,
    )

    print(f"Dataset size: {len(dataset)}")

    image, label = dataset[0]
    print(f"First image shape: {image.shape}")  # should be [3, 224, 224]
    print(f"First image label: {label}")