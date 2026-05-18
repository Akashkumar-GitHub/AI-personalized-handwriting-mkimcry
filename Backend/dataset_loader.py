import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pathlib import Path

from iam_text_parser import parse_iam_label_map


# -----------------------------------
# Image Transform
# -----------------------------------

transform = transforms.Compose([

    transforms.Resize((128, 512)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# -----------------------------------
# Dataset Class
# -----------------------------------

class IAMDataset(Dataset):

    def __init__(self, image_folder, label_file=None):

        self.image_folder = Path(image_folder)

        self.image_paths = []
        self.labels = {}

        if label_file is not None:
            self.labels = parse_iam_label_map(label_file)

        for root, dirs, files in os.walk(self.image_folder):

            for file in files:

                if file.lower().endswith((".png", ".jpg", ".jpeg")):

                    self.image_paths.append(
                        Path(root) / file
                    )

        self.image_paths.sort()

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("L")

        image = transform(image)

        image_id = image_path.stem
        text = self.labels.get(image_id, "")

        return image, text


# -----------------------------------
# Test Loader
# -----------------------------------

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parents[1]

    dataset = IAMDataset(
        base_dir / "datasets"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    print(f"[INFO] Dataset Size: {len(dataset)}")

    for images, texts in dataloader:

        print("[INFO] Batch Shape:")

        print(images.shape)
        print(texts)

        break
