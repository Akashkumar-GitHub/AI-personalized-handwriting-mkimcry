import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

class StyleEncoder(nn.Module):

    def __init__(self, embedding_dim=128):

        super(StyleEncoder, self).__init__()

        # Load pretrained ResNet18
        base_model = models.resnet18(pretrained=True)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )

        # Embedding layer
        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):

        features = self.feature_extractor(x)

        features = features.view(features.size(0), -1)

        embedding = self.embedding(features)

        return embedding


# -----------------------------
# Image Preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -----------------------------
# Load Image
# -----------------------------

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)
    return image


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    IMAGE_PATH = "processed_output.png"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = StyleEncoder().to(device)

    model.eval()

    image = load_image(IMAGE_PATH).to(device)

    with torch.no_grad():

        embedding = model(image)

    print("\n[INFO] Style Embedding Shape:")
    print(embedding.shape)

    print("\n[INFO] Style Vector:")
    print(embedding)