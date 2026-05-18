from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader import IAMDataset
from discriminator import HandwritingDiscriminator
from generator import HandwritingGenerator
from style_encoder import StyleEncoder
from text_encoder import TextEncoder, VOCAB_SIZE, texts_to_tensor


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "datasets"
LABEL_FILE = DATASET_DIR / "words.txt"

EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0002
L1_WEIGHT = 50.0


def build_dataloader():

    label_file = LABEL_FILE if LABEL_FILE.exists() else None
    dataset = IAMDataset(DATASET_DIR, label_file=label_file)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training images found in {DATASET_DIR}. "
            "Add IAM images before running training."
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    generator = HandwritingGenerator().to(device)
    discriminator = HandwritingDiscriminator().to(device)
    style_encoder = StyleEncoder().to(device)
    text_encoder = TextEncoder(VOCAB_SIZE).to(device)

    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()

    g_optimizer = optim.Adam(
        list(generator.parameters())
        + list(style_encoder.parameters())
        + list(text_encoder.parameters()),
        lr=LR,
        betas=(0.5, 0.999)
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=LR,
        betas=(0.5, 0.999)
    )

    dataloader = build_dataloader()

    for epoch in range(EPOCHS):

        for real_images, texts in dataloader:

            real_images = real_images.to(device)
            text_tensor = texts_to_tensor(texts).to(device)
            style_images = real_images.repeat(1, 3, 1, 1)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            discriminator.zero_grad()

            real_outputs = discriminator(real_images)
            d_loss_real = adversarial_loss(real_outputs, real_labels)

            with torch.no_grad():
                text_embedding = text_encoder(text_tensor)
                style_embedding = style_encoder(style_images)
                fake_images = generator(text_embedding, style_embedding)

            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            generator.zero_grad()
            style_encoder.zero_grad()
            text_encoder.zero_grad()

            text_embedding = text_encoder(text_tensor)
            style_embedding = style_encoder(style_images)
            fake_images = generator(text_embedding, style_embedding)
            fake_outputs = discriminator(fake_images)

            g_loss_adv = adversarial_loss(fake_outputs, real_labels)
            g_loss_recon = reconstruction_loss(fake_images, real_images)
            g_loss = g_loss_adv + (L1_WEIGHT * g_loss_recon)

            g_loss.backward()
            g_optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"D Loss: {d_loss.item():.4f} "
            f"G Loss: {g_loss.item():.4f}"
        )

    torch.save(generator.state_dict(), BASE_DIR / "generator.pth")
    torch.save(discriminator.state_dict(), BASE_DIR / "discriminator.pth")
    torch.save(style_encoder.state_dict(), BASE_DIR / "style_encoder.pth")
    torch.save(text_encoder.state_dict(), BASE_DIR / "text_encoder.pth")

    print("\n[INFO] Models saved successfully")


if __name__ == "__main__":
    main()
