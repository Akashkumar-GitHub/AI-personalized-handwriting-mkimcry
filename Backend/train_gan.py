import torch
import torch.nn as nn
import torch.optim as optim

from generator import HandwritingGenerator
from discriminator import HandwritingDiscriminator


# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] Using device: {device}")


# -----------------------------------
# Models
# -----------------------------------

generator = HandwritingGenerator().to(device)

discriminator = HandwritingDiscriminator().to(device)


# -----------------------------------
# Loss Function
# -----------------------------------

criterion = nn.BCELoss()


# -----------------------------------
# Optimizers
# -----------------------------------

g_optimizer = optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

d_optimizer = optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


# -----------------------------------
# Training Parameters
# -----------------------------------

EPOCHS = 50
BATCH_SIZE = 4


# -----------------------------------
# Dummy Data (Replace Later)
# -----------------------------------

def get_real_samples(batch_size):

    real_images = torch.randn(
        batch_size,
        1,
        128,
        512
    ).to(device)

    return real_images


# -----------------------------------
# Training Loop
# -----------------------------------

for epoch in range(EPOCHS):

    # -------------------------
    # Train Discriminator
    # -------------------------

    discriminator.zero_grad()

    # Real Images
    real_images = get_real_samples(BATCH_SIZE)

    real_labels = torch.ones(
        BATCH_SIZE,
        1
    ).to(device)

    fake_labels = torch.zeros(
        BATCH_SIZE,
        1
    ).to(device)

    real_outputs = discriminator(real_images)

    d_loss_real = criterion(
        real_outputs,
        real_labels
    )

    # Fake Images
    text_embedding = torch.randn(
        BATCH_SIZE,
        256
    ).to(device)

    style_embedding = torch.randn(
        BATCH_SIZE,
        128
    ).to(device)

    fake_images = generator(
        text_embedding,
        style_embedding
    )

    fake_outputs = discriminator(
        fake_images.detach()
    )

    d_loss_fake = criterion(
        fake_outputs,
        fake_labels
    )

    d_loss = d_loss_real + d_loss_fake

    d_loss.backward()

    d_optimizer.step()

    # -------------------------
    # Train Generator
    # -------------------------

    generator.zero_grad()

    fake_outputs = discriminator(fake_images)

    g_loss = criterion(
        fake_outputs,
        real_labels
    )

    g_loss.backward()

    g_optimizer.step()

    # -------------------------
    # Print Progress
    # -------------------------

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"D Loss: {d_loss.item():.4f} "
        f"G Loss: {g_loss.item():.4f}"
    )


# -----------------------------------
# Save Models
# -----------------------------------

torch.save(
    generator.state_dict(),
    "generator.pth"
)

torch.save(
    discriminator.state_dict(),
    "discriminator.pth"
)

print("\n[INFO] Models saved successfully")