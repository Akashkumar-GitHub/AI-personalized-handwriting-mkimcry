import torch
import torch.nn as nn


class HandwritingDiscriminator(nn.Module):

    def __init__(self):

        super(HandwritingDiscriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),

            nn.Linear(256 * 8 * 32, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)


# -----------------------------------
# Testing
# -----------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = HandwritingDiscriminator().to(device)

    model.eval()

    # Fake handwriting image
    sample = torch.randn(1, 1, 128, 512).to(device)

    with torch.no_grad():

        prediction = model(sample)

    print("\n[INFO] Discriminator Output:")
    print(prediction)