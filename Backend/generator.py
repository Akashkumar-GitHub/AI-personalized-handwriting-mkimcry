import torch
import torch.nn as nn


class HandwritingGenerator(nn.Module):

    def __init__(
        self,
        text_dim=256,
        style_dim=128,
        latent_dim=384
    ):

        super(HandwritingGenerator, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(text_dim + style_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 8 * 32 * 256),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, text_embedding, style_embedding):

        x = torch.cat(
            [text_embedding, style_embedding],
            dim=1
        )

        x = self.fc(x)

        x = x.view(-1, 256, 8, 32)

        x = self.deconv(x)

        return x


# -----------------------------------
# Testing
# -----------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = HandwritingGenerator().to(device)

    model.eval()

    # Fake embeddings for testing
    text_embedding = torch.randn(1, 256).to(device)

    style_embedding = torch.randn(1, 128).to(device)

    with torch.no_grad():

        generated = model(
            text_embedding,
            style_embedding
        )

    print("\n[INFO] Generated Image Shape:")
    print(generated.shape)