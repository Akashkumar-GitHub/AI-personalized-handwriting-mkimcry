from pathlib import Path

import torch
import torchvision.utils as vutils

from generator import HandwritingGenerator


BASE_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = BASE_DIR / "generator.pth"
OUTPUT_PATH = BASE_DIR / "generated_handwriting.png"


def load_generator(device):

    if not GENERATOR_PATH.exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found: {GENERATOR_PATH}"
        )

    generator = HandwritingGenerator().to(device)
    generator.load_state_dict(
        torch.load(
            GENERATOR_PATH,
            map_location=device
        )
    )
    generator.eval()
    return generator


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    generator = load_generator(device)
    print("[INFO] Generator loaded successfully")

    text_embedding = torch.randn(1, 256).to(device)
    style_embedding = torch.randn(1, 128).to(device)

    with torch.no_grad():
        generated_image = generator(
            text_embedding,
            style_embedding
        )

    generated_image = (generated_image + 1) / 2
    vutils.save_image(generated_image, OUTPUT_PATH)

    print(f"[INFO] Saved generated image at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
