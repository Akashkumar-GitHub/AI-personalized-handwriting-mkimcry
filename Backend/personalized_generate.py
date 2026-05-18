from pathlib import Path

import torch
import torchvision.utils as vutils

from generator import HandwritingGenerator
from style_encoder import StyleEncoder, load_image as load_style_image
from text_encoder import TextEncoder, VOCAB_SIZE, text_to_tensor


BASE_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = BASE_DIR / "generator.pth"
STYLE_ENCODER_PATH = BASE_DIR / "style_encoder.pth"
TEXT_ENCODER_PATH = BASE_DIR / "text_encoder.pth"
STYLE_IMAGE_PATH = BASE_DIR / "processed_output.png"
OUTPUT_PATH = BASE_DIR / "personalized_handwriting.png"

INPUT_TEXT = "Hello this is my AI handwriting"


def load_state(model, checkpoint_path, device, model_name):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{model_name} checkpoint not found: {checkpoint_path}. "
            "Run train_gan.py before personalized generation."
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )
    model.eval()
    return model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    style_encoder = load_state(
        StyleEncoder().to(device),
        STYLE_ENCODER_PATH,
        device,
        "Style encoder"
    )
    text_encoder = load_state(
        TextEncoder(VOCAB_SIZE).to(device),
        TEXT_ENCODER_PATH,
        device,
        "Text encoder"
    )
    generator = load_state(
        HandwritingGenerator().to(device),
        GENERATOR_PATH,
        device,
        "Generator"
    )
    print("[INFO] Models loaded successfully")

    style_image = load_style_image(STYLE_IMAGE_PATH).to(device)
    text_tensor = text_to_tensor(INPUT_TEXT).to(device)

    with torch.no_grad():
        style_embedding = style_encoder(style_image)
        text_embedding = text_encoder(text_tensor)
        generated_image = generator(
            text_embedding,
            style_embedding
        )

    generated_image = (generated_image + 1) / 2
    vutils.save_image(generated_image, OUTPUT_PATH)

    print(f"[INFO] Handwriting saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
