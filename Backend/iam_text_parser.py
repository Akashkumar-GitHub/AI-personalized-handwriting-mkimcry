import os
from pathlib import Path


def parse_iam_labels(label_file_path):

    samples = []

    with open(label_file_path, "r", encoding="utf-8") as file:

        lines = file.readlines()

    for line in lines:

        if line.startswith("#"):
            continue

        parts = line.strip().split(" ")

        if len(parts) < 9:
            continue

        image_id = parts[0]

        text = " ".join(parts[8:])

        samples.append({
            "image_id": image_id,
            "text": text
        })

    return samples


def iam_image_path(image_root, image_id):

    parts = image_id.split("-")
    if len(parts) < 2:
        return Path(image_root) / f"{image_id}.png"

    return (
        Path(image_root)
        / parts[0]
        / f"{parts[0]}-{parts[1]}"
        / f"{image_id}.png"
    )


def parse_iam_label_map(label_file_path):

    return {
        sample["image_id"]: sample["text"]
        for sample in parse_iam_labels(label_file_path)
    }


# -----------------------------------
# Test
# -----------------------------------

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[1]

    LABEL_FILE = BASE_DIR / "datasets" / "words.txt"

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"IAM label file not found: {LABEL_FILE}"
        )

    data = parse_iam_labels(LABEL_FILE)

    print(f"[INFO] Total Samples: {len(data)}")

    print("\n[INFO] First Sample:")

    print(data[0])
