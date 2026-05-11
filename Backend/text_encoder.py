import torch
import torch.nn as nn

class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256
    ):

        super(TextEncoder, self).__init__()

        # Character embedding layer
        self.char_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # BiLSTM Encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Final projection
        self.fc = nn.Linear(hidden_dim * 2, 256)

    def forward(self, x):

        embedded = self.char_embedding(x)

        outputs, (hidden, cell) = self.lstm(embedded)

        # Concatenate forward + backward hidden states
        hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        encoded = self.fc(hidden)

        return encoded


# ------------------------------------
# Character Vocabulary
# ------------------------------------

characters = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?'-"
)

char_to_idx = {
    ch: idx + 1
    for idx, ch in enumerate(characters)
}

idx_to_char = {
    idx: ch
    for ch, idx in char_to_idx.items()
}

VOCAB_SIZE = len(char_to_idx) + 1


# ------------------------------------
# Text Processing
# ------------------------------------

def text_to_tensor(text):

    indices = []

    for ch in text:

        if ch in char_to_idx:
            indices.append(char_to_idx[ch])

    tensor = torch.tensor(indices).unsqueeze(0)

    return tensor


# ------------------------------------
# Main
# ------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TextEncoder(VOCAB_SIZE).to(device)

    model.eval()

    sample_text = "Hello AI"

    text_tensor = text_to_tensor(sample_text).to(device)

    with torch.no_grad():

        embedding = model(text_tensor)

    print("\n[INFO] Text Embedding Shape:")
    print(embedding.shape)

    print("\n[INFO] Text Embedding:")
    print(embedding)