"""Run BhiduAI's local character-level model in a terminal."""

import sys
import subprocess
from pathlib import Path

import torch

# Find the main BhiduAI folder
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.model import TinyGPT


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading BhiduAI...")
print("Using device:", device)


# -----------------------------
# Load checkpoint
# -----------------------------

checkpoint_path = ROOT_DIR / "checkpoints" / "bhiduai.pth"

if not checkpoint_path.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found at {checkpoint_path}. Run training/train.py first."
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)


# -----------------------------
# Restore tokenizer
# -----------------------------

chars = checkpoint["chars"]

vocab_size = checkpoint["vocab_size"]

stoi = {
    character: index
    for index, character in enumerate(chars)
}

itos = {
    index: character
    for index, character in enumerate(chars)
}


def encode(text):
    """Encode text, replacing unseen characters with a space when possible."""
    fallback = stoi.get(" ", 0)
    return [stoi.get(character, fallback) for character in text]


def decode(numbers):
    return "".join(
        itos.get(number, "")
        for number in numbers
    )


# -----------------------------
# Recreate exact model
# -----------------------------

model = TinyGPT(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    n_embd=checkpoint["n_embd"],
    n_head=checkpoint["n_head"],
    n_layer=checkpoint["n_layer"]
)


# Load only model weights
model.load_state_dict(
    checkpoint["model_state"]
)

model.to(device)

model.eval()

print("BhiduAI loaded successfully!")
print("Type 'exit' to close the chatbot.")
print()


# -----------------------------
# Text generation
# -----------------------------

def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8
):
    if max_new_tokens <= 0:
        return ""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    input_ids = encode(prompt)

    if len(input_ids) == 0:
        input_ids = [0]

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )

    original_length = input_tensor.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):

            input_context = input_tensor[
                :, -checkpoint["block_size"]:
            ]

            logits, _ = model(input_context)

            logits = logits[:, -1, :]

            logits = logits / temperature

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_tensor = torch.cat(
                [
                    input_tensor,
                    next_token
                ],
                dim=1
            )

    generated_ids = input_tensor[
        0,
        original_length:
    ].tolist()

    return decode(generated_ids)


# -----------------------------
# Chat loop
# -----------------------------

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("BhiduAI: Goodbye!")
        break

    # Teach BhiduAI
    if user_input.lower().startswith("/teach "):
        teaching_data = user_input[7:].strip()

        if teaching_data == "":
            print("Usage: /teach question => answer")
            continue

        with open(ROOT_DIR / "datasets" / "train.txt", "a", encoding="utf-8") as file:
            file.write("\n" + teaching_data + "\n")

        print("BhiduAI: Teaching data saved.")
        print("BhiduAI: Type /retrain to train using this new data.")
        continue

    # Retrain the model
    if user_input.lower() == "/retrain":
        print("BhiduAI: Retraining started...")
        print("Do not close PowerShell while training.")

        completed = subprocess.run([sys.executable, str(ROOT_DIR / "training" / "train.py")])
        if completed.returncode != 0:
            print("BhiduAI: Training failed. Check the output above.")
            continue

        print()
        print("BhiduAI: Retraining complete.")
        print("Restart chat.py to load the new model.")
        break

    result = generate_text(user_input)

    print("BhiduAI:", result)
    print()
