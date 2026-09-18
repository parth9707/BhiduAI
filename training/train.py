"""Train the character-level BhiduAI model from ``datasets/train.txt``."""

import argparse
import sys
from pathlib import Path

import torch

# Find the main BhiduAI folder
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.model import TinyGPT


# -----------------------------
# Settings
# -----------------------------

BATCH_SIZE = 16
BLOCK_SIZE = 64
MAX_STEPS = 2000
LEARNING_RATE = 0.001

N_EMBD = 64
N_HEAD = 4
N_LAYER = 2


# -----------------------------
# Device
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# -----------------------------
# Character tokenizer
# -----------------------------

def make_batch(encoded_text, batch_size, device):
    starting_positions = torch.randint(
        0,
        len(encoded_text) - BLOCK_SIZE,
        (batch_size,)
    )

    input_batch = torch.stack([
        encoded_text[
            position:position + BLOCK_SIZE
        ]
        for position in starting_positions
    ])

    target_batch = torch.stack([
        encoded_text[
            position + 1:position + BLOCK_SIZE + 1
        ]
        for position in starting_positions
    ])

    return (
        input_batch.to(device),
        target_batch.to(device)
    )


def main():
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--steps and --batch-size must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_path = ROOT_DIR / "datasets" / "train.txt"
    text = dataset_path.read_text(encoding="utf-8")
    if len(text) < BLOCK_SIZE + 1:
        raise ValueError("train.txt must contain at least 65 characters.")

    chars = sorted(set(text))
    stoi = {character: index for index, character in enumerate(chars)}
    encoded_text = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    print(f"Vocabulary size: {len(chars)}")
    print(f"Dataset characters: {len(encoded_text)}")

    model = TinyGPT(len(chars), BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Starting training...")
    model.train()
    for step in range(args.steps):
        input_batch, target_batch = make_batch(encoded_text, args.batch_size, device)
        _, loss = model(input_batch, target_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"Step: {step + 1}/{args.steps} | Loss: {loss.item():.4f}")

    checkpoint_dir = ROOT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "bhiduai.pth"
    torch.save({"model_state": model.state_dict(), "chars": chars, "vocab_size": len(chars),
                "block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD,
                "n_layer": N_LAYER}, checkpoint_path)
    print(f"Training complete! Model saved at: {checkpoint_path}")


if __name__ == "__main__":
    main()
