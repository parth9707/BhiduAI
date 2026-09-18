"""A compact decoder-only Transformer used by BhiduAI."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        k = self.key(x)
        q = self.query(x)

        attention_scores = q @ k.transpose(-2, -1)

        attention_scores = attention_scores * (
            channels ** -0.5
        )

        attention_scores = attention_scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(attention_weights)

        v = self.value(x)

        output = attention_weights @ v

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size):
        super().__init__()

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                head_size=head_size,
                n_embd=n_embd,
                block_size=block_size
            )
            for _ in range(n_head)
        ])

        self.projection = nn.Linear(
            n_head * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        output = self.projection(output)

        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        head_size = n_embd // n_head

        self.attention = MultiHeadAttention(
            n_head=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size
        )

        self.feed_forward = FeedForward(n_embd)

        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=64,
        n_embd=64,
        n_head=4,
        n_layer=2
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size
                )
                for _ in range(n_layer)
            ]
        )

        self.final_norm = nn.LayerNorm(n_embd)

        self.output_layer = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, input_ids, targets=None):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"Input length {sequence_length} exceeds block size {self.block_size}."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.transformer_blocks(x)

        x = self.final_norm(x)

        logits = self.output_layer(x)

        loss = None

        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape

            logits_for_loss = logits.reshape(
                batch_size * sequence_length,
                vocab_size
            )

            targets_for_loss = targets.reshape(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(
                logits_for_loss,
                targets_for_loss
            )

        return logits, loss
