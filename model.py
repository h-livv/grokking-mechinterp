"""The paper's one-block transformer for addition modulo 113.

One block, width 128, four 32-dimensional heads, a 512-unit ReLU MLP,
learned positions, untied embeddings, and no normalization or biases.
Tensors use [batch, position, feature]; linear weights use [output, input].
"""

import torch
from torch import nn


class Transformer(nn.Module):
    """Embed tokens, add attention and MLP updates, then read residue logits.

    Input: int64 [batch, length], length <= 3, IDs 0..113 ('=' is 113).
    Output: [batch, length, 113]. Train only on the last position of [a,b,=].
    Initialization uses PyTorch defaults and positional normal std 0.02,
    following the local plan rather than the authors' exact initialized state.
    """

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(114, 128)
        self.position_embedding = nn.Parameter(torch.empty(3, 128))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.query = nn.Linear(128, 128, bias=False)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.attention_out = nn.Linear(128, 128, bias=False)
        self.mlp_in = nn.Linear(128, 512, bias=False)
        self.relu = nn.ReLU()
        self.mlp_out = nn.Linear(512, 128, bias=False)
        self.unembed = nn.Linear(128, 113, bias=False)

    def forward(self, tokens):
        """Compute scaled causal attention and the two residual additions."""
        batch, length = tokens.shape
        x = self.token_embedding(tokens) + self.position_embedding[:length]

        # Separate heads: [batch, position, 128] -> [batch, head, position, 32].
        q = self.query(x).reshape(batch, length, 4, 32).transpose(1, 2)
        k = self.key(x).reshape(batch, length, 4, 32).transpose(1, 2)
        v = self.value(x).reshape(batch, length, 4, 32).transpose(1, 2)

        # A[i,j] is attention from query i to key j; only j <= i is visible.
        scores = (q @ k.transpose(-2, -1)) / (32 ** 0.5)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        attention = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
        heads = attention @ v
        joined = heads.transpose(1, 2).reshape(batch, length, 128)
        x = x + self.attention_out(joined)

        # The positionwise MLP adds a second update to the residual stream.
        hidden = self.relu(self.mlp_in(x))
        x = x + self.mlp_out(hidden)
        return self.unembed(x)
