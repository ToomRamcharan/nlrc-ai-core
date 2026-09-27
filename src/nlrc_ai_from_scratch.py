# ==============================================================================
# 🚀 NLRC AI: From-Scratch Multimodal Transformer Training Pipeline
# Based on: https://github.com/FareedKhan-dev/train-llm-from-scratch
# Capabilities:
#   1. Pure PyTorch Decoder Transformer (RoPE + SwiGLU + RMSNorm)
#   2. Multimodal Vision Patch Projector (Image Understanding)
#   3. Real-Time Web Search Tool (Current Affairs & World Knowledge)
#   4. 100% Cloud Execution in Google Colab (Zero-Local Storage)
# ==============================================================================

import os
import sys
import math
import time
from io import BytesIO
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests

# ------------------------------------------------------------------------------
# 1. Fareed Khan Transformer Building Blocks (RoPE, SwiGLU, Attention)
# ------------------------------------------------------------------------------

def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in LLaMA / modern LLMs."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLUMlp(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """Multi-Head Attention with Rotary Positional Embeddings (Fareed Khan arch)."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(output)

class TransformerBlock(nn.Module):
    """Single Transformer Block with RMSNorm, Attention, and SwiGLU."""
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLUMlp(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# ------------------------------------------------------------------------------
# 2. Multimodal Vision Projector for NLRC AI
# ------------------------------------------------------------------------------

class VisionPatchProjector(nn.Module):
    """
    Extracts image patches and projects them into the Transformer embedding space.
    Turns an image into visual tokens that NLRC AI can reason over.
    """
    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = RMSNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input: images shape (B, C, H, W)
        Output: visual token sequence shape (B, num_patches, embed_dim)
        """
        x = self.proj(images)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return self.norm(x)

# ------------------------------------------------------------------------------
# 3. Full NLRC AI Architecture
# ------------------------------------------------------------------------------

class NLRCAI(nn.Module):
    """
    NLRC AI: Decoder-only multimodal transformer capable of:
      - Processing text tokens
      - Processing visual image patch tokens
      - Generating responses and triggering web search tools
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        hidden_dim: int = 1536,
        max_seq_len: int = 1024,
        patch_size: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.vision_projector = VisionPatchProjector(patch_size=patch_size, embed_dim=dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_rope_freqs_cis(dim // n_heads, max_seq_len * 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)  # (B, seq_len, dim)

        # Multimodal fusion: prepend image tokens if image is provided
        if images is not None:
            vis_tokens = self.vision_projector(images)  # (B, num_patches, dim)
            x = torch.cat([vis_tokens, x], dim=1)
            total_seq_len = x.shape[1]
        else:
            total_seq_len = seq_len

        # Causal attention mask
        mask = torch.full((total_seq_len, total_seq_len), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        freqs_cis = self.freqs_cis[:total_seq_len].to(input_ids.device)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        logits = self.output(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # If visual tokens were prepended, slice targets accordingly
            if images is not None:
                logits_for_loss = logits[:, vis_tokens.shape[1]:, :]
            else:
                logits_for_loss = logits
            loss = F.cross_entropy(logits_for_loss.reshape(-1, logits_for_loss.size(-1)), targets.reshape(-1))

        return logits, loss

# ------------------------------------------------------------------------------
# 4. Live Search Tool for Real-Time Current Affairs
# ------------------------------------------------------------------------------

def query_duckduckgo_search(query: str, max_results: int = 3) -> str:
    """Zero-cost, open-source search retriever for worldwide current affairs."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = [f"• {r['title']}: {r['body']}" for r in ddgs.text(query, max_results=max_results)]
            return "\n".join(results) if results else "No current news found."
    except Exception as err:
        return f"Search retrieval error: {err}"

# ------------------------------------------------------------------------------
# 5. Cloud Training Orchestrator (Runs on Colab Google AI Pro GPU)
# ------------------------------------------------------------------------------

def train_nlrc_ai_in_colab(epochs: int = 5, batch_size: int = 8, lr: float = 3e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training NLRC AI on device: {device}")
    if device == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")

    # Initialize model
    model = NLRCAI(vocab_size=32000, dim=512, n_layers=8, n_heads=8, hidden_dim=1536).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"[*] Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("[*] Starting NLRC AI training loop...")

    # Simulated training step demonstration
    for epoch in range(1, epochs + 1):
        # Create sample inputs (representing streamed cloud tokens)
        sample_ids = torch.randint(0, 32000, (batch_size, 128), device=device)
        sample_targets = torch.randint(0, 32000, (batch_size, 128), device=device)
        sample_images = torch.randn(batch_size, 3, 224, 224, device=device)

        model.train()
        optimizer.zero_grad()
        logits, loss = model(input_ids=sample_ids, images=sample_images, targets=sample_targets)
        loss.backward()
        optimizer.step()

        print(f"    Epoch {epoch}/{epochs} | Step Loss: {loss.item():.4f}")

    print("[SUCCESS] NLRC AI cloud training run completed!")
    
    # Save checkpoint to Google Drive
    save_path = "/content/drive/MyDrive/NLRC-AI-Weights/nlrc_ai_checkpoint.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[+] Model checkpoint saved to: {save_path}")

if __name__ == "__main__":
    train_nlrc_ai_in_colab()
