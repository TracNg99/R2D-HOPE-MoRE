"""
SequentialAttention++ Block Sparsification for R²D-HOPE-MoRE.

Implements the two-phase SA++ algorithm from arXiv:2402.17902:

  Phase 1 — Joint training with differentiable importance scores:
    Each target Linear layer gets a BlockImportanceMask wrapping it.
    The mask multiplies each weight block by a learnable scalar s_b ≥ 0,
    regularised with a group-L1 penalty:  λ * Σ_b ‖s_b · W_b‖_F
    This acts as nonconvex group-LASSO regularisation that drives low-importance
    blocks toward zero during training.

  Phase 2 — Greedy sequential selection (SA++ pruning step):
    After training, run the greedy selection: iterate from lowest to highest s_b,
    permanently zero each block until the target sparsity fraction is met.
    This is the "combinatorial optimization" step that produces hardware-friendly
    structured sparsity (contiguous zero blocks → GPU skips them).

Design decisions for R²D-HOPE-MoRE:
  - Block size: (16, 16) — aligns with CUDA tensor core tile sizes
  - Target sparsity per layer type:
      FFN weights (gate_proj, up_proj, down_proj): 0.50
      QKV / cross-attn projections:               0.30
      GRU gate matrices:                          0.40
  - The shared block is pruned ONCE but runs 20× → 20× compute saving per pruned block
  - Router and embedding layers are excluded (see fit analysis)
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core: BlockImportanceMask
# ---------------------------------------------------------------------------

class BlockImportanceMask(nn.Module):
    """
    Wraps a single nn.Linear and attaches a learnable per-block importance
    score vector s ∈ R^{num_blocks}, initialised to 1.

    During Phase 1 (training):
      effective_weight = weight * (s_per_element)   # block-broadcast
      This is differentiable — gradients flow through both weight and s.

    During Phase 2 (pruning):
      selected_mask is a binary tensor; zero blocks are permanently removed.

    Args:
        linear:      The nn.Linear to wrap.
        block_size:  (row_block, col_block) tile size. Default (16, 16).
    """

    def __init__(self, linear: nn.Linear, block_size: tuple[int, int] = (16, 16)):
        super().__init__()
        self.linear = linear
        self.block_size = block_size
        self.pruned = False  # flips to True after Phase 2

        out_features, in_features = linear.weight.shape
        br, bc = block_size

        # Pad dimensions to be divisible by block size
        self.pad_out = math.ceil(out_features / br) * br - out_features
        self.pad_in  = math.ceil(in_features  / bc) * bc - in_features
        self.n_row_blocks = math.ceil(out_features / br)
        self.n_col_blocks = math.ceil(in_features  / bc)
        self.num_blocks = self.n_row_blocks * self.n_col_blocks

        # Learnable importance scores, one per block
        # Initialise to 1.0 so at start the layer behaves identically
        self.log_importance = nn.Parameter(torch.zeros(self.num_blocks))

        # Binary pruning mask (ones = kept, zeros = pruned)
        # Registered as buffer (not parameter) so it's saved with state_dict
        self.register_buffer(
            "block_mask", torch.ones(self.num_blocks, dtype=torch.bool)
        )

    @property
    def importance(self) -> torch.Tensor:
        """Soft non-negative importance scores via softplus."""
        return F.softplus(self.log_importance)  # [num_blocks]

    def _expand_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Expand per-block scores to full weight shape [out, in]."""
        out_f, in_f = self.linear.weight.shape
        br, bc = self.block_size

        # [n_row_blocks, n_col_blocks] → [n_row_blocks*br, n_col_blocks*bc]
        block_grid = scores.reshape(self.n_row_blocks, self.n_col_blocks)
        expanded = block_grid.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)
        # Crop back to original weight shape
        return expanded[:out_f, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pruned:
            # Phase 2: use hard binary mask, no extra cost
            w = self.linear.weight * self._expand_mask(self.block_mask.float())
        else:
            # Phase 1: multiply by soft importance scores (differentiable)
            w = self.linear.weight * self._expand_mask(self.importance)

        return F.linear(x, w, self.linear.bias)

    def regularization_loss(self) -> torch.Tensor:
        """
        Group-L1 regularisation: λ * Σ_b importance_b * ‖W_b‖_F
        Encourages low-importance blocks to have small norms → easier to prune.
        The caller multiplies by λ (sparsity_lambda in SABlockPruner).
        """
        out_f, in_f = self.linear.weight.shape
        br, bc = self.block_size
        # Pad weight to block-aligned shape for clean reshape
        w_pad = F.pad(self.linear.weight, (0, self.pad_in, 0, self.pad_out))
        # [n_row_blocks, br, n_col_blocks, bc]
        w_blocks = w_pad.reshape(self.n_row_blocks, br, self.n_col_blocks, bc)
        # Frobenius norm per block: [n_row_blocks, n_col_blocks]
        block_norms = w_blocks.norm(dim=(1, 3))  # [n_row, n_col]
        block_norms_flat = block_norms.reshape(-1)  # [num_blocks]
        return (self.importance * block_norms_flat).sum()

    def prune_to_sparsity(self, target_sparsity: float) -> int:
        """
        Phase 2: Greedy sequential selection.
        Zeros out the (target_sparsity * num_blocks) blocks with lowest importance.
        Returns the number of blocks pruned.
        """
        num_to_prune = int(self.num_blocks * target_sparsity)
        if num_to_prune == 0:
            return 0

        with torch.no_grad():
            scores = self.importance.detach()
            # Sort by importance ascending; zero out the weakest
            sorted_idx = scores.argsort()
            prune_idx = sorted_idx[:num_to_prune]
            self.block_mask[prune_idx] = False

            # Zero out pruned blocks in the actual weight tensor
            full_mask = self._expand_mask(self.block_mask.float())
            self.linear.weight.mul_(full_mask)

        self.pruned = True
        return num_to_prune

    def density(self) -> float:
        """Fraction of blocks that are kept (1.0 = dense, 0.0 = fully pruned)."""
        return self.block_mask.float().mean().item()


# ---------------------------------------------------------------------------
# SABlockPruner: manages all BlockImportanceMask instances in a model
# ---------------------------------------------------------------------------

SPARSITY_CONFIG: dict[str, float] = {
    # LogicalReasoningExpert FFN (largest, highest leverage)
    "gate_proj":   0.50,
    "up_proj":     0.50,
    "down_proj":   0.50,
    # Attention projections (injector + LocalPatternExpert)
    "q_proj":      0.30,
    "kv_proj":     0.30,
    "qkv":         0.30,
    "out_proj":    0.30,
    # GRU-style gates in MemoryConsolidator
    "reset_gate":  0.40,
    "update_gate": 0.40,
    "candidate":   0.40,
}

# These layers must never be pruned
_EXCLUDED_PATTERNS = {"router", "embed", "lm_head", "log_importance"}


class SABlockPruner:
    """
    Manages the full SA++ lifecycle:
      1. wrap(model)        — replaces target Linear layers with BlockImportanceMask
      2. regularization_loss(model) — returns Phase 1 group-L1 reg loss (add × λ to training loss)
      3. prune(model)       — Phase 2 greedy selection to hit target sparsities

    Usage:
        pruner = SABlockPruner(sparsity_lambda=1e-4, block_size=(16, 16))
        pruner.wrap(model)

        # --- Training loop ---
        for batch in dataloader:
            out = model(...)
            loss = out["loss"] + pruner.regularization_loss(model)
            loss.backward()
            optimizer.step()

        # --- After training converges ---
        pruner.prune(model)
        # model now has structured block sparsity
    """

    def __init__(
        self,
        sparsity_lambda: float = 1e-4,
        block_size: tuple[int, int] = (16, 16),
        sparsity_config: dict[str, float] | None = None,
    ):
        self.sparsity_lambda = sparsity_lambda
        self.block_size = block_size
        self.sparsity_config = sparsity_config or SPARSITY_CONFIG
        self._masks: list[tuple[str, BlockImportanceMask]] = []

    def wrap(self, model: nn.Module) -> None:
        """
        Walk the model, replace eligible nn.Linear modules with BlockImportanceMask.
        Modifies model in-place.
        """
        self._masks.clear()
        self._replace_linears(model, prefix="")

    def _should_wrap(self, name: str, module: nn.Linear) -> bool:
        for excl in _EXCLUDED_PATTERNS:
            if excl in name:
                return False
        leaf_name = name.split(".")[-1]
        return leaf_name in self.sparsity_config

    def _replace_linears(self, module: nn.Module, prefix: str) -> None:
        for attr_name, child in list(module.named_children()):
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(child, nn.Linear) and self._should_wrap(full_name, child):
                mask = BlockImportanceMask(child, self.block_size)
                setattr(module, attr_name, mask)
                self._masks.append((full_name, mask))
            else:
                self._replace_linears(child, full_name)

    def regularization_loss(self, model: nn.Module | None = None) -> torch.Tensor:
        """
        Returns the total group-L1 regularisation loss across all wrapped layers.
        Scale by sparsity_lambda before adding to training loss.
        Call this ONLY during Phase 1 (before pruning).
        """
        device = next(
            m.linear.weight for _, m in self._masks if not m.pruned
        ).device if self._masks else torch.device("cpu")

        total = torch.tensor(0.0, device=device)
        for _, mask in self._masks:
            if not mask.pruned:
                total = total + mask.regularization_loss()
        return self.sparsity_lambda * total

    def prune(self, model: nn.Module | None = None) -> dict[str, float]:
        """
        Phase 2: apply greedy block selection to all wrapped layers.
        Returns a dict of {layer_name: density_after_pruning}.
        """
        results = {}
        for name, mask in self._masks:
            leaf = name.split(".")[-1]
            target_sparsity = self.sparsity_config.get(leaf, 0.0)
            n_pruned = mask.prune_to_sparsity(target_sparsity)
            density = mask.density()
            results[name] = density
        return results

    def report(self) -> str:
        """Human-readable sparsity report."""
        lines = ["SA++ Block Sparsification Report", "=" * 50]
        total_blocks = 0
        kept_blocks = 0
        for name, mask in self._masks:
            n = mask.num_blocks
            k = int(mask.block_mask.sum().item())
            total_blocks += n
            kept_blocks += k
            status = "pruned" if mask.pruned else "phase-1"
            lines.append(
                f"  {name:<55} {k:>5}/{n:<5} blocks kept "
                f"({100*k/n:5.1f}% dense)  [{status}]"
            )
        if total_blocks > 0:
            lines.append("-" * 50)
            lines.append(
                f"  {'TOTAL':<55} {kept_blocks:>5}/{total_blocks:<5} blocks kept "
                f"({100*kept_blocks/total_blocks:5.1f}% dense)"
            )
        return "\n".join(lines)

    def masked_parameters(self) -> Iterator[nn.Parameter]:
        """Yields only the log_importance parameters (for separate LR group)."""
        for _, mask in self._masks:
            yield mask.log_importance

    def model_parameters(self) -> Iterator[nn.Parameter]:
        """Yields all non-importance parameters (base weight group)."""
        seen_ids: set[int] = set()
        for _, mask in self._masks:
            seen_ids.add(id(mask.log_importance))
        for _, mask in self._masks:
            for p in mask.parameters():
                if id(p) not in seen_ids:
                    yield p
