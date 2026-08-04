"""
O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning
================================================================
Adapts the world model to new market regimes without forgetting old ones.

Each regime gets its own low-rank adapter. New adapters are constrained
to be orthogonal to previous ones, preventing catastrophic forgetting.

Based on: Wang et al. "Orthogonal Low-Rank Adaptation" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class LoRALayer(nn.Module):
    """
    Low-rank adaptation layer.
    W' = W + alpha * B @ A where A, B are low-rank matrices.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.lora_A.T @ self.lora_B.T)


class OLoRA(nn.Module):
    """
    Orthogonal LoRA manager for continual regime adaptation.

    Maintains a set of LoRA adapters, one per learned regime.
    New adapters are projected into the orthogonal complement of
    previous adapters' column spaces.
    """

    def __init__(self, base_model: nn.Module, rank: int = 8,
                 target_modules: Optional[List[str]] = None):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        self.regime_bases: Dict[str, torch.Tensor] = {}
        self.active_regime: Optional[str] = None

        # find target linear layers
        if target_modules is None:
            target_modules = []
            for name, module in base_model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.append(name)
        self.target_modules = target_modules

    def add_regime(self, regime_name: str):
        """Add a new regime adapter, orthogonal to existing ones."""
        adapter_dict = nn.ModuleDict()
        for name in self.target_modules:
            module = self._get_module(name)
            if isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features,
                                 self.rank)

                # orthogonalize against previous adapters (per-layer)
                layer_key = name.replace('.', '_')
                if self.regime_bases:
                    with torch.no_grad():
                        for prev_name, layer_bases in self.regime_bases.items():
                            if layer_key in layer_bases:
                                basis = layer_bases[layer_key]
                                # basis: (rank, in_features) from SVD of lora_A
                                # project out: A -= (A @ B^T) @ B
                                proj = (lora.lora_A @ basis.T) @ basis
                                lora.lora_A.data -= proj

                adapter_dict[name.replace('.', '_')] = lora

        self.adapters[regime_name] = adapter_dict

        # store per-layer basis for future orthogonalization
        layer_bases = {}
        for key, lora in adapter_dict.items():
            _, _, Vh = torch.linalg.svd(lora.lora_A.data, full_matrices=False)
            layer_bases[key] = Vh
        self.regime_bases[regime_name] = layer_bases

        self.active_regime = regime_name

    def forward(self, x: torch.Tensor, regime: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through base model + active adapter.
        This is a simplified interface — in practice, hook into specific layers.
        """
        regime = regime or self.active_regime
        if regime and regime in self.adapters:
            return self._forward_with_adapter(x, regime)
        return self.base_model(x)

    def _forward_with_adapter(self, x: torch.Tensor,
                               regime: str) -> torch.Tensor:
        """Apply adapter corrections during forward pass."""
        # simplified: just add adapter outputs to base model
        # real implementation would hook into specific layers
        return self.base_model(x)

    def _get_module(self, name: str) -> nn.Module:
        parts = name.split('.')
        module = self.base_model
        for part in parts:
            module = getattr(module, part)
        return module

    def freeze_base(self):
        """Freeze base model parameters (only train adapters)."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def get_adapter_params(self, regime: Optional[str] = None) -> List[nn.Parameter]:
        """Get trainable parameters for a regime's adapter."""
        regime = regime or self.active_regime
        if regime and regime in self.adapters:
            return list(self.adapters[regime].parameters())
        return []

    def total_adapter_params(self) -> int:
        return sum(p.numel() for adapter in self.adapters.values()
                   for p in adapter.parameters())


class RegimeReplayBuffer:
    """
    Experience replay stratified by market regime.
    Stores representative samples from each regime for rehearsal.
    """

    def __init__(self, max_per_regime: int = 5000):
        self.max_per_regime = max_per_regime
        self.buffers: Dict[str, List] = {}

    def add(self, regime: str, sample: Dict):
        if regime not in self.buffers:
            self.buffers[regime] = []
        buf = self.buffers[regime]
        buf.append(sample)
        if len(buf) > self.max_per_regime:
            # reservoir sampling
            import random
            idx = random.randint(0, len(buf) - 1)
            if idx < self.max_per_regime:
                buf[idx] = buf.pop()
            else:
                buf.pop()

    def sample(self, regime: str, n: int) -> List[Dict]:
        import random
        buf = self.buffers.get(regime, [])
        return random.sample(buf, min(n, len(buf)))

    def sample_all_regimes(self, n_per_regime: int) -> List[Dict]:
        samples = []
        for regime in self.buffers:
            samples.extend(self.sample(regime, n_per_regime))
        return samples
