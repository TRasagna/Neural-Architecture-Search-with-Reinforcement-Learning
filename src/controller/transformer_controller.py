"""Transformer-based Controller for Neural Architecture Search.

Alternative to RNN controller using multi-head attention for better
long-range dependencies in architecture generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from ..search_space.base_space import BaseSearchSpace


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer controller."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerController(nn.Module):
    """Transformer-based architecture controller.

    Uses self-attention to capture dependencies between architecture
    decisions across different layers and components.
    """

    def __init__(
        self,
        search_space: BaseSearchSpace,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_length: int = 1000,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.search_space = search_space
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temperature = temperature
        self.max_length = max_length

        # Search space dimensions
        self.num_ops = len(search_space.operations)
        self.max_arch_layers = search_space.max_layers

        # Input embeddings
        self.op_embedding = nn.Embedding(self.num_ops + 1, d_model)  # +1 for special tokens
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output heads
        self.operation_head = nn.Linear(d_model, self.num_ops)
        self.skip_head = nn.Linear(d_model, 2)

        # CNN-specific heads
        if hasattr(search_space, 'num_filters_choices'):
            self.filter_head = nn.Linear(d_model, len(search_space.num_filters_choices))
        if hasattr(search_space, 'kernel_size_choices'):
            self.kernel_head = nn.Linear(d_model, len(search_space.kernel_size_choices))

        # Transformer-specific heads
        if hasattr(search_space, 'num_heads_choices'):
            self.heads_head = nn.Linear(d_model, len(search_space.num_heads_choices))
        if hasattr(search_space, 'hidden_size_choices'):
            self.hidden_head = nn.Linear(d_model, len(search_space.hidden_size_choices))

        # Special tokens
        self.start_token_id = self.num_ops
        self.pad_token_id = 0

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize transformer parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        batch_size: int = 1,
        max_length: Optional[int] = None
    ) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
        """Generate architecture specifications using transformer.

        Args:
            batch_size: Number of architectures to generate
            max_length: Maximum sequence length

        Returns:
            architectures: List of architecture specifications
            log_probs: Log probabilities of sampled sequences
            entropies: Entropy of action distributions
        """
        if max_length is None:
            max_length = self.max_arch_layers * 3  # operation + hyperparams + skip per layer

        device = next(self.parameters()).device

        # Initialize with start tokens
        sequences = torch.full(
            (batch_size, 1), 
            self.start_token_id, 
            dtype=torch.long, 
            device=device
        )

        log_probs = []
        entropies = []

        for step in range(max_length):
            # Get embeddings and add positional encoding
            embeddings = self.op_embedding(sequences)
            embeddings = self.pos_encoding(embeddings)

            # Create causal mask for autoregressive generation
            seq_len = sequences.size(1)
            mask = self._generate_square_subsequent_mask(seq_len).to(device)

            # Transformer forward pass
            output = self.transformer(embeddings, mask)

            # Use last position for next token prediction
            last_hidden = output[:, -1, :]  # [batch_size, d_model]

            # Sample next token based on step type
            step_type = self._get_step_type(step)
            next_tokens, step_log_probs, step_entropies = self._sample_next_token(
                last_hidden, step_type
            )

            # Append to sequences
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            log_probs.append(step_log_probs)
            entropies.append(step_entropies)

        # Convert sequences to architecture specifications
        architectures = self._sequences_to_architectures(sequences, batch_size)

        # Combine log probs and entropies
        total_log_probs = torch.stack(log_probs).sum(dim=0)
        total_entropies = torch.stack(entropies).sum(dim=0)

        return architectures, total_log_probs, total_entropies

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for transformer."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _get_step_type(self, step: int) -> str:
        """Determine decision type for current step."""
        cycle_pos = step % 3
        if cycle_pos == 0:
            return 'operation'
        elif cycle_pos == 1:
            return 'hyperparams'
        else:
            return 'skip'

    def _sample_next_token(
        self, 
        hidden_state: torch.Tensor, 
        step_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample next token based on step type."""
        if step_type == 'operation':
            return self._sample_operation_token(hidden_state)
        elif step_type == 'hyperparams':
            return self._sample_hyperparameter_token(hidden_state)
        else:  # skip
            return self._sample_skip_token(hidden_state)

    def _sample_operation_token(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample operation tokens."""
        logits = self.operation_head(hidden_state) / self.temperature
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        tokens = dist.sample()
        log_probs = dist.log_prob(tokens)  
        entropies = dist.entropy()

        return tokens, log_probs, entropies

    def _sample_hyperparameter_token(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hyperparameter tokens."""
        # For simplicity, sample from filter choices if available
        if hasattr(self, 'filter_head'):
            logits = self.filter_head(hidden_state) / self.temperature
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            tokens = dist.sample()
            log_probs = dist.log_prob(tokens)
            entropies = dist.entropy()

            return tokens, log_probs, entropies
        else:
            # Default to operation sampling
            return self._sample_operation_token(hidden_state)

    def _sample_skip_token(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample skip connection tokens."""
        logits = self.skip_head(hidden_state) / self.temperature
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        tokens = dist.sample()
        log_probs = dist.log_prob(tokens)
        entropies = dist.entropy()

        return tokens, log_probs, entropies

    def _sequences_to_architectures(self, sequences: torch.Tensor, batch_size: int) -> List[Dict]:
        """Convert token sequences to architecture specifications."""
        architectures = []

        for batch_idx in range(batch_size):
            sequence = sequences[batch_idx].cpu().numpy()[1:]  # Remove start token

            arch_spec = {
                'layers': [],
                'skip_connections': [],
                'global_config': {}
            }

            # Parse sequence into architecture components
            for i in range(0, len(sequence), 3):
                if i + 2 < len(sequence):
                    op_token = sequence[i]
                    hp_token = sequence[i + 1] 
                    skip_token = sequence[i + 2]

                    # Build layer specification
                    layer = {
                        'operation': self.search_space.operations[op_token % self.num_ops]
                    }

                    # Add hyperparameters
                    if hasattr(self.search_space, 'num_filters_choices'):
                        filter_idx = hp_token % len(self.search_space.num_filters_choices)
                        layer['num_filters'] = self.search_space.num_filters_choices[filter_idx]

                    arch_spec['layers'].append(layer)
                    arch_spec['skip_connections'].append(bool(skip_token % 2))

            architectures.append(arch_spec)

        return architectures


def create_transformer_controller(config: Dict, search_space: BaseSearchSpace) -> TransformerController:
    """Factory function to create transformer controller from config."""
    return TransformerController(
        search_space=search_space,
        d_model=config.get('d_model', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1),
        temperature=config.get('temperature', 1.0),
    )
