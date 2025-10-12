"""RNN-based Controller for Neural Architecture Search.

Implements the ENAS-style LSTM controller that learns to generate
neural network architectures by sampling from learned distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..search_space.base_space import BaseSearchSpace
from ..utils.metrics import calculate_reward


class RNNController(nn.Module):
    """LSTM-based controller for architecture search.

    The controller learns to generate architecture specifications by
    sampling from learned probability distributions. Training uses
    REINFORCE with baseline.
    """

    def __init__(
        self,
        search_space: BaseSearchSpace,
        hidden_size: int = 256,
        num_layers: int = 2,
        temperature: float = 1.0,
        tanh_constant: float = 1.5,
        op_tanh_reduce: float = 2.5,
    ):
        super().__init__()

        self.search_space = search_space
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce

        # Get search space dimensions
        self.num_ops = len(search_space.operations)
        self.max_layers = search_space.max_layers

        # LSTM Controller
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Embedding for operation tokens
        self.op_embedding = nn.Embedding(self.num_ops, hidden_size)

        # Output heads for different architecture decisions
        self.operation_head = nn.Linear(hidden_size, self.num_ops)
        self.skip_head = nn.Linear(hidden_size, 2)  # Binary: skip or not

        # Layer-specific heads (for CNN)
        if hasattr(search_space, 'num_filters_choices'):
            self.filter_head = nn.Linear(hidden_size, len(search_space.num_filters_choices))
        if hasattr(search_space, 'kernel_size_choices'):
            self.kernel_head = nn.Linear(hidden_size, len(search_space.kernel_size_choices))

        # Transformer-specific heads
        if hasattr(search_space, 'num_heads_choices'):
            self.heads_head = nn.Linear(hidden_size, len(search_space.num_heads_choices))
        if hasattr(search_space, 'hidden_size_choices'):
            self.hidden_head = nn.Linear(hidden_size, len(search_space.hidden_size_choices))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize controller parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.08, 0.08)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        batch_size: int = 1,
        max_length: Optional[int] = None
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Generate architecture specifications.

        Args:
            batch_size: Number of architectures to generate
            max_length: Maximum sequence length (uses search_space default if None)

        Returns:
            architectures: Dict containing architecture specifications
            log_probs: Log probabilities of sampled actions
            entropies: Entropy of action distributions
        """
        if max_length is None:
            max_length = self.max_layers * 4  # Operations + hyperparams per layer

        device = next(self.parameters()).device

        # Initialize hidden state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        hidden = (h_0, c_0)

        # Start token (learnable embedding)
        start_token = torch.zeros(batch_size, 1, self.hidden_size, device=device)

        architectures = []
        log_probs = []
        entropies = []

        input_token = start_token

        for step in range(max_length):
            # LSTM forward pass
            output, hidden = self.lstm(input_token, hidden)
            output = output.squeeze(1)  # [batch_size, hidden_size]

            # Generate architecture decisions
            step_arch, step_log_prob, step_entropy = self._sample_architecture_step(
                output, step
            )

            architectures.append(step_arch)
            log_probs.append(step_log_prob)
            entropies.append(step_entropy)

            # Prepare input for next step
            input_token = self._get_next_input(step_arch, output.unsqueeze(1))

            # Early stopping condition
            if self._should_stop(step_arch, step):
                break

        # Combine results
        combined_architectures = self._combine_architecture_steps(architectures)
        total_log_probs = torch.stack(log_probs).sum(dim=0)
        total_entropies = torch.stack(entropies).sum(dim=0)

        return combined_architectures, total_log_probs, total_entropies

    def _sample_architecture_step(
        self, 
        hidden_state: torch.Tensor, 
        step: int
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Sample architecture decisions for current step."""
        batch_size = hidden_state.size(0)

        # Determine what to sample based on step
        step_type = self._get_step_type(step)

        if step_type == 'operation':
            return self._sample_operation(hidden_state)
        elif step_type == 'skip':
            return self._sample_skip_connection(hidden_state)
        elif step_type == 'hyperparams':
            return self._sample_hyperparameters(hidden_state)
        else:
            # Default: operation sampling
            return self._sample_operation(hidden_state)

    def _sample_operation(self, hidden_state: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Sample operation type."""
        logits = self.operation_head(hidden_state)
        logits = torch.tanh(logits / self.op_tanh_reduce) * self.tanh_constant

        # Apply temperature
        logits = logits / self.temperature

        # Sample operation
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        operation_idx = dist.sample()

        # Calculate log prob and entropy
        log_prob = dist.log_prob(operation_idx)
        entropy = dist.entropy()

        # Convert to operation names
        operations = [self.search_space.operations[idx.item()] for idx in operation_idx]

        arch_step = {'type': 'operation', 'operations': operations}

        return arch_step, log_prob, entropy

    def _sample_skip_connection(self, hidden_state: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Sample skip connection decisions."""
        logits = self.skip_head(hidden_state)
        logits = torch.tanh(logits) * self.tanh_constant / self.temperature

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        skip_decision = dist.sample()

        log_prob = dist.log_prob(skip_decision)
        entropy = dist.entropy()

        skip_flags = [bool(decision.item()) for decision in skip_decision]

        arch_step = {'type': 'skip', 'skip_connections': skip_flags}

        return arch_step, log_prob, entropy

    def _sample_hyperparameters(self, hidden_state: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Sample hyperparameters (filters, kernel size, etc.)."""
        arch_step = {'type': 'hyperparams'}
        total_log_prob = 0
        total_entropy = 0

        # Sample number of filters (for CNN)
        if hasattr(self, 'filter_head'):
            logits = self.filter_head(hidden_state) / self.temperature
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            filter_idx = dist.sample()

            total_log_prob += dist.log_prob(filter_idx)
            total_entropy += dist.entropy()

            filters = [self.search_space.num_filters_choices[idx.item()] for idx in filter_idx]
            arch_step['num_filters'] = filters

        # Sample kernel size (for CNN)
        if hasattr(self, 'kernel_head'):
            logits = self.kernel_head(hidden_state) / self.temperature
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            kernel_idx = dist.sample()

            total_log_prob += dist.log_prob(kernel_idx)
            total_entropy += dist.entropy()

            kernels = [self.search_space.kernel_size_choices[idx.item()] for idx in kernel_idx]
            arch_step['kernel_sizes'] = kernels

        # Sample attention heads (for Transformer)
        if hasattr(self, 'heads_head'):
            logits = self.heads_head(hidden_state) / self.temperature
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            heads_idx = dist.sample()

            total_log_prob += dist.log_prob(heads_idx)
            total_entropy += dist.entropy()

            num_heads = [self.search_space.num_heads_choices[idx.item()] for idx in heads_idx]
            arch_step['num_heads'] = num_heads

        return arch_step, total_log_prob, total_entropy

    def _get_step_type(self, step: int) -> str:
        """Determine what type of decision to make at current step."""
        # Simple pattern: operation -> hyperparams -> skip -> operation -> ...
        cycle_pos = step % 3
        if cycle_pos == 0:
            return 'operation'
        elif cycle_pos == 1:
            return 'hyperparams'
        else:
            return 'skip'

    def _get_next_input(self, arch_step: Dict, current_output: torch.Tensor) -> torch.Tensor:
        """Prepare input token for next LSTM step."""
        # Use current LSTM output as next input (with some transformation)
        return current_output

    def _should_stop(self, arch_step: Dict, step: int) -> bool:
        """Determine if architecture generation should stop."""
        # Stop after generating enough layers
        layer_count = (step + 1) // 3
        return layer_count >= self.max_layers

    def _combine_architecture_steps(self, architecture_steps: List[Dict]) -> Dict:
        """Combine individual steps into complete architecture specifications."""
        batch_size = len(architecture_steps[0]['operations']) if architecture_steps else 1

        combined = []
        for batch_idx in range(batch_size):
            arch_spec = {
                'layers': [],
                'skip_connections': [],
                'global_config': {}
            }

            current_layer = {}
            for step in architecture_steps:
                if step['type'] == 'operation':
                    current_layer['operation'] = step['operations'][batch_idx]
                elif step['type'] == 'hyperparams':
                    if 'num_filters' in step:
                        current_layer['num_filters'] = step['num_filters'][batch_idx]
                    if 'kernel_sizes' in step:
                        current_layer['kernel_size'] = step['kernel_sizes'][batch_idx]
                    if 'num_heads' in step:
                        current_layer['num_heads'] = step['num_heads'][batch_idx]
                elif step['type'] == 'skip':
                    arch_spec['skip_connections'].append(step['skip_connections'][batch_idx])
                    # Complete current layer and start new one
                    if current_layer:
                        arch_spec['layers'].append(current_layer)
                        current_layer = {}

            # Add final layer if exists
            if current_layer:
                arch_spec['layers'].append(current_layer)

            combined.append(arch_spec)

        return combined


def create_controller(config: Dict, search_space: BaseSearchSpace) -> RNNController:
    """Factory function to create RNN controller from config."""
    return RNNController(
        search_space=search_space,
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 2),
        temperature=config.get('temperature', 1.0),
        tanh_constant=config.get('tanh_constant', 1.5),
        op_tanh_reduce=config.get('op_tanh_reduce', 2.5),
    )
