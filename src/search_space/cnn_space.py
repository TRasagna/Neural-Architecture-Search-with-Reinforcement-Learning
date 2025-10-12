"""CNN Search Space for Image Classification.

Defines search space for convolutional neural networks including
operations, hyperparameters, and architecture constraints.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base_space import BaseSearchSpace


class CNNSearchSpace(BaseSearchSpace):
    """Search space for CNN architectures."""

    def __init__(
        self,
        name: str = "cnn_search_space",
        max_layers: int = 12,
        operations: Optional[List[str]] = None,
        num_filters_choices: Optional[List[int]] = None,
        kernel_size_choices: Optional[List[int]] = None,
        stride_choices: Optional[List[int]] = None,
        activation_choices: Optional[List[str]] = None,
        pooling_choices: Optional[List[str]] = None,
        include_skip_connections: bool = True,
        include_batch_norm: bool = True,
        include_dropout: bool = True,
    ):
        super().__init__(name, max_layers)

        # Default operations if not provided
        if operations is None:
            operations = [
                'conv3x3',
                'conv5x5', 
                'conv7x7',
                'dwconv3x3',  # Depthwise separable
                'dwconv5x5',
                'maxpool3x3',
                'avgpool3x3',
                'identity'
            ]

        self.operations = operations

        # Hyperparameter choices
        self.num_filters_choices = num_filters_choices or [16, 32, 64, 128, 256, 512]
        self.kernel_size_choices = kernel_size_choices or [1, 3, 5, 7]
        self.stride_choices = stride_choices or [1, 2]
        self.activation_choices = activation_choices or ['relu', 'swish', 'gelu', 'leaky_relu']
        self.pooling_choices = pooling_choices or ['max', 'avg', 'none']

        # Architecture features
        self.include_skip_connections = include_skip_connections
        self.include_batch_norm = include_batch_norm
        self.include_dropout = include_dropout

        # Add basic constraints
        self._add_default_constraints()

    def get_operations(self) -> List[str]:
        """Return available operations."""
        return self.operations

    def validate_architecture(self, architecture: Dict) -> bool:
        """Validate CNN architecture specification."""
        try:
            # Check required fields
            required_fields = ['layers']
            for field in required_fields:
                if field not in architecture:
                    return False

            # Check layers
            layers = architecture['layers']
            if not isinstance(layers, list) or len(layers) == 0:
                return False

            if len(layers) > self.max_layers:
                return False

            # Validate each layer
            for i, layer in enumerate(layers):
                if not self._validate_layer(layer, i):
                    return False

            # Check constraints
            valid, _ = self.check_constraints(architecture)
            return valid

        except Exception:
            return False

    def _validate_layer(self, layer: Dict, layer_idx: int) -> bool:
        """Validate individual layer specification."""
        # Check operation
        if 'operation' not in layer:
            return False

        operation = layer['operation']
        if operation not in self.operations:
            return False

        # Check operation-specific parameters
        if operation.startswith('conv'):
            # Convolution layers need filters
            if 'num_filters' not in layer:
                return False
            if layer['num_filters'] not in self.num_filters_choices:
                return False

        # Check optional parameters
        if 'kernel_size' in layer:
            if layer['kernel_size'] not in self.kernel_size_choices:
                return False

        if 'stride' in layer:
            if layer['stride'] not in self.stride_choices:
                return False

        if 'activation' in layer:
            if layer['activation'] not in self.activation_choices:
                return False

        return True

    def get_search_space_size(self) -> int:
        """Estimate total search space size."""
        # Rough estimate: operations^layers * hyperparams^layers
        base_ops = len(self.operations)
        hyperparams_per_layer = (
            len(self.num_filters_choices) * 
            len(self.kernel_size_choices) * 
            len(self.stride_choices) *
            len(self.activation_choices)
        )

        total_size = 0
        for num_layers in range(1, self.max_layers + 1):
            layer_combinations = (base_ops * hyperparams_per_layer) ** num_layers
            if self.include_skip_connections:
                # Exponential growth for skip connections
                skip_combinations = 2 ** (num_layers * (num_layers - 1) // 2)
                layer_combinations *= skip_combinations
            total_size += layer_combinations

        return min(total_size, 10**15)  # Cap at reasonable number

    def sample_random_architecture(self) -> Dict:
        """Sample a random valid CNN architecture."""
        num_layers = random.randint(1, self.max_layers)

        layers = []
        for i in range(num_layers):
            layer = self._sample_random_layer(i, num_layers)
            layers.append(layer)

        architecture = {
            'layers': layers,
            'global_config': {
                'input_channels': 3,  # RGB images
                'num_classes': 10,    # Default for CIFAR-10
                'input_size': 32      # Default image size
            }
        }

        # Add skip connections if enabled
        if self.include_skip_connections:
            architecture['skip_connections'] = self._sample_skip_connections(num_layers)

        return architecture

    def _sample_random_layer(self, layer_idx: int, total_layers: int) -> Dict:
        """Sample random layer configuration."""
        # Sample operation
        operation = random.choice(self.operations)

        layer = {
            'operation': operation,
            'layer_idx': layer_idx
        }

        # Add operation-specific parameters
        if operation.startswith('conv') or operation.startswith('dwconv'):
            layer['num_filters'] = random.choice(self.num_filters_choices)

            # Extract kernel size from operation name or sample
            if '3x3' in operation:
                layer['kernel_size'] = 3
            elif '5x5' in operation:
                layer['kernel_size'] = 5
            elif '7x7' in operation:
                layer['kernel_size'] = 7
            else:
                layer['kernel_size'] = random.choice(self.kernel_size_choices)

            layer['stride'] = random.choice(self.stride_choices)

            # Reduce stride probability for later layers
            if layer_idx > total_layers // 2:
                layer['stride'] = 1

        # Add activation
        layer['activation'] = random.choice(self.activation_choices)

        # Add batch normalization and dropout with high probability
        if self.include_batch_norm:
            layer['batch_norm'] = random.random() > 0.2  # 80% chance

        if self.include_dropout:
            if random.random() > 0.7:  # 30% chance
                layer['dropout'] = random.uniform(0.1, 0.5)

        return layer

    def _sample_skip_connections(self, num_layers: int) -> List[List[int]]:
        """Sample skip connection pattern."""
        skip_connections = []

        for i in range(num_layers):
            # Each layer can connect to previous layers
            possible_connections = list(range(i))

            # Sample connections with decreasing probability for distant layers
            connections = []
            for j in possible_connections:
                distance = i - j
                # Higher probability for closer layers
                prob = max(0.1, 0.8 / distance)
                if random.random() < prob:
                    connections.append(j)

            skip_connections.append(connections)

        return skip_connections

    def _add_default_constraints(self):
        """Add default architecture constraints."""

        def check_filter_progression(arch):
            """Ensure filter counts generally increase with depth."""
            layers = arch['layers']
            for i in range(1, len(layers)):
                if 'num_filters' in layers[i] and 'num_filters' in layers[i-1]:
                    # Allow same or increasing filter count
                    if layers[i]['num_filters'] < layers[i-1]['num_filters'] // 2:
                        return False
            return True

        def check_reasonable_depth(arch):
            """Ensure reasonable depth for small images."""
            max_stride_layers = sum(
                1 for layer in arch['layers'] 
                if layer.get('stride', 1) > 1
            )
            # Don't reduce spatial dimensions too much
            return max_stride_layers <= 4

        def check_final_representation(arch):
            """Ensure final feature maps aren't too small."""
            spatial_size = arch.get('global_config', {}).get('input_size', 32)
            for layer in arch['layers']:
                if layer.get('stride', 1) > 1:
                    spatial_size //= layer['stride']
                if 'pooling' in layer and layer['pooling'] in ['max', 'avg']:
                    spatial_size //= 2

            return spatial_size >= 2  # At least 2x2 final feature maps

        self.add_constraint('filter_progression', check_filter_progression)
        self.add_constraint('reasonable_depth', check_reasonable_depth)
        self.add_constraint('final_representation', check_final_representation)

    def get_complexity_estimate(self, architecture: Dict) -> Dict[str, float]:
        """Estimate computational complexity of architecture."""
        total_flops = 0
        total_params = 0

        input_size = architecture.get('global_config', {}).get('input_size', 32)
        current_size = input_size
        in_channels = architecture.get('global_config', {}).get('input_channels', 3)

        for layer in architecture['layers']:
            operation = layer['operation']

            if operation.startswith('conv') or operation.startswith('dwconv'):
                out_channels = layer['num_filters']
                kernel_size = layer['kernel_size']
                stride = layer.get('stride', 1)

                # Calculate FLOPs and parameters
                if operation.startswith('dwconv'):
                    # Depthwise separable convolution
                    # Depthwise
                    flops_dw = current_size * current_size * in_channels * kernel_size * kernel_size
                    params_dw = in_channels * kernel_size * kernel_size

                    # Pointwise
                    flops_pw = current_size * current_size * in_channels * out_channels
                    params_pw = in_channels * out_channels

                    total_flops += flops_dw + flops_pw
                    total_params += params_dw + params_pw
                else:
                    # Standard convolution
                    flops = current_size * current_size * in_channels * out_channels * kernel_size * kernel_size
                    params = in_channels * out_channels * kernel_size * kernel_size

                    total_flops += flops
                    total_params += params

                # Update dimensions
                current_size = current_size // stride
                in_channels = out_channels

        return {
            'flops': total_flops,
            'params': total_params,
            'depth': len(architecture['layers']),
            'final_spatial_size': current_size
        }


def create_cnn_search_space(config: Dict) -> CNNSearchSpace:
    """Factory function to create CNN search space from config."""
    return CNNSearchSpace(
        name=config.get('name', 'cnn_search_space'),
        max_layers=config.get('max_layers', 12),
        operations=config.get('operations'),
        num_filters_choices=config.get('num_filters_choices'),
        kernel_size_choices=config.get('kernel_size_choices'),
        stride_choices=config.get('stride_choices'),
        activation_choices=config.get('activation_choices'),
        pooling_choices=config.get('pooling_choices'),
        include_skip_connections=config.get('include_skip_connections', True),
        include_batch_norm=config.get('include_batch_norm', True),
        include_dropout=config.get('include_dropout', True),
    )
