"""Metrics calculation utilities."""
import torch
from typing import Dict, List

def calculate_reward(accuracy: float, architecture: Dict, objectives: List[Dict] = None) -> float:
    """Calculate reward for architecture based on multiple objectives."""
    if not objectives:
        return accuracy

    total_reward = 0.0
    for obj in objectives:
        if obj['name'] == 'accuracy':
            reward = accuracy if obj.get('maximize', True) else -accuracy
        elif obj['name'] == 'flops':
            # Simplified FLOP estimation
            num_layers = len(architecture.get('layers', []))
            flops = num_layers * 1e6  # Simplified
            reward = -flops if not obj.get('maximize', False) else flops
        else:
            reward = 0.0

        total_reward += obj.get('weight', 1.0) * reward

    return total_reward
