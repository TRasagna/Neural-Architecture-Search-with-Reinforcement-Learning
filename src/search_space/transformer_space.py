"""Transformer search space."""
from .base_space import BaseSearchSpace
import random

class TransformerSearchSpace(BaseSearchSpace):
    def __init__(self, max_layers=8):
        super().__init__("transformer", max_layers)
        self.operations = ['attention', 'feed_forward']
        self.num_heads_choices = [2, 4, 8, 12, 16]
        self.hidden_size_choices = [128, 256, 512, 768]

    def get_operations(self):
        return self.operations

    def validate_architecture(self, arch):
        return len(arch.get('layers', [])) <= self.max_layers

    def get_search_space_size(self):
        return 10**6

    def sample_random_architecture(self):
        num_layers = random.randint(1, self.max_layers)
        layers = []
        for i in range(num_layers):
            layers.append({
                'num_heads': random.choice(self.num_heads_choices),
                'hidden_size': random.choice(self.hidden_size_choices),
                'ff_size': random.choice([512, 1024, 2048])
            })
        return {'layers': layers}
