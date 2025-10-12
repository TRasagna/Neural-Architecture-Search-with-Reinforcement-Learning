"""Child model builder."""
import torch.nn as nn

class ChildModelBuilder:
    def __init__(self, search_space):
        self.search_space = search_space

    def build_model(self, architecture, **kwargs):
        if 'cnn' in str(type(self.search_space)).lower():
            return CNNModel(architecture, **kwargs)
        else:
            return TransformerModel(architecture, **kwargs)

class CNNModel(nn.Module):
    def __init__(self, architecture, num_classes=10, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()

        in_channels = 3
        for layer_spec in architecture.get('layers', []):
            if layer_spec.get('operation', '').startswith('conv'):
                out_channels = layer_spec.get('num_filters', 64)
                kernel_size = layer_spec.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
                self.layers.append(nn.ReLU())
                in_channels = out_channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

class TransformerModel(nn.Module):
    def __init__(self, architecture, num_classes=2, vocab_size=50000, **kwargs):
        super().__init__()
        first_layer = architecture['layers'][0]
        hidden_size = first_layer['hidden_size']

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList()

        for layer_spec in architecture['layers']:
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=layer_spec['hidden_size'],
                nhead=layer_spec['num_heads'],
                dim_feedforward=layer_spec.get('ff_size', 2048),
                batch_first=True
            ))

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # Simple pooling
        return self.classifier(x)
