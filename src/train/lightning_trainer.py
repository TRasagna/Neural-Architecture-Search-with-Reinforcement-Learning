"""PyTorch Lightning trainer."""
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.optim import AdamW

class LightningChildModel(pl.LightningModule):
    def __init__(self, architecture, search_space, num_classes=10, learning_rate=1e-3, **kwargs):
        super().__init__()
        from ..child.builder import ChildModelBuilder

        builder = ChildModelBuilder(search_space)
        self.model = builder.build_model(architecture, num_classes=num_classes, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            logits = self(x)
        else:
            x, mask, y = batch
            logits = self.model(x, mask)

        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            logits = self(x)
        else:
            x, mask, y = batch
            logits = self.model(x, mask)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
