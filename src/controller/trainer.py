"""Controller training with REINFORCE."""
import torch
import torch.nn as nn
from typing import Dict, List

class ControllerTrainer:
    def __init__(self, controller, learning_rate=3.5e-4):
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
        self.baseline = 0.0

    def train_step(self, log_probs, rewards):
        advantages = rewards - self.baseline
        policy_loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.baseline = 0.95 * self.baseline + 0.05 * rewards.mean().item()

        return {"policy_loss": policy_loss.item(), "baseline": self.baseline}
