"""Ray Tune integration for distributed NAS."""

import ray
from ray import tune
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, Any, List, Optional

from ..train.lightning_trainer import LightningChildModel
from ..controller.rnn_controller import RNNController
from ..controller.transformer_controller import TransformerController
from ..search_space.cnn_space import CNNSearchSpace
from ..search_space.transformer_space import TransformerSearchSpace
from ..utils.config import load_config
from ..utils.metrics import calculate_reward


class RayNASTrainer:
    """Ray Tune orchestrated NAS training."""

    def __init__(self, config: Dict):
        self.config = config
        self.search_space = self._create_search_space()
        self.controller = self._create_controller()

    def _create_search_space(self):
        """Create search space from config."""
        space_config = self.config['search']
        space_type = space_config.get('search_space', 'cnn_basic')

        if 'cnn' in space_type:
            return CNNSearchSpace(
                max_layers=space_config.get('max_layers', 12)
            )
        elif 'transformer' in space_type:
            return TransformerSearchSpace(
                max_layers=space_config.get('max_layers', 8)
            )
        else:
            raise ValueError(f"Unknown search space: {space_type}")

    def _create_controller(self):
        """Create controller from config."""
        controller_config = self.config['controller']
        controller_type = controller_config.get('type', 'rnn')

        if controller_type == 'rnn':
            return RNNController(
                search_space=self.search_space,
                **controller_config
            )
        elif controller_type == 'transformer':
            return TransformerController(
                search_space=self.search_space,
                **controller_config
            )
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

    def run_search(self):
        """Run distributed architecture search."""
        ray_config = self.config['ray']

        # Define search space for Ray Tune
        search_config = {
            "architecture_spec": tune.sample_from(
                lambda spec: self._sample_architecture()
            ),
            "config": self.config
        }

        # Configure Ray Train
        scaling_config = ScalingConfig(
            num_workers=1,
            use_gpu=ray_config.get('num_gpus_per_trial', 1) > 0,
            resources_per_worker={
                "CPU": ray_config.get('num_cpus_per_trial', 4),
                "GPU": ray_config.get('num_gpus_per_trial', 1)
            }
        )

        # Configure training
        run_config = RunConfig(
            name="nas_search",
            local_dir=ray_config.get('local_dir', './ray_results'),
            checkpoint_config={"checkpoint_frequency": 5}
        )

        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self._train_architecture,
            scaling_config=scaling_config,
            run_config=run_config
        )

        # Run hyperparameter search
        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": search_config},
            tune_config=tune.TuneConfig(
                num_samples=self.config['search']['num_architectures'],
                max_concurrent_trials=ray_config.get('max_concurrent_trials', 4),
                metric="val_acc",
                mode="max"
            )
        )

        results = tuner.fit()

        # Process results
        best_result = results.get_best_result("val_acc", "max")
        print(f"Best validation accuracy: {best_result.metrics['val_acc']:.4f}")

        return results

    def _sample_architecture(self) -> Dict:
        """Sample architecture using controller."""
        with torch.no_grad():
            architectures, _, _ = self.controller(batch_size=1)
            return architectures[0]

    def _train_architecture(self, config: Dict):
        """Train single architecture (Ray worker function)."""
        architecture_spec = config["architecture_spec"]
        train_config = config["config"]

        # Create Lightning model
        model = LightningChildModel(
            architecture=architecture_spec,
            search_space=self.search_space,
            **train_config.get('child', {})
        )

        # Create data loaders (simplified - would need actual implementation)
        train_loader, val_loader = self._create_data_loaders(train_config)

        # Configure callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_acc",
                patience=train_config['search'].get('early_stopping_patience', 5),
                mode="max"
            ),
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1
            )
        ]

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=train_config['search']['max_epochs_per_arch'],
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,  # Ray handles logging
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision=train_config.get('compute', {}).get('precision', 32)
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Get final metrics
        val_metrics = trainer.validate(model, val_loader)[0]

        # Calculate reward for controller
        reward = calculate_reward(
            accuracy=val_metrics['val_acc'],
            architecture=architecture_spec,
            objectives=train_config['search'].get('objectives', [])
        )

        # Report to Ray Tune
        return {
            "val_acc": val_metrics['val_acc'],
            "val_loss": val_metrics['val_loss'],
            "reward": reward,
            "architecture": architecture_spec
        }

    def _create_data_loaders(self, config: Dict):
        """Create data loaders - simplified implementation."""
        # This would need full implementation based on dataset config
        from torch.utils.data import DataLoader, TensorDataset

        # Dummy data for illustration
        batch_size = config['dataset']['batch_size']

        if config['dataset']['name'] == 'cifar10':
            # Dummy CIFAR-10 like data
            train_data = torch.randn(1000, 3, 32, 32)
            train_labels = torch.randint(0, 10, (1000,))
            val_data = torch.randn(200, 3, 32, 32)
            val_labels = torch.randint(0, 10, (200,))
        else:
            # Dummy text data
            train_data = torch.randint(0, 1000, (1000, 128))
            train_labels = torch.randint(0, 2, (1000,))
            val_data = torch.randint(0, 1000, (200, 128))
            val_labels = torch.randint(0, 2, (200,))

        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


def run_distributed_search(config_path: str):
    """Entry point for distributed search."""
    config = load_config(config_path)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=config['ray'].get('num_cpus', 8),
            num_gpus=config['ray'].get('num_gpus', 4)
        )

    # Run search
    trainer = RayNASTrainer(config)
    results = trainer.run_search()

    return results
