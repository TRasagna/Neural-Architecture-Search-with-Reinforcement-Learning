"""Model registry for storing best architectures."""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class ModelRegistry:
    def __init__(self, storage_path: str = "./model_registry.json"):
        self.storage_path = storage_path
        self.architectures = self._load_architectures()

    def _load_architectures(self) -> List[Dict]:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return []

    def _save_architectures(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.architectures, f, indent=2)

    def register_architecture(self, architecture_spec: Dict, metrics: Dict, search_id: str = None) -> str:
        arch_id = f"arch_{len(self.architectures):04d}"

        architecture_entry = {
            'id': arch_id,
            'spec': architecture_spec,
            'metrics': metrics,
            'search_id': search_id,
            'created_at': datetime.now().isoformat()
        }

        self.architectures.append(architecture_entry)
        self._save_architectures()

        return arch_id

    def get_best_architectures(self, limit: int = 10, model_type: str = "cnn") -> List[Dict]:
        # Sort by validation accuracy (or other primary metric)
        sorted_archs = sorted(
            self.architectures,
            key=lambda x: x['metrics'].get('val_acc', 0),
            reverse=True
        )
        return sorted_archs[:limit]

    def get_architecture(self, arch_id: str) -> Optional[Dict]:
        for arch in self.architectures:
            if arch['id'] == arch_id:
                return arch
        return None

    def count_architectures(self) -> int:
        return len(self.architectures)
