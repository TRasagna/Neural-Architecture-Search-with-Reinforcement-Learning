# NAS-RL: Neural Architecture Search with Reinforcement Learning

[![CI](https://github.com/your-org/nas-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/nas-rl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A complete, production-ready Neural Architecture Search (NAS) system that uses Reinforcement Learning to automatically discover optimal neural network architectures. Features distributed training, weight sharing, MLOps integration, and deployment infrastructure.

## 🚀 Key Features

- **RL-based Architecture Search**: RNN and Transformer-based controllers using REINFORCE/PPO
- **Efficient Training**: ENAS-style weight sharing and progressive search strategies  
- **Distributed Orchestration**: Ray Tune for scalable architecture evaluation
- **MLOps Integration**: W&B/MLflow tracking, DVC versioning, Prefect workflows
- **Production Ready**: FastAPI serving, Kubernetes deployment, monitoring dashboards
- **Multi-Modal**: Supports CNN (CIFAR-10) and Transformer (NLP) search spaces

## 📁 Project Structure

```
nas-rl/
├── configs/           # Experiment configurations
├── src/
│   ├── controller/    # RL policy networks (RNN/Transformer)
│   ├── search_space/  # Architecture definitions (CNN/Transformer)
│   ├── child/         # Child model builder + weight sharing
│   ├── train/         # PyTorch Lightning training loops
│   ├── ray/          # Ray Tune distributed orchestration
│   ├── mlops/        # W&B, MLflow, DVC integration
│   ├── api/          # FastAPI model serving
│   └── dashboard/    # Streamlit monitoring UI
├── infra/            # Docker, K8s, CI/CD
├── notebooks/        # Tutorials and experiments
└── tests/           # Unit and integration tests
```

## 🛠️ Tech Stack

- **ML/DL**: PyTorch, PyTorch Lightning, Ray Tune
- **MLOps**: Weights & Biases, MLflow, DVC, Prefect
- **Serving**: FastAPI, Docker, Kubernetes (Helm)
- **Monitoring**: Prometheus, Grafana
- **CI/CD**: GitHub Actions

## ⚡ Quick Start

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/your-org/nas-rl.git
cd nas-rl
pip install -r requirements.txt

# 2. Download data
dvc pull
# or
python scripts/download_data.py

# 3. Run development search (CPU, small budget)
python -m src.train.quick_run --config configs/dev.yaml

# 4. Start local services
docker-compose up --build
```

### Production Search

```bash
# Single node with 4 GPUs
python -m src.ray.run_search --config configs/search_cifar10_ray.yaml

# Multi-node cluster (see K8s deployment)
kubectl apply -f infra/k8s/
```

## 📊 Supported Tasks

### 1. CNN Search (CIFAR-10)
- **Search Space**: Convolution cells, skip connections, batch norm, pooling
- **Metrics**: Validation accuracy, FLOPs, inference latency
- **Budget**: 5-50 epochs per architecture evaluation

### 2. Transformer Search (SST-2)  
- **Search Space**: Encoder layers, attention heads, hidden dims, FFN size
- **Metrics**: F1 score, parameters, inference time
- **Budget**: 3-20 epochs per architecture evaluation

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Controller    │───▶│  Child Models   │───▶│   Evaluation    │
│  (RNN/Trans.)   │    │    Builder      │    │   (Training)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │              ┌─────────────────┐             │
         └──────────────│     Reward      │◀────────────┘
                        │   (Val. Acc.)   │
                        └─────────────────┘

Weight Sharing Supernet (ENAS-style):
┌─────────────────────────────────────────────────────────┐
│  Shared Weights: Conv1, Conv2, ..., ConvN              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐               │
│  │ Arch 1  │───│ Arch 2  │───│ Arch N  │  (Subgraphs) │
│  └─────────┘   └─────────┘   └─────────┘               │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Experiments & Results

### Expected Performance (Reference)
- **CIFAR-10**: ~94-96% validation accuracy with 2-5M FLOPs
- **SST-2**: ~88-92% F1 score with <10M parameters  
- **Search Time**: 8-24 GPU hours (depending on search space size)

### Efficiency Features
- **Weight Sharing**: 1000x speedup vs. training from scratch
- **Early Stopping**: Stop poor architectures after 3-5 epochs
- **Progressive Search**: Start with shallow, expand to deep networks
- **Parameter Reuse**: Warm-start similar architectures

## 📈 Monitoring & MLOps

### Experiment Tracking
```python
import wandb
run = wandb.init(project="nas-rl-search")
# Automatic logging of:
# - Architecture specifications (JSON)
# - Training metrics (loss, accuracy) 
# - System metrics (GPU util, memory)
# - Model artifacts (checkpoints, configs)
```

### Model Registry
```bash
# Register best architecture
python -m src.mlops.model_registry register \
  --architecture arch_001.json \
  --checkpoint best_model.ckpt \
  --metrics val_acc:0.945
```

### Monitoring Dashboard
- **Grafana**: Real-time training metrics, system health
- **Streamlit**: Architecture visualization, search progress
- **W&B**: Experiment comparison, hyperparameter sweeps

## 🚀 Deployment

### API Serving
```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"architecture": "arch_001", "input": [...]}'
```

### Kubernetes Deployment
```bash
# Deploy with Helm
helm install nas-rl infra/k8s/helm/ \
  --set image.tag=latest \
  --set resources.gpu=4

# Scale workers
kubectl scale deployment nas-worker --replicas=8
```

## 🧪 Development

### Run Tests  
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires GPU)
pytest tests/integration/ -v --gpu

# Coverage report
pytest --cov=src tests/
```

### Code Quality
```bash
# Linting
flake8 src/
black src/
isort src/

# Type checking  
mypy src/
```

## 🔧 Configuration

### Search Configuration (`configs/search_cifar10_ray.yaml`)
```yaml
search:
  controller_type: "rnn"  # or "transformer"
  num_architectures: 1000
  max_epochs_per_arch: 20
  early_stopping_patience: 5

ray:
  num_gpus_per_trial: 1
  num_trials_parallel: 4
  cpu_per_trial: 4

mlops:
  wandb_project: "nas-rl-cifar10"
  model_registry: "wandb"  # or "mlflow"

optimization:
  weight_sharing: true
  parameter_reuse: true
  progressive_search: false
```

### Environment Variables (`.env`)
```bash
# MLOps Credentials
WANDB_API_KEY=your_wandb_key
MLFLOW_TRACKING_URI=http://mlflow:5000
DVC_REMOTE_URL=s3://your-bucket/nas-rl

# API Authentication  
API_SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256

# Resource Limits
MAX_CONCURRENT_SEARCHES=4
GPU_MEMORY_LIMIT=16GB
```

## 💰 Cost Estimation

### Cloud Training Costs (AWS)
- **Development**: `g4dn.xlarge` (~$0.50/hour) × 4 hours = **$2**
- **Full Search**: `p3.8xlarge` (~$12/hour) × 24 hours = **$288**  
- **Multi-node**: `p3.16xlarge` × 4 nodes × 12 hours = **$1,152**

### Cost Optimization Tips
1. Use Spot instances (50-70% savings)
2. Enable early stopping (reduce poor architectures)
3. Start with smaller search spaces
4. Use weight sharing (major speedup)
5. Progressive search (shallow → deep)

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors**
```bash
# Reduce batch size or model size
export CUDA_VISIBLE_DEVICES=0
python -m src.train.quick_run --batch_size 64
```

**Ray Cluster Connection**  
```bash
# Check Ray status
ray status
ray list nodes

# Restart Ray
ray stop
ray start --head
```

**DVC Data Issues**
```bash  
# Re-pull data
dvc repro
dvc push --remote storage
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ENAS Paper**: [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268)
- **DARTS**: [Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)  
- **Ray Tune**: [Scalable Hyperparameter Tuning](https://docs.ray.io/en/latest/tune/)
- **PyTorch Lightning**: [The lightweight PyTorch wrapper](https://pytorch-lightning.readthedocs.io/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/nas-rl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/nas-rl/discussions)
- **Documentation**: [Full Docs](https://nas-rl.readthedocs.io/)

---

**Happy Architecture Searching! 🔍🏗️**
