# NAS-RL: Neural Architecture Search with Reinforcement Learning

[![CI](https://github.com/TRasagna/Neural-Architecture-Search-with-Reinforcement-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/TRasagna/Neural-Architecture-Search-with-Reinforcement-Learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An RL-based Neural Architecture Search system. A recurrent **controller** samples network architectures from a defined search space, child models are built and trained, and the controller is updated with **REINFORCE** using validation performance as the reward. Search runs are orchestrated with **Ray Tune** and exposed through a **FastAPI** service.

## What's implemented

| Component | File(s) | Details |
|---|---|---|
| RL controller | `src/controller/rnn_controller.py`, `src/controller/transformer_controller.py` | LSTM-based (and Transformer-based) controllers that sample operations, skip connections, and hyperparameters step by step |
| Policy training | `src/controller/trainer.py` | REINFORCE with a moving-average baseline |
| Search spaces | `src/search_space/cnn_space.py`, `src/search_space/transformer_space.py` | CNN search space (for CIFAR-10) and a Transformer search space (for SST-2) with validation and random sampling |
| Child models | `src/child/builder.py`, `src/train/lightning_trainer.py` | Builds CNN or Transformer models from a sampled architecture and trains them with PyTorch Lightning |
| Distributed search | `src/ray/run_search.py` | Ray Tune / Ray Train orchestration of architecture evaluation |
| Model registry | `src/mlops/model_registry.py` | Stores discovered architectures and their metrics, and ranks the best ones |
| API | `src/api/main.py` | Token-protected FastAPI service: trigger a search, check status, list best architectures, predict, and Prometheus metrics |
| CI and data | `.github/workflows/ci.yml`, `dvc.yaml`, `scripts/download_data.py` | GitHub Actions test run; DVC tracking for CIFAR-10 and SST-2 data |

## Current status and roadmap
- ✅ Controller, REINFORCE training, CNN search space, and child-model builder
- ✅ Ray Tune integration, FastAPI service, and architecture registry
- ✅ Unit tests for the controller and search space, run in CI
- ⏳ Benchmark results on CIFAR-10 and SST-2
- ⏳ Kubernetes deployment (a starter Helm chart is in `infra/k8s/helm/`)

## Project structure
```
configs/              search configs (dev, CIFAR-10, SST-2) and a sample search request
src/controller/       RNN and Transformer controllers, REINFORCE trainer
src/search_space/     CNN and Transformer search spaces
src/child/            child-model builder
src/train/            Lightning training for child models
src/ray/              Ray Tune search orchestration
src/mlops/            architecture registry
src/api/              FastAPI service
tests/unit/           controller and search-space tests
infra/                Dockerfile and Helm chart
```

## Getting started
```bash
git clone https://github.com/TRasagna/Neural-Architecture-Search-with-Reinforcement-Learning.git
cd Neural-Architecture-Search-with-Reinforcement-Learning
make install
cp .env.example .env          # set API_SECRET_KEY

python scripts/download_data.py   # CIFAR-10
make test

# Start the API, then trigger a search with configs/search_request.json
uvicorn src.api.main:app --port 8000
```

## Tech stack
Python · PyTorch · PyTorch Lightning · Reinforcement Learning (REINFORCE) · Ray Tune · FastAPI · DVC · Docker · GitHub Actions

## License
MIT. See [LICENSE](LICENSE).
