# Hierarchical Multi-Agent Reinforcement Learning Control for EV-VPP in Balancing Markets
This repository contains the official implementation of the paper:

> **Hierarchical-Multi-Agent-Reinforcement-Learning-Control-for-EV-VPP-in-Balancing-Markets**  
> *Author: Koshin Hayashi, Sihui Xue, Daisuke Kodaira*  
> *Submitted to IEEE Access*

## 📝 Abstract

The project studies coordinated EV fleet control to satisfy both local EV charging goals (e.g., target SoC before departure) and global balancing requests from the power system. It includes a custom EV environment, hierarchical/benchmark agents, and training/evaluation utilities. The framework supports MADDPG-style learning with local and global critics, and benchmark comparisons such as MILP and independent/shared-observation baselines.

## 📂 Repository Structure

```text
.
├── Main.py                     # Entry point
├── Config.py                   # Centralized hyperparameters and experiment flags
├── requirements.txt            # Python dependencies
├── environment/                # EV-VPP simulation environment and preprocessing
│   ├── EVEnv.py
│   ├── ev_info_loader.py
│   ├── observation_config.py
│   ├── normalize.py
│   └── readcsv.py
├── training/                   # Training loop and agent implementations
│   ├── train.py
│   ├── Agent/                  # MADDPG components (actor, critic, replay, noise)
│   └── benchmark_agents/       # MILP, Independent DDPG, Shared-Obs DDPG/SAC
├── tools/                      # Logging, evaluation, execution, visualization helpers
└── data/                       # Input EV profiles and balancing-demand data
```

## 💻 Installation

### 1. Prerequisites

- Python 3.10+
- (Optional) CUDA-enabled GPU for faster training

### 2. Setup

```bash
git clone https://github.com/SmartGridLab/Hierarchical-Multi-Agent-Reinforcement-Learning-Control-for-EV-VPP-in-Balancing-Markets.git
cd Hierarchical-Multi-Agent-Reinforcement-Learning-Control-for-EV-VPP-in-Balancing-Markets

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 🚀 Usage

Run from the repository root:

```bash
python Main.py
```

Training behavior, agent selection, reward weights, and environment settings are configured in:

- `Config.py`

To switch methods, update the boolean flags in `Config.py` (e.g., `USE_MILP`, `USE_INDEPENDENT_DDPG`, `USE_SHARED_OBS_DDPG`, `USE_SHARED_OBS_SAC`, `REGULAR_MADDPG`).

## 📊 Methodology (High Level)

1. Simulate EV arrivals, dwell time, SoC dynamics, and balancing-market demand in `environment/EVEnv.py`.
2. Train agents with replay-based off-policy learning in `training/train.py`.
3. Combine local EV-serving objectives and global balancing objectives through hierarchical value learning and reward design.
4. Compare with benchmark controllers (MILP and alternative RL baselines).

## 📚 Notes

- Results and logs are generated during training and can be visualized via utilities under `tools/`.
- For reproducible experiments, keep seeds and flags fixed in `Config.py`.


@article{Hayashi2026EVVPP,
  title={Hierarchical Multi-Agent Reinforcement Learning Control for EV-VPP in Balancing Markets},
  author={Hayashi, Koshin and Xue, Sihui and Kodaira, Daisuke},
  journal={Submitted to IEEE Access},
  year={2026}
}
