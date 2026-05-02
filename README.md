# Hierarchical Multi-Agent Reinforcement Learning Control for EV-VPP in Balancing Markets

This repository contains the official implementation of the paper:

> **Hierarchical Multi-Agent Reinforcement Learning Control for EV-VPP in Balancing Markets**  
> *Author: Koshin Hayashi, Sihui Xue, Daisuke Kodaira*  
> *Submitted to IEEE Access*

## 📝 Abstract

This project implements hierarchical multi-agent reinforcement learning for EV charging control in an EV-based virtual power plant (EV-VPP). The framework jointly learns local charging decisions at each charging station and global demand-tracking behavior, with the goal of sending EVs out at their target state of charge (SoC) while following a grid-side adjustment signal as closely as possible.

The repository includes a multi-station EV charging environment, a MADDPG-style learning agent with local critics, a global TD3-style critic, and a mixer, as well as benchmark controllers such as Independent DDPG, Shared-Observation DDPG, Shared-Observation SAC, and MILP.

## 📂 Repository Structure

```text
.
├── Main.py                     # Main training entry point
├── Config.py                   # Centralized hyperparameters and experiment flags
├── requirements.txt            # Python dependencies
├── environment/                # EV-VPP simulation environment and preprocessing
│   ├── EVEnv.py                # EV charging environment (reset/step)
│   ├── ev_info_loader.py       # EV profile and arrival-SoC loaders
│   ├── observation_config.py   # Observation feature flags and dimensions
│   ├── normalize.py            # Observation normalization helpers
│   └── readcsv.py              # AG-demand CSV loader
├── training/                   # Training loop and agent implementations
│   ├── train.py                # Main training loop
│   ├── Agent/                  # MADDPG components
│   │   ├── maddpg.py           # MADDPG agent with local/global critics
│   │   ├── actor.py            # DeepSets-based local actor
│   │   ├── critic.py           # Local/global critics with attention-based components
│   │   ├── mlp.py              # MLP-based critic and mixer components
│   │   ├── noise.py            # Gaussian/OU noise and epsilon-greedy helpers
│   │   └── replay_buffer.py    # GPU-friendly replay buffer
│   └── benchmark_agents/       # Baseline controllers
│       ├── independent_ddpg.py # Independent DDPG baseline
│       ├── shared_obs_ddpg.py  # Shared-observation DDPG baseline
│       ├── shared_obs_sac.py   # Shared-observation SAC baseline
│       └── milp_agent.py       # MILP baseline controller
├── tools/                      # Logging, evaluation, execution, visualization helpers
│   ├── Utils.py                # Plotting and utility helpers
│   ├── evaluator.py            # Evaluation helpers for test_history.json and metrics
│   ├── execute.py              # Execute/evaluate saved models
│   ├── log.py                  # Critic observation debugging utilities
│   ├── run_execute_latest.py   # Run execute.py on the latest archive
│   └── runtensorboard.py       # Launch TensorBoard
├── data/                       # Input EV profiles and AG-demand data
└── archive/                    # Saved models and output artifacts
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

If you use a GPU, install a CUDA-compatible PyTorch build first.  
You can also switch the device manually in `Config.py` by setting `Config.DEVICE` to `cuda` or `cpu`.

## 🚀 Usage

Run training from the repository root:

```bash
python Main.py
```

Most project-wide settings, including environment parameters, reward weights, device settings, and training hyperparameters, are configured in:

- `Config.py`

Benchmark agents can be selected by updating the corresponding flags in `Config.py`, such as:

- `USE_INDEPENDENT_DDPG`
- `USE_SHARED_OBS_DDPG`
- `USE_SHARED_OBS_SAC`

If all benchmark-agent flags are set to `False`, the default trainer is the MADDPG-based hierarchical MARL agent.

To evaluate a saved model:

```bash
python -m tools.execute
```

To run evaluation using the latest archived model:

```bash
python -m tools.run_execute_latest
```

To launch TensorBoard:

```bash
python -m tools.runtensorboard
```

## 📊 Methodology (High Level)

1. Simulate EV arrivals, dwell time, departure, SoC dynamics, per-station charging constraints, and AG-demand tracking in `environment/EVEnv.py`.
2. Train local station-level policies using MADDPG-style off-policy learning in `training/train.py`.
3. Use local critics, a global TD3-style critic, and a mixer to coordinate local EV-serving objectives with global grid-side demand-tracking objectives.
4. Represent variable numbers of EVs at each station using permutation-invariant DeepSets-style actor and critic components.
5. Compare the proposed MARL controller with benchmark controllers, including Independent DDPG, Shared-Observation DDPG, Shared-Observation SAC, and MILP.

## 📚 Notes

- Results, logs, saved models, and output artifacts are generated under `archive/`.
- Evaluation utilities for saved models are available under `tools/`.
- For reproducible experiments, keep random seeds, environment settings, and benchmark flags fixed in `Config.py`.
- **MILP solver options**: In this project, PuLP CBC options such as `gapRel` / `gapAbs` may require compatibility handling depending on the PuLP/CBC interface.
- **`MAX_EV_PER_STATION` consistency**: Components such as `LocalEvMLPCritic` and `normalize.py` assume the same `Config.MAX_EV_PER_STATION`. Changing it mid-run or mismatching shapes can cause dimension errors.
- **Permutation invariance**: The actor and local critic are designed to be insensitive to the order of EV slots by using a DeepSets-style encoder. This makes training robust to EV index ordering.

```bibtex
@article{Hayashi2026EVVPP,
  title={Hierarchical Multi-Agent Reinforcement Learning Control for EV-VPP in Balancing Markets},
  author={Hayashi, Koshin and Xue, Sihui and Kodaira, Daisuke},
  journal={Submitted to IEEE Access},
  year={2026}
}
```
