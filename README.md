# Hierarchical MARL control

Hierarchical MARL control is a multi-agent reinforcement learning project for EV charging control.
It jointly learns local charging decisions at each station and global demand-tracking behavior, with the goal of sending EVs out at their target SoC while following a grid-side adjustment signal as closely as possible.

## Overview

- **Environment**: A multi-station EV charging simulation with a variable number of EV slots per station
  - Simulates EV arrivals, dwell time, departure, SoC updates, per-station constraints, and AG tracking
- **Training agent**: MADDPG-style training with local critics, a global TD3-style critic, and a mixer
- **Benchmarks**: Independent DDPG / Shared-Observation DDPG / Shared-Observation SAC / MILP (PuLP + CBC)

## Directory layout

```text
Config.py                       Central location for project-wide hyperparameters
Main.py                         Main training entry point
environment/
  EVEnv.py                      EV charging environment (reset/step)
  ev_info_loader.py             EV profile and arrival-SoC loaders
  normalize.py                  Observation normalization helpers
  observation_config.py         Observation feature flags and dimensions
  readcsv.py                    AG-demand CSV loader
training/
  train.py                      Main training loop
  Agent/
    maddpg.py                   MADDPG agent with local/global critics
    actor.py                    DeepSets-based local actor
    critic.py                   Local/global critics with attention-based components
    mlp.py                      MLP-based critic and mixer components
    noise.py                    Gaussian/OU noise and epsilon-greedy helpers
    replay_buffer.py            GPU-friendly replay buffer
  benchmark_agents/
    independent_ddpg.py         Independent DDPG baseline
    shared_obs_ddpg.py          Shared-observation DDPG baseline
    shared_obs_sac.py           Shared-observation SAC baseline
    milp_agent.py               MILP baseline controller
tools/
  Utils.py                      Plotting and utility helpers
  evaluator.py                  Evaluation helpers for test_history.json and metrics
  execute.py                    Execute/evaluate saved models
  runtensorboard.py             Launch TensorBoard
data/                           Input data (CSV, EV profiles, etc.)
archive/                        Saved models and output artifacts
```

## Setup

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

If you use a GPU, install a CUDA-compatible PyTorch build first.
You can also switch the device manually in `Config.py` by setting `Config.DEVICE` to `cuda` or `cpu`.

## Main workflows

### Training

```bash
python Main.py
```

Most settings live in `Config.py`.
You can switch benchmark agents with flags such as `USE_INDEPENDENT_DDPG`, `USE_SHARED_OBS_DDPG`, and `USE_SHARED_OBS_SAC`. If all of them are `False`, the default trainer is MADDPG.

### Evaluate a saved model

```bash
python -m tools.execute
```

### TensorBoard

```bash
python -m tools.runtensorboard
```

## Notes

- **MILP solver options**: In this project, PuLP CBC options such as `gapRel` / `gapAbs` are passed through `options=["-ratio ..."]` style arguments for compatibility.
- **`MAX_EV_PER_STATION` consistency**: Components such as `LocalEvMLPCritic` and `normalize.py` assume the same `Config.MAX_EV_PER_STATION`. Changing it mid-run or mismatching shapes can cause dimension errors.
- **Permutation invariance**: The actor and local critic are designed to be insensitive to the order of EV slots by using a DeepSets-style encoder. This makes training robust to EV index ordering.
