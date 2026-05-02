"""
Config.py — central configuration module for the project.

This file collects hyperparameters and runtime settings for training,
the environment, exploration, and output handling in one place.
Other modules import `Config` and reference these constants directly.
"""
import os
import torch

# --- Core settings ---
ENV_SEED =2545  # Fixed environment random seed for reproducibility.

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")  # Prefer CUDA when available; otherwise use CPU.
PROJECT_ROOT =os .path .dirname (os .path .abspath (__file__ ))  # Absolute path to the project root directory.


# --- Losses and gradients ---
SMOOTHL1_BETA =0.05  # SmoothL1 beta: L2 below this threshold, L1 above it.
GRAD_CLIP_MAX =5.0  # Gradient clipping norm limit for local actor/critic.
GRAD_CLIP_MAX_GLOBAL =5.0  # Gradient clipping norm limit for the global critic.
BIAS_GRAD_CLIP_MAX =0.1  # Separate clipping limit for bias parameters.


# --- Rewards ---
LOCAL_R_SOC_HIT =1.0  # Local reward when an EV departs after reaching its target SoC.
GLOBAL_BALANCE_REWARD =1  # Scale factor for the global demand-tracking reward.

# Extra reward at the departure step when the target SoC is met.
SOC_HIT_BONUS =0.5

LOCAL_DEFICIT_SHAPING_COEF =0.35  # Small bounded shaping term for deficit reduction to support sparse departure rewards without destabilizing Q.
LOCAL_DEFICIT_SHAPING_CLIP =0.05  # Absolute shaping cap per station per step.
LOCAL_DEFICIT_SHAPING_URGENCY_GAIN =1.0  # Extra weighting for deficit reduction when departure is close.
LOCAL_DEFICIT_SHAPING_URGENCY_STEPS =48  # Urgency window in steps (48 x 5 min = about 4 hours).


# --- Training hyperparameters ---
NUM_EPISODES =500
BATCH_SIZE =512  # Minibatch size per training update.
GAMMA =0.985  # Discount factor (local critic / SoC long-horizon objective).
GAMMA_GLOBAL =0.95
TAU =0.005  # Polyak update rate for local networks (actor/local critic).

TAU_GLOBAL =0.005  # Polyak update rate for the global critic.
ACTOR_HIDDEN_SIZE =256  # Hidden layer width of the actor network.

LR_ACTOR =5e-5
LR_CRITIC_LOCAL =1e-4
LOCAL_CRITIC_HIDDEN_SIZE =256  # Hidden layer width of the local critic.
MEMORY_SIZE =int (5e5)  # Replay buffer capacity in transitions.

# Number of steps to collect before starting gradient updates.
WARMUP_STEPS =1500

# Global critic learning settings.
LR_GLOBAL_CRITIC =1e-4
GLOBAL_CRITIC_HIDDEN_SIZE =256  # Hidden layer width of the global critic.


# --- TD3-related settings ---
TD3_SIGMA_GLOBAL =0.20  # Standard deviation of target policy smoothing noise for the global critic.
TD3_CLIP_GLOBAL =0.7  # Clipping range for global target policy smoothing noise.
POLICY_DELAY =2  # Update the actor once every N critic updates.


# --- Q drift mitigation ---
# Soft bound on GlobalMLPCritic mixer bias `b` via
# MIXER_B_MAX * tanh(b_raw / MIXER_B_MAX).
# Bounding the bias keeps the global value scale controlled while leaving enough
# headroom for the Bellman fixed point of the dispatch-tracking reward.
MIXER_B_MAX =50.0
MIXER_B_MAX_ENABLE =True


# --- Gradient blending ---
# actor_grad = (1 - w) * local_Q_grad + w * global_Q_grad
Q_MIX_GLOBAL_WEIGHT =0.5  # Weight assigned to the global-Q gradient.


# --- Environment settings ---
TOL_NARROW_METRICS =150.0  # Power-tracking tolerance width for narrow metrics (kW).
SOC_WIDE =20  # SoC deficit width for the departure reward transition.

EV_CAPACITY =100.0  # EV battery capacity (kWh).
EPISODE_STEPS =288  # Number of steps per episode (288 x 5 min = 24 hours).

# EV charging/discharging power and timing settings.
MAX_EV_POWER_KW =27.5  # Maximum charge/discharge power per EV (kW).
TIME_STEP_MINUTES =5  # Duration of one environment step in minutes.
POWER_TO_ENERGY =TIME_STEP_MINUTES /60.0  # Power-to-energy conversion factor (kW * 5/60 = kWh).

MAX_EV_PER_STATION =10  # Maximum number of EV slots per charging station.
EV_SOC_ARRIVAL_DISTRIBUTION_PATH =os .path .join (PROJECT_ROOT ,"data","input_EVinfo","soc_arrival_distribution.csv")
EV_PROFILE_DATA_PATH =os .path .join (PROJECT_ROOT ,"data","input_EVinfo","WORKPLACE_neededsocanddwelltime.csv")
NUM_EVS =10000  # Number of EVs in the profile pool used for arrival simulation.
NUM_STATIONS =5  # Number of charging stations / agents.

# Number of EVs pre-placed at each station at episode start.
INITIAL_EVS_PER_STATION =3


# --- Exploration noise (action-space) ---
# The MADDPG agent uses independent Gaussian exploration noise per station and EV slot.

# Noise parameters. OU_* names are kept because benchmark agents still use
# OUNoise; MADDPG uses OU_SIGMA and OU_CLIP as Gaussian scale and clipping.
OU_THETA =0.15
OU_SIGMA =0.5
OU_DT =1.0
OU_INIT_X =0.0
OU_CLIP =1.0

# Multiplier on the noise sample (applied to either OU or Gaussian).
# With Gaussian, scale=σ_initial=0.5 means actions get ±0.5 std jitter at the
# start, decaying to the schedule floor. (Action space is [-1,1].)
OU_NOISE_GAIN =1.0

OU_NOISE_START_EPISODE =1
OU_NOISE_END_EPISODE =40
OU_NOISE_SCALE_INITIAL =1.0
OU_NOISE_SCALE_FINAL =0.10


# --- Epsilon-greedy exploration ---
# Each EV slot independently rolls epsilon-greedy exploration. This is the
# long-tail anti-fixation jitter: a tiny but nonzero per-slot ε ensures even
# "forgotten" EVs occasionally get a random push, preventing any slot from
# being permanently neglected by the deterministic policy.
EPSILON_START_EPISODE =1
EPSILON_END_EPISODE =20
EPSILON_INITIAL =1.0
EPSILON_FINAL =0.01
RANDOM_ACTION_RANGE =(-1 ,1 )


# --- Output and evaluation ---
DEMAND_ADJUSTMENT_DIR =os .path .join (PROJECT_ROOT ,"data","input.demand_fromPJM","output_5min")  # Directory containing PJM-based demand-adjustment data.

TRAIN_INTERIM_CSV_INTERVAL_EPISODES =5  # Episode interval for interim test/CSV output.
TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES =5  # Episode interval for interim graph output.

TB_VERBOSE =True


# --- Switching constraints ---
USE_SWITCHING_CONSTRAINTS =False  # Whether to enforce EV charge/discharge direction switching constraints.
MAX_SWITCH_COUNT =10 **9  # Maximum number of direction switches; effectively infinite when constraints are off.
LOCAL_SWITCH_PENALTY =0.0  # Penalty coefficient for switching direction.


# --- Arrival profiles and station power limits ---
# Per-station CSV files for time-dependent EV arrival probabilities.
PER_STATION_ARRIVAL_PROFILE_PATHS =[
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__COMM VITALITY _ 1400 WALNUT1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__COMM VITALITY _ 1104 SPRUCE1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__COMM VITALITY _ 1500PEARL1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__BOULDER _ CARPENTER PARK1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__BOULDER _ N BOULDER REC 1.csv"),
]

# Station-level total-power limit settings.
USE_STATION_TOTAL_POWER_LIMIT =False  # Whether to enforce a total power cap per station.
STATION_MAX_TOTAL_POWER_KW =float (MAX_EV_POWER_KW *MAX_EV_PER_STATION )  # Station total-power limit (kW).
LOCAL_STATION_LIMIT_PENALTY =0.0  # Penalty coefficient for exceeding the station power limit.


# --- Agent selection flags ---
USE_MILP =False  # Whether to use the MILP benchmark controller.

USE_INDEPENDENT_DDPG =False  # Whether to use the independent DDPG baseline.
GLOBAL_REWARD_WEIGHT =0.5  # Weight of the global reward in independent DDPG.

# When True, use only the global critic (roughly equivalent to plain MADDPG).
REGULAR_MADDPG =False

USE_SHARED_OBS_DDPG =False  # Shared-observation DDPG baseline flag.
USE_SHARED_OBS_SAC =False  # Shared-observation SAC baseline flag.
MEASURE_LP_STEP_TIME =False  # Whether to measure LP-solver runtime.
MEASURE_STEP_INDEX =0  # Step index at which timing is measured.


# --- Shared-observation parameters (initialized from matching MADDPG settings) ---
SHARED_OBS_LR_ACTOR =LR_ACTOR  # Actor learning rate for shared-observation DDPG.
SHARED_OBS_LR_CRITIC_LOCAL =LR_CRITIC_LOCAL  # Local critic learning rate for shared-observation DDPG.
SHARED_OBS_GLOBAL_REWARD_WEIGHT =GLOBAL_REWARD_WEIGHT  # Global reward weight for shared-observation DDPG.
SHARED_OBS_EPSILON_END_EPISODE =EPSILON_END_EPISODE  # Final-epsilon episode for shared-observation DDPG.
SHARED_OBS_OU_NOISE_END_EPISODE =OU_NOISE_END_EPISODE  # Final OU-noise schedule episode for shared-observation DDPG.
SHARED_OBS_OU_NOISE_SCALE_FINAL =OU_NOISE_SCALE_FINAL  # Final OU-noise scale for shared-observation DDPG.


# --- SAC-specific parameters ---
SAC_LR_ACTOR =LR_ACTOR  # Actor learning rate for SAC.
SAC_LR_CRITIC =LR_CRITIC_LOCAL  # Critic learning rate for SAC.
SAC_LR_ALPHA =1e-4  # Learning rate for entropy temperature alpha.
SAC_TAU =TAU  # Polyak update rate for SAC.
SAC_ALPHA_INIT =0.2  # Initial entropy-temperature value.
SAC_TARGET_ENTROPY_SCALE =1.0  # Scaling factor for target entropy.
SAC_GLOBAL_REWARD_WEIGHT =GLOBAL_REWARD_WEIGHT  # Global reward weight for SAC.


# --- MILP ---
MILP_W_AG =1.0  # Weight for the AG-tracking objective.
MILP_W_SOC =100.0  # Weight for the SoC target objective.
MILP_AG_DEADBAND =10  # AG-tracking deadband in kW.
MILP_HORIZON =1  # Optimization horizon in steps.
MILP_DEADBAND_PENALTY =1.0  # Penalty coefficient for exceeding the deadband.
MILP_SOC_PENALTY =2.0  # Penalty coefficient for missing target SoC.
