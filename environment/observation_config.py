"""
Observation feature configuration.
Defines the features included in the observation space, their order, and the
derived observation dimensions used by the actor, critics, and normalizer.
"""

try:
    from Config import USE_SWITCHING_CONSTRAINTS
except ImportError:
    USE_SWITCHING_CONSTRAINTS = False


LOCAL_DEMAND_STEPS = 1
LOCAL_USE_STEP = True

GLOBAL_DEMAND_STEPS = 1
GLOBAL_USE_STEP = True
GLOBAL_USE_TOTAL_POWER = True


BASE_EV_FEATURES = (
    "presence",
    "soc",
    "remaining_time",
    "needed_soc",
)
SWITCH_EV_FEATURES = (
    "switch_count",
    "last_direction",
)


def get_ev_feature_names():
    names = list(BASE_EV_FEATURES)
    if bool(USE_SWITCHING_CONSTRAINTS):
        names.extend(SWITCH_EV_FEATURES)
    return tuple(names)


def get_local_tail_feature_names():
    names = [f"demand_{i}" for i in range(int(LOCAL_DEMAND_STEPS))]
    if LOCAL_USE_STEP:
        names.append("step")
    return tuple(names)


def get_global_tail_feature_names():
    names = []
    if GLOBAL_USE_TOTAL_POWER:
        names.append("total_power")
    if GLOBAL_USE_STEP:
        names.append("step")
    names.extend(f"demand_{i}" for i in range(int(GLOBAL_DEMAND_STEPS)))
    return tuple(names)


def ev_block_dim(max_evs):
    return int(max_evs) * EV_FEAT_DIM


def local_obs_dim(max_evs):
    return ev_block_dim(max_evs) + LOCAL_TAIL_DIM


def global_obs_dim(max_evs, n_agent):
    return int(n_agent) * ev_block_dim(max_evs) + GLOBAL_TAIL_DIM


EV_FEATURE_NAMES = get_ev_feature_names()
LOCAL_TAIL_FEATURE_NAMES = get_local_tail_feature_names()
GLOBAL_TAIL_FEATURE_NAMES = get_global_tail_feature_names()

EV_FEAT_DIM = len(EV_FEATURE_NAMES)
LOCAL_TAIL_DIM = len(LOCAL_TAIL_FEATURE_NAMES)
GLOBAL_TAIL_DIM = len(GLOBAL_TAIL_FEATURE_NAMES)
