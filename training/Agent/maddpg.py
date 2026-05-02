
"""
MADDPG agent for multi-station EV charging.

This module defines the training-time agent used by `training/train.py`.
Each station owns an actor and a local critic. A separate twin global critic
evaluates system-wide dispatch tracking from all stations at once. The actor
update mixes the station-local objective, which is mainly driven by EV SoC
completion, with the global objective, which is mainly driven by aggregate
power tracking.

Inputs are normalized observations from `environment.normalize`, normalized
actions in [-1, 1], local rewards with shape [batch, stations], and a scalar
global reward per transition. Physical charging limits are applied before the
critics see an action, so Q-values are trained on executable kW commands rather
than on unclipped neural-network outputs.

The update sequence is:
1. sample replay transitions after warmup,
2. update station-local critics with the long-horizon local discount,
3. update the twin global critics with `GAMMA_GLOBAL`,
4. update actors every `POLICY_DELAY` steps by explicitly mixing local and
   global policy gradients,
5. Polyak-average target actors and critics.
"""

import copy
import math
import os

import torch
import torch .nn as nn
import torch .nn .functional as F
import torch .optim as optim

from environment.normalize import denormalize_soc
from Config import (
NUM_EPISODES ,BATCH_SIZE ,GAMMA ,GAMMA_GLOBAL ,TAU ,TAU_GLOBAL ,
LOCAL_CRITIC_HIDDEN_SIZE ,GLOBAL_CRITIC_HIDDEN_SIZE ,
LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC ,
RANDOM_ACTION_RANGE ,SMOOTHL1_BETA ,
EPSILON_START_EPISODE ,EPSILON_END_EPISODE ,EPSILON_INITIAL ,EPSILON_FINAL ,
OU_NOISE_START_EPISODE ,OU_NOISE_END_EPISODE ,
OU_NOISE_SCALE_INITIAL ,OU_NOISE_SCALE_FINAL ,OU_NOISE_GAIN ,
OU_SIGMA ,OU_CLIP ,
TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,
POLICY_DELAY ,
MEMORY_SIZE ,WARMUP_STEPS ,
Q_MIX_GLOBAL_WEIGHT ,
EV_CAPACITY ,POWER_TO_ENERGY ,
REGULAR_MADDPG ,
MAX_EV_POWER_KW ,MAX_EV_PER_STATION ,
BIAS_GRAD_CLIP_MAX ,GRAD_CLIP_MAX ,GRAD_CLIP_MAX_GLOBAL ,
)

from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_DEMAND_STEPS ,LOCAL_TAIL_DIM ,
GLOBAL_DEMAND_STEPS ,GLOBAL_USE_STEP ,GLOBAL_USE_TOTAL_POWER ,
)

try :
    from .actor import Actor
    from .critic import LocalEvMLPCritic ,GlobalMLPCritic
    from .replay_buffer import ReplayBuffer
    from .noise import (
    GaussianNoise ,
    linear_epsilon_decay ,
    sample_epsilon_random_action ,
    sample_per_slot_random_mask ,
    )
except ImportError :
    from actor import Actor
    from critic import LocalEvMLPCritic ,GlobalMLPCritic
    from replay_buffer import ReplayBuffer
    from noise import (
    GaussianNoise ,
    linear_epsilon_decay ,
    sample_epsilon_random_action ,
    sample_per_slot_random_mask ,
    )


device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
if torch .cuda .is_available ():
    torch .set_default_device ("cuda")


def _clip_bias_gradients (model ,max_norm =1.0 ):
    """Clip only bias gradients before the full-parameter gradient clip."""
    for name ,param in model .named_parameters ():
        if param .grad is not None and 'bias'in name :
            torch .nn .utils .clip_grad_norm_ (param ,max_norm )


class MADDPG :
    def __init__ (self ,s_dim ,max_evs_per_station ,n_agent ,
    gamma =GAMMA ,
    tau =TAU ,
    batch =BATCH_SIZE ,
    lr_a =LR_ACTOR ,lr_c =LR_CRITIC_LOCAL ,lr_global_c =LR_GLOBAL_CRITIC ,
    num_episodes =NUM_EPISODES ,
    tau_global :float =TAU_GLOBAL ,
    td3_sigma :float =TD3_SIGMA_GLOBAL ,
    td3_clip :float =TD3_CLIP_GLOBAL ,
    smoothl1_beta =1.0 ,
    **kwargs ):
        """Create actors, critics, target networks, replay memory, and exploration state."""
        del num_episodes
        del kwargs
        if max_evs_per_station !=MAX_EV_PER_STATION :
            raise AssertionError (
            f"max_evs_per_station={max_evs_per_station} != Config.MAX_EV_PER_STATION={MAX_EV_PER_STATION}. "
            "GlobalMLPCritic and normalize.py use Config.MAX_EV_PER_STATION directly; "
            "passing a different value causes silent shape mismatches."
            )
        self .s_dim ,self .a_dim ,self .n =s_dim ,max_evs_per_station ,n_agent
        self .max_ev_per_station =max_evs_per_station
        self .gamma ,self .tau ,self .batch =gamma ,tau ,batch
        self .lr_global_c =lr_global_c
        self .current_episode =0

        self .actor_norms =[0 ]*n_agent
        self .critic_norms =[0 ]*n_agent
        self .actor_losses =[]
        self .critic_losses =[]
        self .last_global_critic_loss =0.0
        self .last_actor_loss =0.0
        self .last_critic_loss =0.0
        self .last_actor_grad_norm =0.0
        self .last_local_critic_grad_norm =0.0
        self .last_global_critic_grad_norm =0.0
        self .last_actor_clip_count =0
        self .last_local_critic_clip_count =0
        self .last_global_critic_clip_count =0
        self .local_critic_clip_counts =[0 ]*n_agent
        self .actor_clip_counts =[0 ]*n_agent
        self .critic_norms_before_clip =[0 ]*n_agent
        self .actor_norms_before_clip =[0 ]*n_agent
        self .actor_source_local_norms_before_clip =[0.0 ]*n_agent
        self .actor_source_global_norms_before_clip =[0.0 ]*n_agent
        self .actor_source_global_ratio =[0.0 ]*n_agent
        self .actor_source_cos =[0.0 ]*n_agent
        self .actor_source_cos_valid =[0 ]*n_agent
        self .last_global_critic_grad_norm_before_clip =0.0
        self .last_actor_source_local_grad_norm_before_clip =0.0
        self .last_actor_source_global_grad_norm_before_clip =0.0
        self .last_actor_source_global_ratio =0.0
        self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =0.0
        self .last_local_q_values_per_agent =[0.0 ]*n_agent
        self .last_global_q_value =0.0

        self .env =None
        self .use_tensorboard =False
        self .writer =None

        self .random_action_range =RANDOM_ACTION_RANGE

        self .epsilon_start_episode =EPSILON_START_EPISODE
        self .epsilon_end_episode =EPSILON_END_EPISODE
        self .epsilon_initial =EPSILON_INITIAL
        self .epsilon_final =EPSILON_FINAL
        self .epsilon =self .epsilon_initial

        self .ou_noise_start_episode =OU_NOISE_START_EPISODE
        self .ou_noise_end_episode =OU_NOISE_END_EPISODE
        self .ou_noise_scale_initial =OU_NOISE_SCALE_INITIAL
        self .ou_noise_scale_final =OU_NOISE_SCALE_FINAL
        self .ou_noise_scale =self .ou_noise_scale_initial

        self .ou_noise =GaussianNoise (
        n_agent ,max_evs_per_station ,
        sigma =float (OU_SIGMA ),
        clip =float (OU_CLIP )if OU_CLIP is not None and OU_CLIP >0 else None ,
        )

        self .test_mode =False

        self .buf =ReplayBuffer (cap =int (MEMORY_SIZE ))
        self .buf .maddpg_ref =self
        self .max_evs =max_evs_per_station

        self .active_evs =[0 ]*n_agent

        self .ev_state_dim =EV_FEAT_DIM
        self .local_tail_dim =LOCAL_TAIL_DIM
        self .station_state_dim =self .ev_state_dim *self .max_evs +self .local_tail_dim

        self .actors =[
        Actor (s_dim ,max_evs_per_station ,station_state_dim =self .station_state_dim ).to (device )
        for _ in range (n_agent )
        ]
        self .t_actors =[copy .deepcopy (ac )for ac in self .actors ]

        self .critics =[
        LocalEvMLPCritic (
        ev_feat_dim =EV_FEAT_DIM ,
        a_dim =max_evs_per_station ,
        max_evs =max_evs_per_station ,
        hid =LOCAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self .station_state_dim ,
        ).to (device )
        for _ in range (n_agent )
        ]
        self .t_critics =[copy .deepcopy (cr )for cr in self .critics ]


        # Global critic targets use their own Polyak rate and TD3 smoothing
        # scale because the dispatch-tracking value has a shorter horizon than
        # the station-local SoC value.
        self .tau_global =float (tau_global )
        self .td3_sigma =float (td3_sigma )
        self .td3_clip =float (td3_clip )

        additional_features =(
        (1 if GLOBAL_USE_TOTAL_POWER else 0 )
        +(1 if GLOBAL_USE_STEP else 0 )
        +int (GLOBAL_DEMAND_STEPS )
        )
        self ._global_station_dim =EV_FEAT_DIM *self .max_evs
        global_obs_dim =(self .n *self ._global_station_dim )+additional_features

        self .global_critic1 =GlobalMLPCritic (
        global_obs_dim ,max_evs_per_station ,n_agent ,
        hid =GLOBAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self ._global_station_dim ,
        init_gain =0.3 ,
        ).to (device )
        self .global_critic2 =GlobalMLPCritic (
        global_obs_dim ,max_evs_per_station ,n_agent ,
        hid =GLOBAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self ._global_station_dim ,
        init_gain =0.3 ,
        ).to (device )
        self .t_global_critic1 =copy .deepcopy (self .global_critic1 )
        self .t_global_critic2 =copy .deepcopy (self .global_critic2 )

        self .opt_a =[optim .Adam (self .actors [i ].parameters (),lr =lr_a )for i in range (n_agent )]
        self .opt_c =[optim .Adam (self .critics [i ].parameters (),lr =lr_c )for i in range (n_agent )]
        self .opt_global_c1 =optim .Adam (self .global_critic1 .parameters (),lr =self .lr_global_c )
        self .opt_global_c2 =optim .Adam (self .global_critic2 .parameters (),lr =self .lr_global_c )

        self .loss_fn =nn .SmoothL1Loss (beta =smoothl1_beta )

        self .clip_bias_gradients =_clip_bias_gradients

        self ._ep_q_raw_global =[]
        self ._ep_q_raw_local =[[]for _ in range (n_agent )]

        self .policy_delay =max (1 ,int (POLICY_DELAY ))

        self .update_step =0


    def update_active_evs (self ,env ):
        """Synchronize the agent-side active-slot counts with the environment mask."""
        num_stations =min (self .n ,env .num_stations )
        counts =env .ev_mask [:num_stations ].sum (dim =1 ).cpu ().tolist ()
        if self .n >num_stations :
            counts .extend ([0 ]*(self .n -num_stations ))
        self .active_evs =counts
        self .env =env

    def _convert_to_global_critic_obs (self ,s ,actual_station_powers ):
        """
        Build the flat global-critic state from per-station observations.

        `s` has shape [batch, stations, station_state_dim]. The global critic
        receives all EV feature blocks concatenated across stations, followed
        by optional aggregate power, optional normalized time step, and a
        global demand look-ahead vector derived from the local observation tail.
        `actual_station_powers` is the physically clipped station power in kW
        for the action being evaluated.
        """
        B =s .size (0 )

        ev_features_per_station =self .max_evs *EV_FEAT_DIM
        ev_features_all =s [:,:,:ev_features_per_station ]
        station_features_flat =ev_features_all .reshape (B ,-1 )

        if GLOBAL_USE_TOTAL_POWER :
            MAX_POSSIBLE_POWER =MAX_EV_POWER_KW *self .n *self .max_evs
            total_ev_power_raw =actual_station_powers .sum (dim =1 ,keepdim =True )
            total_ev_power =torch .clamp (total_ev_power_raw /MAX_POSSIBLE_POWER ,-1.0 ,1.0 )
        else :
            total_ev_power =s .new_zeros (B ,1 )

        if GLOBAL_USE_STEP :
            current_step =s [:,0 ,-1 :]
        else :
            current_step =s .new_zeros (B ,1 )

        L_global =int (GLOBAL_DEMAND_STEPS )
        if L_global >0 :
            tail =s [:,0 ,ev_features_per_station :]
            tail_idx =0

            ag_local =None
            if LOCAL_DEMAND_STEPS >0 and tail .size (1 )>=tail_idx +LOCAL_DEMAND_STEPS :
                ag_local =tail [:,tail_idx :tail_idx +LOCAL_DEMAND_STEPS ]

            if ag_local is not None :
                if LOCAL_DEMAND_STEPS >=L_global :
                    ag_lookahead =ag_local [:,:L_global ]
                else :
                    pad =s .new_zeros (B ,L_global -LOCAL_DEMAND_STEPS )
                    ag_lookahead =torch .cat ([ag_local ,pad ],dim =1 )
            else :
                ag_lookahead =s .new_zeros (B ,L_global )
        else :
            ag_lookahead =s .new_zeros (B ,0 )

        parts =[station_features_flat ]
        if GLOBAL_USE_TOTAL_POWER :
            parts .append (total_ev_power )
        if GLOBAL_USE_STEP :
            parts .append (current_step )
        parts .append (ag_lookahead )
        global_obs =torch .cat (parts ,dim =1 )
        return global_obs


    def _apply_soc_constraint (self ,actions_kw ,current_socs ,ev_padding_mask =None ,use_ste =False ):
        """
        Clip proposed kW actions so one step cannot overcharge or overdischarge.

        The environment stores EV energy in kWh. `actions_kw` is converted to
        the one-step energy increment, clipped to the feasible SoC interval,
        and converted back to kW. When `use_ste` is true, the forward value is
        clipped while the backward pass uses a straight-through estimator so
        actors still receive gradients at the physical boundary.
        """
        proposed_delta_kwh =actions_kw *POWER_TO_ENERGY
        max_charge =EV_CAPACITY -current_socs
        max_discharge =current_socs
        clamped_kwh =torch .clamp (proposed_delta_kwh ,-max_discharge ,max_charge )
        if use_ste :
            clamped_kwh =proposed_delta_kwh +(clamped_kwh -proposed_delta_kwh ).detach ()
        clamped_actions_kw =clamped_kwh /POWER_TO_ENERGY
        if ev_padding_mask is not None :
            clamped_actions_kw =clamped_actions_kw .masked_fill (ev_padding_mask ,0.0 )
        station_powers_kw =clamped_actions_kw .sum (dim =2 )
        return clamped_actions_kw ,station_powers_kw


    def act (self ,state ,env =None ,noise =True ):
        """Return normalized station-by-slot actions for the current observation."""
        if self .test_mode :
            noise =False

        if env is not None :
            self .update_active_evs (env )

        active_agents =[i for i ,num in enumerate (self .active_evs )if num >0 ]
        tensor_actions =torch .zeros ((self .n ,self .max_evs ),dtype =torch .float32 ,device =device )

        # Sample one noise tensor for the whole step. active_evs gates which
        # slots actually receive it.
        is_training =noise and not self .test_mode
        step_noise =None
        if is_training and self .ou_noise_scale >0.0 :
            step_noise =self .ou_noise .sample (self .active_evs )

        with torch .no_grad ():
            for agent_idx in active_agents :
                num_active =self .active_evs [agent_idx ]
                agent_state =state [agent_idx :agent_idx +1 ]
                a =self .actors [agent_idx ](agent_state )

                if is_training :
                    # 1) Continuous Gaussian perturbation on the policy output.
                    if step_noise is not None :
                        a [0 ,:num_active ]=(
                        a [0 ,:num_active ]
                        +(OU_NOISE_GAIN *self .ou_noise_scale )
                        *step_noise [agent_idx ,:num_active ]
                        )

                    # 2) Epsilon-greedy slot-level random override.
                    # Replacing individual EV slots, rather than whole
                    # stations, keeps exploration from locking a station
                    # into a persistent charge-only or discharge-only role.
                    eps_now =float (self .epsilon )
                    if eps_now >0.0 :
                        mask =sample_per_slot_random_mask (
                        num_active ,eps_now ,like_tensor =a [0 ,:num_active ]
                        )
                        if mask .any ():
                            rand_a =sample_epsilon_random_action (
                            int (mask .sum ().item ()),
                            action_range =self .random_action_range ,
                            like_tensor =a [0 ,:num_active ],
                            )
                            a [0 ,:num_active ][mask ]=rand_a

                a =torch .clamp (a ,-1.0 ,1.0 )
                tensor_actions [agent_idx ,:num_active ]=a .squeeze (0 )[:num_active ]

        return tensor_actions


    def _zero_update_logs (self ):
        """Reset scalar diagnostics when no gradient update is performed."""
        self .last_actor_loss =0.0
        self .last_critic_loss =0.0
        self .last_actor_grad_norm =0.0
        self .last_local_critic_grad_norm =0.0
        self .last_global_critic_grad_norm =0.0
        self .last_actor_source_local_grad_norm_before_clip =0.0
        self .last_actor_source_global_grad_norm_before_clip =0.0
        self .last_actor_source_global_ratio =0.0
        self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =0.0


    def _build_update_ctx (self ,s ,s2 ,a ,r_local ,d ,
    actual_station_powers ,actual_ev_power_kw ):
        """Precompute masks, action tensors, and objective switches for one update."""
        batch_size =s .size (0 )
        max_evs =self .a_dim

        a_actual =(
        torch .clamp (actual_ev_power_kw /MAX_EV_POWER_KW ,-1.0 ,1.0 )
        if actual_ev_power_kw is not None else a
        )

        ev_block =s [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,self .max_evs ,EV_FEAT_DIM )
        presence_mask =(ev_block [...,0 ]<=0.5 )
        ev_padding_mask =presence_mask
        key_padding_mask =presence_mask .all (dim =2 )

        ev_block_s2 =s2 [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,self .max_evs ,EV_FEAT_DIM )
        presence_mask_s2 =(ev_block_s2 [...,0 ]<=0.5 )
        ev_padding_mask_s2 =presence_mask_s2

        skip_local =(Q_MIX_GLOBAL_WEIGHT ==1.0 )
        skip_global =(Q_MIX_GLOBAL_WEIGHT ==0.0 )
        if REGULAR_MADDPG :
            skip_local =True
            skip_global =False

        max_station_power =MAX_EV_POWER_KW *MAX_EV_PER_STATION
        w_eff =1.0 if REGULAR_MADDPG else Q_MIX_GLOBAL_WEIGHT

        return {
        's':s ,'s2':s2 ,
        'r_local':r_local ,'d':d ,
        'actual_station_powers':actual_station_powers ,
        'a_actual':a_actual ,
        'batch_size':batch_size ,'max_evs':max_evs ,
        'ev_block_s2':ev_block_s2 ,
        'ev_padding_mask':ev_padding_mask ,
        'ev_padding_mask_s2':ev_padding_mask_s2 ,
        'key_padding_mask':key_padding_mask ,
        'skip_local':skip_local ,'skip_global':skip_global ,
        'max_station_power':max_station_power ,
        'w_eff':w_eff ,
        }


    def _update_local_critics (self ,ctx ):
        """
        Update one station-local critic per station.

        Local critics use the long-horizon discount `self.gamma` because their
        reward is tied to SoC progress over the dwell time of each EV. Target
        actions are generated by target actors, clipped to SoC feasibility, and
        evaluated by the corresponding target local critic.
        """
        if ctx ['skip_local']:
            self .critic_losses =[]
            return 0

        s ,s2 =ctx ['s'],ctx ['s2']
        a_actual =ctx ['a_actual']
        r_local ,d =ctx ['r_local'],ctx ['d']
        actual_station_powers =ctx ['actual_station_powers']
        ev_padding_mask_s2 =ctx ['ev_padding_mask_s2']
        max_station_power =ctx ['max_station_power']
        ev_block_s2 =ctx ['ev_block_s2']
        local_critic_clip_count =0

        with torch .no_grad ():
            next_actions_all =torch .stack (
            [self .t_actors [i ](s2 [:,i ,:])for i in range (self .n )],dim =1
            )
            next_actions_all =torch .clamp (next_actions_all ,-1.0 ,1.0 )

            next_actions_kw =next_actions_all *MAX_EV_POWER_KW
            current_socs_s2 =denormalize_soc (ev_block_s2 [...,1 ])
            next_actions_clamped ,next_agent_powers =self ._apply_soc_constraint (
            next_actions_kw ,current_socs_s2 ,ev_padding_mask_s2
            )
            next_powers_norm =torch .clamp (next_agent_powers /max_station_power ,-1.0 ,1.0 )
            next_actions_normalized =torch .clamp (
            next_actions_clamped /MAX_EV_POWER_KW ,-1.0 ,1.0
            )
            target_qs =[]
            for i in range (self .n ):
                tq1 =self .t_critics [i ](
                s2 [:,i ,:],next_actions_normalized [:,i ,:],
                actual_station_powers =next_powers_norm [:,i ],
                )
                tq1 =torch .nan_to_num_ (tq1 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
                target_qs .append (tq1 )

        y_targets =[]
        for i in range (self .n ):
            y =r_local [:,i :i +1 ]+self .gamma *target_qs [i ]*(1 -d [:,i :i +1 ])
            y_targets .append (torch .nan_to_num_ (y ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided.")
        powers_norm =torch .clamp (actual_station_powers /max_station_power ,-1.0 ,1.0 )

        q_vals =[]
        for i in range (self .n ):
            q_val =self .critics [i ](
            s [:,i ,:],a_actual [:,i ,:],actual_station_powers =powers_norm [:,i ]
            )
            q_vals .append (torch .nan_to_num_ (q_val ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        self .critic_losses =[]
        for i in range (self .n ):
            loss_c =F .smooth_l1_loss (q_vals [i ],y_targets [i ],beta =SMOOTHL1_BETA ,reduction ='mean')
            loss_c =torch .nan_to_num_ (loss_c ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            self .critic_losses .append (loss_c .item ()if torch .isfinite (loss_c )else 0.0 )

            if torch .isfinite (loss_c ):
                self .opt_c [i ].zero_grad ()
                loss_c .backward (retain_graph =True )
                self .clip_bias_gradients (self .critics [i ],max_norm =BIAS_GRAD_CLIP_MAX )
                gn =torch .nn .utils .clip_grad_norm_ (
                self .critics [i ].parameters (),max_norm =GRAD_CLIP_MAX
                )
                gnf =float (gn )
                if gnf >5.0 :
                    local_critic_clip_count +=1
                self .critic_norms_before_clip [i ]=gnf
                self .critic_norms [i ]=min (gnf ,GRAD_CLIP_MAX )
                self .opt_c [i ].step ()


        return local_critic_clip_count


    def _update_actors (self ,ctx ):
        """
        Update station actors with an explicit local/global gradient mixture.

        For station i, the local gradient is taken from its local critic and
        the global gradient is taken from the twin global critic while allowing
        only station i's action slice to carry gradients. The applied gradient is
        `g_mix = (1 - w_eff) * g_local + w_eff * g_global`; the separate norms
        and cosine are recorded to diagnose objective conflict.
        """
        s =ctx ['s']
        ev_padding_mask =ctx ['ev_padding_mask']
        key_padding_mask =ctx ['key_padding_mask']
        batch_size ,max_evs =ctx ['batch_size'],ctx ['max_evs']
        skip_local ,skip_global =ctx ['skip_local'],ctx ['skip_global']
        max_station_power =ctx ['max_station_power']
        w_eff =ctx ['w_eff']
        actor_clip_count =0

        current_actions =[self .actors [i ](s [:,i ,:])for i in range (self .n )]
        cur_a_all_new =torch .stack (current_actions ,dim =1 )
        actions_all =cur_a_all_new *MAX_EV_POWER_KW

        ev_feats =s [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,max_evs ,EV_FEAT_DIM )
        current_socs_all =denormalize_soc (ev_feats [...,1 ])

        if ev_padding_mask is not None :
            actions_all =actions_all .masked_fill (ev_padding_mask ,0.0 )

        clamped_actions_all ,recomputed_actual_station_powers =self ._apply_soc_constraint (
        actions_all ,current_socs_all ,ev_padding_mask ,use_ste =True
        )
        clamped_actions_all_critic =clamped_actions_all

        s_global_actor =None
        recomputed_actual_station_powers_normalized =None
        if not skip_global :
            s_global_actor =self ._convert_to_global_critic_obs (
            s ,recomputed_actual_station_powers )
            recomputed_actual_station_powers_normalized =torch .clamp (
            recomputed_actual_station_powers /max_station_power ,-1.0 ,1.0
            )

        self .actor_losses =[0.0 ]*self .n
        self .actor_clip_counts =[0 ]*self .n
        self .actor_source_local_norms_before_clip =[0.0 ]*self .n
        self .actor_source_global_norms_before_clip =[0.0 ]*self .n
        self .actor_source_global_ratio =[0.0 ]*self .n
        self .actor_source_cos =[0.0 ]*self .n
        self .actor_source_cos_valid =[0 ]*self .n

        for i in range (self .n ):
            params =list (self .actors [i ].parameters ())

            agent_actual_power =recomputed_actual_station_powers [:,i ]
            agent_actual_power_normalized =torch .clamp (
            agent_actual_power /max_station_power ,-1.0 ,1.0
            )

            if skip_local :
                q_local =torch .zeros ((batch_size ,1 ),device =device )
            else :
                s_flat_i =s [:,i ,:]
                agent_a =torch .clamp (
                clamped_actions_all_critic [:,i ,:]/MAX_EV_POWER_KW ,-1.0 ,1.0 )
                q_local =self .critics [i ](
                s_flat_i ,agent_a ,actual_station_powers =agent_actual_power_normalized )

            q_local =torch .nan_to_num_ (q_local ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

            a_all_kw =clamped_actions_all .detach ().clone ()
            a_all_kw [:,i ,:]=clamped_actions_all [:,i ,:]
            a_all_kw_masked =a_all_kw .masked_fill (ev_padding_mask ,0.0 )
            a_all_kw_normalized =torch .clamp (a_all_kw_masked /MAX_EV_POWER_KW ,-1.0 ,1.0 )

            if skip_global :
                q_global =torch .zeros ((batch_size ,1 ),device =device )
            else :
                q1 ,_ =self .global_critic1 (
                s_global_actor ,a_all_kw_normalized ,key_padding_mask ,
                actual_station_powers =recomputed_actual_station_powers_normalized ,
                )
                q2 ,_ =self .global_critic2 (
                s_global_actor ,a_all_kw_normalized ,key_padding_mask ,
                actual_station_powers =recomputed_actual_station_powers_normalized ,
                )
                # Use the conservative twin-critic estimate for actor updates too.
                # This avoids exploiting one over-optimistic critic after global tracking has saturated.
                q_global =torch .minimum (q1 ,q2 )

            q_global =torch .nan_to_num_ (q_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

            q_l_mean =q_local .mean ()
            q_g_mean =q_global .mean ()

            self ._ep_q_raw_local [i ].append (q_l_mean .detach ())
            if i ==0 :
                self ._ep_q_raw_global .append (q_g_mean .detach ())

            self .opt_a [i ].zero_grad ()
            local_grads =[None ]*len (params )
            global_grads =[None ]*len (params )

            if not skip_local :
                local_grads =list (torch .autograd .grad (
                -q_l_mean ,params ,retain_graph =True ,allow_unused =True
                ))
            if not skip_global :
                global_grads =list (torch .autograd .grad (
                -q_g_mean ,params ,retain_graph =True ,allow_unused =True
                ))

            sq_l =torch .zeros ((),device =device )
            sq_g =torch .zeros ((),device =device )
            dot_lg =torch .zeros ((),device =device )
            has_l ,has_g =False ,False
            for g_l ,g_g in zip (local_grads ,global_grads ):
                if g_l is not None :
                    sq_l =sq_l +(g_l .detach ()*g_l .detach ()).sum ()
                    has_l =True
                if g_g is not None :
                    sq_g =sq_g +(g_g .detach ()*g_g .detach ()).sum ()
                    has_g =True
                if g_l is not None and g_g is not None :
                    dot_lg =dot_lg +(g_l .detach ()*g_g .detach ()).sum ()

            norm_l =float (torch .sqrt (torch .clamp (sq_l ,min =0.0 )).item ())if has_l else 0.0
            norm_g =float (torch .sqrt (torch .clamp (sq_g ,min =0.0 )).item ())if has_g else 0.0
            ratio_g =norm_g /max (norm_l +norm_g ,1e-12 )

            cos_lg =0.0
            cos_valid =0
            if has_l and has_g and norm_l >1e-12 and norm_g >1e-12 :
                cos_tensor =dot_lg /(
                torch .sqrt (torch .clamp (sq_l ,min =1e-24 ))*
                torch .sqrt (torch .clamp (sq_g ,min =1e-24 ))
                )
                cos_lg =float (torch .clamp (cos_tensor ,-1.0 ,1.0 ).item ())
                cos_valid =1

            self .actor_source_local_norms_before_clip [i ]=norm_l
            self .actor_source_global_norms_before_clip [i ]=norm_g
            self .actor_source_global_ratio [i ]=ratio_g
            self .actor_source_cos [i ]=cos_lg
            self .actor_source_cos_valid [i ]=cos_valid

            for idx ,p in enumerate (params ):
                g_l =local_grads [idx ]
                g_g =global_grads [idx ]
                if g_l is not None and g_g is not None :
                    g_mix =(1.0 -w_eff )*g_l +w_eff *g_g
                elif g_l is not None :
                    g_mix =g_l
                elif g_g is not None :
                    g_mix =g_g
                else :
                    g_mix =None
                p .grad =g_mix .clone ()if g_mix is not None else None

            has_nan_inf =any (
            p .grad is not None and not torch .isfinite (p .grad ).all ()
            for p in params
            )
            if has_nan_inf :
                self .actor_norms [i ]=0.0
                continue

            self .clip_bias_gradients (self .actors [i ],max_norm =BIAS_GRAD_CLIP_MAX )
            grad_norm_before =torch .nn .utils .clip_grad_norm_ (
            self .actors [i ].parameters (),max_norm =GRAD_CLIP_MAX
            ).item ()
            grad_norm_after =min (grad_norm_before ,GRAD_CLIP_MAX )

            if grad_norm_before >5.0 :
                actor_clip_count +=1
                if i <len (self .actor_clip_counts ):
                    self .actor_clip_counts [i ]=1

            if math .isfinite (grad_norm_after ):
                self .actor_norms [i ]=grad_norm_after
                self .actor_norms_before_clip [i ]=grad_norm_before
                self .opt_a [i ].step ()
            else :
                self .actor_norms [i ]=0.0

            self .actor_losses [i ]=-(
            (1.0 -w_eff )*q_l_mean .detach ()+w_eff *q_g_mean .detach ()
            ).item ()

        return actor_clip_count


    def _polyak_update_targets (self ):
        """
        Soft-update target actors, local critics, and global critics.

        Local networks use `self.tau`; the twin global critics use
        `self.tau_global` so the shorter-horizon global value can track its
        online critic at an independently chosen rate.
        """
        src_local ,tgt_local =[],[]
        for i in range (self .n ):
            for p ,tp in zip (self .actors [i ].parameters (),self .t_actors [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
            for p ,tp in zip (self .critics [i ].parameters (),self .t_critics [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
        torch ._foreach_lerp_ (tgt_local ,src_local ,self .tau )

        src_global ,tgt_global =[],[]
        for p ,tp in zip (self .global_critic1 .parameters (),self .t_global_critic1 .parameters ()):
            src_global .append (p .data );tgt_global .append (tp .data )
        for p ,tp in zip (self .global_critic2 .parameters (),self .t_global_critic2 .parameters ()):
            src_global .append (p .data );tgt_global .append (tp .data )
        torch ._foreach_lerp_ (tgt_global ,src_global ,self .tau_global )


    def _aggregate_update_logs (self ,local_critic_clip_count ,actor_clip_count ):
        """Collect the latest losses, Q-values, gradient norms, and clip counts."""
        if self .critic_losses :
            self .last_critic_loss =sum (self .critic_losses )/len (self .critic_losses )
        if self .actor_losses :
            self .last_actor_loss =sum (self .actor_losses )/len (self .actor_losses )

        if self .critic_norms :
            avg_cn =sum (self .critic_norms )/len (self .critic_norms )
            self .last_local_critic_grad_norm =avg_cn if math .isfinite (avg_cn )else 0.0
        else :
            self .last_local_critic_grad_norm =0.0
        self .last_local_critic_clip_count =local_critic_clip_count

        if self .actor_norms :
            avg_an =sum (self .actor_norms )/len (self .actor_norms )
            self .last_actor_grad_norm =avg_an if math .isfinite (avg_an )else 0.0
        else :
            self .last_actor_grad_norm =0.0
        self .last_actor_clip_count =actor_clip_count

        src_local =self .actor_source_local_norms_before_clip
        self .last_actor_source_local_grad_norm_before_clip =(
        float (sum (src_local )/len (src_local ))if src_local else 0.0
        )
        src_global_norms =self .actor_source_global_norms_before_clip
        self .last_actor_source_global_grad_norm_before_clip =(
        float (sum (src_global_norms )/len (src_global_norms ))if src_global_norms else 0.0
        )
        self .last_actor_source_global_ratio =(
        float (sum (self .actor_source_global_ratio )/len (self .actor_source_global_ratio ))
        if self .actor_source_global_ratio else 0.0
        )

        cos_valid_count =int (sum (self .actor_source_cos_valid ))if self .actor_source_cos_valid else 0
        if cos_valid_count >0 :
            cos_sum =sum (
            float (v )for v ,flag in zip (self .actor_source_cos ,self .actor_source_cos_valid )
            if flag
            )
            self .last_actor_source_cos =cos_sum /float (cos_valid_count )
        else :
            self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =(
        float (cos_valid_count )/float (len (self .actor_source_cos_valid ))
        if self .actor_source_cos_valid else 0.0
        )

        self .last_local_q_values_per_agent =[
        float (self ._ep_q_raw_local [i ][-1 ].item ())if self ._ep_q_raw_local [i ]else 0.0
        for i in range (self .n )
        ]
        self .last_global_q_value =(
        float (self ._ep_q_raw_global [-1 ].item ())if self ._ep_q_raw_global else 0.0
        )


    def _update_global_critic (self ,ctx ):
        """
        Update the twin global critics with a one-step TD3-style target.

        The global critic observes all station EV states and all station
        actions simultaneously. Target policy smoothing is applied to target
        actor outputs, then actions are clipped to physical SoC limits before
        the target Q is computed. The target uses `GAMMA_GLOBAL`, which is
        shorter than the local discount because dispatch tracking is an
        immediate aggregate-power objective.
        """
        if ctx ['skip_global']:
            return

        s =ctx ['s']
        a_actual =ctx ['a_actual']
        r_global_n =ctx ['r_global_n']
        s2_n =ctx ['s2_n']
        d_n =ctx ['d_n']
        actual_station_powers =ctx ['actual_station_powers']
        ev_padding_mask =ctx ['ev_padding_mask']
        key_padding_mask =ctx ['key_padding_mask']
        max_station_power =ctx ['max_station_power']

        # The replay sampler is called with n_step=1, so `s2_n` is the ordinary
        # next state.
        s2_for_target =s2_n

        # Build masks from the target next-state EV presence flags so target
        # actions are ignored for empty EV slots and empty stations.
        ev_block_for_target =s2_for_target [:,:,:self .max_evs *EV_FEAT_DIM ].reshape (
        s .size (0 ),self .n ,self .max_evs ,EV_FEAT_DIM )
        presence_mask_for_target =(ev_block_for_target [...,0 ]<=0.5 )
        ev_padding_mask_for_target =presence_mask_for_target
        key_padding_mask_for_target =presence_mask_for_target .all (dim =2 )

        with torch .no_grad ():
            next_a_all =torch .stack (
            [self .t_actors [i ](s2_for_target [:,i ,:])for i in range (self .n )],dim =1
            )
            g_noise =torch .randn_like (next_a_all )*self .td3_sigma
            g_noise =torch .clamp (g_noise ,-self .td3_clip ,self .td3_clip )
            next_a_all =torch .clamp (next_a_all +g_noise ,-1.0 ,1.0 )

            next_a_kw =next_a_all *MAX_EV_POWER_KW
            current_socs_s2_g =denormalize_soc (ev_block_for_target [...,1 ])
            clamped_actions_next ,next_station_powers =self ._apply_soc_constraint (
            next_a_kw ,current_socs_s2_g ,ev_padding_mask_for_target
            )
            s2_global =self ._convert_to_global_critic_obs (s2_for_target ,next_station_powers )

            next_a_kw_masked =clamped_actions_next .masked_fill (ev_padding_mask_for_target ,0.0 )
            next_a_kw_normalized =torch .clamp (next_a_kw_masked /MAX_EV_POWER_KW ,-1.0 ,1.0 )
            next_station_powers_normalized =torch .clamp (
            next_station_powers /max_station_power ,-1.0 ,1.0
            )

            tq1_s ,_ =self .t_global_critic1 (
            s2_global ,next_a_kw_normalized ,key_padding_mask_for_target ,
            actual_station_powers =next_station_powers_normalized ,
            )
            tq2_s ,_ =self .t_global_critic2 (
            s2_global ,next_a_kw_normalized ,key_padding_mask_for_target ,
            actual_station_powers =next_station_powers_normalized ,
            )
            target_q_global =torch .min (tq1_s ,tq2_s )

            # One-step global TD target in the n-step sampler format:
            # y = r_global + GAMMA_GLOBAL * min(Q1', Q2') * (1 - done_any).
            done_mask_global =d_n .max (dim =1 ,keepdim =True )[0 ]
            y_global =r_global_n +GAMMA_GLOBAL *target_q_global *(1 -done_mask_global )
            y_global =torch .nan_to_num_ (y_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        s_global =self ._convert_to_global_critic_obs (s ,actual_station_powers )
        a_actual_global =a_actual .masked_fill (ev_padding_mask ,0.0 )
        actual_station_powers_normalized =torch .clamp (
        actual_station_powers /max_station_power ,-1.0 ,1.0
        )

        q1_s ,_ =self .global_critic1 (
        s_global ,a_actual_global ,key_padding_mask ,
        actual_station_powers =actual_station_powers_normalized ,
        )
        q2_s ,_ =self .global_critic2 (
        s_global ,a_actual_global ,key_padding_mask ,
        actual_station_powers =actual_station_powers_normalized ,
        )
        q1_s =torch .nan_to_num_ (q1_s ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        q2_s =torch .nan_to_num_ (q2_s ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        loss_g1 =self .loss_fn (q1_s ,y_global )
        loss_g2 =self .loss_fn (q2_s ,y_global )
        loss_g1 =torch .nan_to_num_ (loss_g1 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        loss_g2 =torch .nan_to_num_ (loss_g2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        loss_g_total =loss_g1 +loss_g2
        if not torch .isfinite (loss_g_total ):
            return

        self .opt_global_c1 .zero_grad ()
        self .opt_global_c2 .zero_grad ()
        loss_g_total .backward ()

        has_nan_inf =any (
        p .grad is not None and not torch .isfinite (p .grad ).all ()
        for p in self .global_critic1 .parameters ()
        )
        if not has_nan_inf :
            self .clip_bias_gradients (self .global_critic1 ,max_norm =BIAS_GRAD_CLIP_MAX )
            gn_b =torch .nn .utils .clip_grad_norm_ (
            self .global_critic1 .parameters (),max_norm =GRAD_CLIP_MAX_GLOBAL
            ).item ()
            gn_a =min (gn_b ,GRAD_CLIP_MAX_GLOBAL )
            self .last_global_critic_clip_count =1 if gn_b >GRAD_CLIP_MAX_GLOBAL else 0
            self .last_global_critic_grad_norm =gn_a if math .isfinite (gn_a )else 0.0
            self .last_global_critic_grad_norm_before_clip =gn_b if math .isfinite (gn_b )else 0.0
            self .last_global_critic_loss =loss_g1 .item ()if torch .isfinite (loss_g1 )else 0.0
            if math .isfinite (gn_a ):
                self .opt_global_c1 .step ()

        has_nan_inf2 =any (
        p .grad is not None and not torch .isfinite (p .grad ).all ()
        for p in self .global_critic2 .parameters ()
        )
        if not has_nan_inf2 :
            self .clip_bias_gradients (self .global_critic2 ,max_norm =BIAS_GRAD_CLIP_MAX )
            gn_b2 =torch .nn .utils .clip_grad_norm_ (
            self .global_critic2 .parameters (),max_norm =GRAD_CLIP_MAX_GLOBAL
            ).item ()
            gn_a2 =min (gn_b2 ,GRAD_CLIP_MAX_GLOBAL )
            if math .isfinite (gn_a2 ):
                self .opt_global_c2 .step ()


    def update (self):
        """
        Run one gradient update if replay memory has passed warmup.

        Training is disabled in test mode. After warmup, one batch updates local
        critics, the twin global critics, and, every `POLICY_DELAY` calls, the
        actors plus target networks. Global replay sampling is fixed to one-step
        targets and uses `GAMMA_GLOBAL`.
        """
        if self .test_mode :
            self ._zero_update_logs ()
            return

        if self .buf .size <WARMUP_STEPS :
            self ._zero_update_logs ()
            return

        self .update_step +=1
        actor_update_due =(self .update_step %self .policy_delay ==0 )

        sampled =self .buf .sample_with_nstep_global (self .batch ,1 ,GAMMA_GLOBAL )
        s ,s2 ,a ,r_local ,_r_global ,d ,actual_station_powers ,actual_ev_power_kw ,r_global_n ,s2_n ,d_n ,_n_eff =sampled
        ctx =self ._build_update_ctx (s ,s2 ,a ,r_local ,d ,
        actual_station_powers ,actual_ev_power_kw )
        # One-step global-target fields returned by the replay sampler.
        ctx ['r_global_n']=r_global_n
        ctx ['s2_n']=s2_n
        ctx ['d_n']=d_n

        local_critic_clip_count =self ._update_local_critics (ctx )
        self ._update_global_critic (ctx )
        self .actor_losses =[0.0 ]*self .n
        actor_clip_count =0
        if actor_update_due :
            actor_clip_count =self ._update_actors (ctx )
            self ._polyak_update_targets ()

        self ._aggregate_update_logs (local_critic_clip_count ,actor_clip_count )


    def episode_start (self ):
        """Advance exploration schedules and reset per-episode diagnostics."""
        if not self .test_mode :
            self .current_episode +=1

        self ._ep_q_raw_global =[]
        self ._ep_q_raw_local =[[]for _ in range (self .n )]

        if self .test_mode :
            self .epsilon =0.0
            self .ou_noise_scale =0.0
        else :
            ep_in_phase =self .current_episode
            self .epsilon =linear_epsilon_decay (
            ep_in_phase ,
            self .epsilon_start_episode ,self .epsilon_end_episode ,
            self .epsilon_initial ,self .epsilon_final ,
            )

            s0n ,s1n =self .ou_noise_start_episode ,self .ou_noise_end_episode
            n0 ,n1 =self .ou_noise_scale_initial ,self .ou_noise_scale_final
            if ep_in_phase <s0n :
                self .ou_noise_scale =0.0
            elif ep_in_phase >=s1n :
                self .ou_noise_scale =n1
            else :
                rn =(ep_in_phase -s0n )/max (1 ,(s1n -s0n ))
                self .ou_noise_scale =n0 +(n1 -n0 )*rn
            self .ou_noise_scale =max (self .ou_noise_scale_final ,self .ou_noise_scale )

        self .ou_noise .reset ()

    def episode_end (self ):
        """Hook kept for symmetry with benchmark agents."""
        if self .test_mode :
            return

    def set_test_mode (self ,mode :bool ):
        """Switch all online and target networks between train and eval mode."""
        self .test_mode =mode
        self .training =not mode

        if mode :
            for actor in self .actors :
                actor .eval ()
            for critic in self .critics :
                critic .eval ()
            for t_actor in self .t_actors :
                t_actor .eval ()
            for t_critic in self .t_critics :
                t_critic .eval ()
            self .global_critic1 .eval ()
            self .global_critic2 .eval ()
            self .t_global_critic1 .eval ()
            self .t_global_critic2 .eval ()
        else :
            for actor in self .actors :
                actor .train ()
            for critic in self .critics :
                critic .train ()
            for t_actor in self .t_actors :
                t_actor .train ()
            for t_critic in self .t_critics :
                t_critic .train ()
            self .global_critic1 .train ()
            self .global_critic2 .train ()
            self .t_global_critic1 .train ()
            self .t_global_critic2 .train ()

    def cache_experience (self ,s ,s2 ,a ,r_local ,r_global ,d ,
    actual_station_powers =None ,actual_ev_power_kw =None ):
        """Store an executable environment transition in replay memory."""
        if self .test_mode :
            return
        self .buf .cache (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_power_kw )

    def save_actors (self ,path ,episode ):
        """Save each station actor as a separate checkpoint file."""
        os .makedirs (path ,exist_ok =True )
        for i in range (self .n ):
            torch .save (self .actors [i ].state_dict (),os .path .join (path ,f"actor_{i}_ep{episode}.pth"))

    def load_actors (self ,path ,episode ,map_location =None ):
        """Load station actor checkpoints and synchronize target actors."""
        for i in range (self .n ):
            actor_path =os .path .join (path ,f"actor_{i}_ep{episode}.pth")
            try :
                sd =torch .load (
                actor_path ,
                map_location =map_location if map_location is not None else device ,
                weights_only =True ,
                )
            except TypeError :
                sd =torch .load (
                actor_path ,
                map_location =map_location if map_location is not None else device ,
                )
            head_key ="ev_action_head.0.weight"
            current_sd =self .actors [i ].state_dict ()
            if head_key in sd and head_key in current_sd :
                loaded_w =sd [head_key]
                target_w =current_sd [head_key]
                if (
                loaded_w .dim ()==2
                and target_w .dim ()==2
                and loaded_w .shape [0 ]==target_w .shape [0 ]
                and loaded_w .shape [1 ]==target_w .shape [1 ]+1
                ):
                    tail_start =target_w .shape [1 ]-int (self .local_tail_dim )
                    sd [head_key]=torch .cat (
                    [loaded_w [:,:tail_start ],loaded_w [:,tail_start +1 :]],
                    dim =1 ,
                    )
            self .actors [i ].load_state_dict (sd )
            if i <len (self .t_actors ):
                self .t_actors [i ].load_state_dict (self .actors [i ].state_dict ())
