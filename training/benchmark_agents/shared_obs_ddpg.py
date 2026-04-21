
"""
Centralized DDPG benchmark agent.

This agent uses:
- one centralized actor that sees all stations at once
- one centralized critic that evaluates the joint action

The class name `SharedObsDDPG` is kept for compatibility with the existing
config / training / execute entrypoints.
"""

import copy 
import math 
import os 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 
import torch .optim as optim 

from Config import (
ACTOR_HIDDEN_SIZE ,
BATCH_SIZE ,
EPSILON_FINAL ,
EPSILON_INITIAL ,
EPSILON_START_EPISODE ,
GAMMA ,
GRAD_CLIP_MAX ,
LOCAL_CRITIC_HIDDEN_SIZE ,
LR_ACTOR ,
LR_CRITIC_LOCAL ,
MAX_EV_POWER_KW ,
MEMORY_SIZE ,
OU_NOISE_GAIN ,
OU_NOISE_SCALE_INITIAL ,
OU_NOISE_START_EPISODE ,
RANDOM_ACTION_RANGE ,
SHARED_OBS_EPSILON_END_EPISODE ,
SHARED_OBS_GLOBAL_REWARD_WEIGHT ,
SHARED_OBS_OU_NOISE_END_EPISODE ,
SHARED_OBS_OU_NOISE_SCALE_FINAL ,
SMOOTHL1_BETA ,
TAU ,
)
from environment.observation_config import EV_FEAT_DIM 

from training.Agent.replay_buffer import ReplayBuffer 
from training.Agent.noise import OUNoise ,linear_epsilon_decay 

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


class CentralizedJointActor (nn .Module ):
    def __init__ (self ,global_obs_dim ,n_agents ,max_evs_per_station ,hid =ACTOR_HIDDEN_SIZE ):
        super ().__init__ ()
        self .global_obs_dim =int (global_obs_dim )
        self .n_agents =int (n_agents )
        self .max_evs =int (max_evs_per_station )
        self .joint_action_dim =self .n_agents *self .max_evs 

        self .net =nn .Sequential (
        nn .Linear (self .global_obs_dim ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        )
        self .action_head =nn .Linear (hid ,self .joint_action_dim )
        self .apply (self ._init_weights )

    @staticmethod 
    def _init_weights (module ):
        if isinstance (module ,nn .Linear ):
            nn .init .xavier_uniform_ (module .weight ,gain =0.1 )
            nn .init .constant_ (module .bias ,0.0 )

    def forward (self ,global_obs ):
        squeeze_output =False 
        if global_obs .dim ()==1 :
            global_obs =global_obs .unsqueeze (0 )
            squeeze_output =True 

        features =self .net (global_obs )
        raw_actions =torch .tanh (self .action_head (features ))
        raw_actions =raw_actions .view (-1 ,self .n_agents ,self .max_evs )

        if squeeze_output :
            raw_actions =raw_actions .squeeze (0 )
        return raw_actions 


class CentralizedJointCritic (nn .Module ):
    def __init__ (self ,global_obs_dim ,joint_action_dim ,hid =LOCAL_CRITIC_HIDDEN_SIZE ):
        super ().__init__ ()
        self .global_obs_dim =int (global_obs_dim )
        self .joint_action_dim =int (joint_action_dim )
        input_dim =self .global_obs_dim +self .joint_action_dim 

        self .net =nn .Sequential (
        nn .Linear (input_dim ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,1 ),
        )
        self .apply (self ._init_weights )

    @staticmethod 
    def _init_weights (module ):
        if isinstance (module ,nn .Linear ):
            nn .init .xavier_uniform_ (module .weight ,gain =0.1 )
            nn .init .constant_ (module .bias ,0.0 )

    def forward (self ,global_obs ,joint_action ):
        if global_obs .dim ()==1 :
            global_obs =global_obs .unsqueeze (0 )
        if joint_action .dim ()==2 :
            joint_action =joint_action .unsqueeze (0 )

        critic_input =torch .cat ([global_obs ,joint_action .reshape (global_obs .size (0 ),-1 )],dim =-1 )
        return self .net (critic_input )


class SharedObsDDPG :
    """
    Centralized RL benchmark agent with a single joint actor and critic.
    """

    visualizer_layout ="centralized_joint"

    def __init__ (
    self ,
    s_dim ,
    max_evs_per_station ,
    n_agent ,
    gamma =GAMMA ,
    tau =TAU ,
    batch =BATCH_SIZE ,
    lr_a =LR_ACTOR ,
    lr_c =LR_CRITIC_LOCAL ,
    num_episodes =10000 ,
    smoothl1_beta =SMOOTHL1_BETA ,
    global_reward_weight =None ,
    epsilon_end_episode =None ,
    ou_noise_end_episode =None ,
    ou_noise_scale_final =None ,
    ):
        self .s_dim =int (s_dim )
        self .max_evs =int (max_evs_per_station )
        self .a_dim =self .max_evs 
        self .n =int (n_agent )
        self .gamma =float (gamma )
        self .tau =float (tau )
        self .batch =int (batch )
        self .total_episodes =int (num_episodes )
        self .current_episode =0 
        self .update_step =0 
        self .training =True 
        self .test_mode =False 
        self .use_tensorboard =False 
        self .writer =None 

        self .global_obs_dim =self .n *self .s_dim 
        self .joint_action_dim =self .n *self .max_evs 
        self .loss_beta =float (smoothl1_beta )
        self .global_reward_weight =(
        float (SHARED_OBS_GLOBAL_REWARD_WEIGHT )
        if global_reward_weight is None else float (global_reward_weight )
        )

        self .random_action_range =RANDOM_ACTION_RANGE 
        self .epsilon_start_episode =int (EPSILON_START_EPISODE )
        self .epsilon_end_episode =(
        int (SHARED_OBS_EPSILON_END_EPISODE )
        if epsilon_end_episode is None else int (epsilon_end_episode )
        )
        self .epsilon_initial =float (EPSILON_INITIAL )
        self .epsilon_final =float (EPSILON_FINAL )
        self .epsilon =self .epsilon_initial 

        self .ou_noise_start_episode =int (OU_NOISE_START_EPISODE )
        self .ou_noise_end_episode =(
        int (SHARED_OBS_OU_NOISE_END_EPISODE )
        if ou_noise_end_episode is None else int (ou_noise_end_episode )
        )
        self .ou_noise_scale_initial =float (OU_NOISE_SCALE_INITIAL )
        self .ou_noise_scale_final =(
        float (SHARED_OBS_OU_NOISE_SCALE_FINAL )
        if ou_noise_scale_final is None else float (ou_noise_scale_final )
        )
        self .ou_noise_scale =self .ou_noise_scale_initial 
        self .ou_noise =OUNoise (self .n ,self .max_evs )

        self .active_evs =[0 ]*self .n 

        self .buf =ReplayBuffer (cap =int (MEMORY_SIZE ))
        self .buf .maddpg_ref =self 

        self .actor =CentralizedJointActor (
        global_obs_dim =self .global_obs_dim ,
        n_agents =self .n ,
        max_evs_per_station =self .max_evs ,
        ).to (device )
        self .t_actor =copy .deepcopy (self .actor )

        self .critic =CentralizedJointCritic (
        global_obs_dim =self .global_obs_dim ,
        joint_action_dim =self .joint_action_dim ,
        ).to (device )
        self .t_critic =copy .deepcopy (self .critic )

        self .opt_a =optim .Adam (self .actor .parameters (),lr =lr_a )
        self .opt_c =optim .Adam (self .critic .parameters (),lr =lr_c )

        self ._reset_log_state ()

    def _reset_log_state (self ):
        self .last_central_q_value =0.0 
        self .last_central_actor_loss =0.0 
        self .last_central_critic_loss =0.0 
        self .last_central_actor_grad_norm_before_clip =0.0 
        self .last_central_actor_grad_norm =0.0 
        self .last_central_critic_grad_norm_before_clip =0.0 
        self .last_central_critic_grad_norm =0.0 
        self .last_central_actor_clip_count =0 
        self .last_central_critic_clip_count =0 

        
        self .last_actor_loss =0.0 
        self .last_critic_loss =0.0 
        self .last_actor_grad_norm =0.0 
        self .last_local_critic_grad_norm =0.0 
        self .last_global_critic_grad_norm =0.0 
        self .last_global_critic_grad_norm_before_clip =0.0 
        self .last_global_critic_loss =0.0 
        self .last_global_critic_clip_count =0 

        self .critic_norms =[]
        self .critic_norms_before_clip =[]
        self .actor_norms =[]
        self .actor_norms_before_clip =[]
        self .critic_losses =[]
        self .actor_losses =[]
        self .local_critic_clip_counts =[]
        self .actor_clip_counts =[]

    def _reshape_global_obs (self ,state ):
        if state .dim ()==2 :
            return state .reshape (1 ,-1 )
        return state .reshape (state .size (0 ),-1 )

    def _action_mask_from_state (self ,state ):
        batch_state =state 
        squeeze_output =False 
        if batch_state .dim ()==2 :
            batch_state =batch_state .unsqueeze (0 )
            squeeze_output =True 

        presence =batch_state [:,:,:self .max_evs *EV_FEAT_DIM ]
        presence =presence .reshape (batch_state .size (0 ),self .n ,self .max_evs ,EV_FEAT_DIM )[...,0 ]
        mask =(presence >0.0 ).to (batch_state .dtype )

        if squeeze_output :
            mask =mask .squeeze (0 )
        return mask 

    def _actor_forward (self ,state ):
        global_obs =self ._reshape_global_obs (state )
        raw_actions =self .actor (global_obs )
        action_mask =self ._action_mask_from_state (state )
        masked_actions =raw_actions *action_mask 
        return masked_actions ,action_mask 

    def _target_actor_forward (self ,state ):
        global_obs =self ._reshape_global_obs (state )
        raw_actions =self .t_actor (global_obs )
        action_mask =self ._action_mask_from_state (state )
        return raw_actions *action_mask 

    def update_active_evs (self ,env ):
        self .active_evs =[0 ]*self .n 
        for idx in range (min (self .n ,env .num_stations )):
            self .active_evs [idx ]=int (env .ev_mask [idx ].sum ().item ())

    def act (self ,state ,env =None ,noise =True ):
        if self .test_mode :
            noise =False 

        if not isinstance (state ,torch .Tensor ):
            state =torch .as_tensor (state ,dtype =torch .float32 ,device =device )
        else :
            state =state .to (device )

        if env is not None :
            self .update_active_evs (env )
        else :
            inferred_mask =self ._action_mask_from_state (state )
            if inferred_mask .dim ()==2 :
                self .active_evs =inferred_mask .sum (dim =1 ).to (torch .int64 ).tolist ()

        with torch .no_grad ():
            deterministic_actions ,action_mask =self ._actor_forward (state )
            actions =deterministic_actions .clone ()

            if noise :
                if torch .rand (1 ,device =device ).item ()<self .epsilon :
                    lo ,hi =(
                    (self .random_action_range [0 ],self .random_action_range [1 ])
                    if isinstance (self .random_action_range ,(tuple ,list ))
                    else (-float (self .random_action_range ),float (self .random_action_range ))
                    )
                    random_actions =torch .empty_like (actions ).uniform_ (lo ,hi )
                    actions =random_actions *action_mask 
                elif self .ou_noise_scale >0.0 :
                    ou_noise =self .ou_noise .sample (self .active_evs ).to (device )
                    actions =actions +(OU_NOISE_GAIN *self .ou_noise_scale )*ou_noise 
                    actions =actions *action_mask 

            actions =torch .clamp (actions ,-1.0 ,1.0 )
        return actions .squeeze (0 )if actions .dim ()==3 and actions .size (0 )==1 else actions 

    def update (self ):
        if self .test_mode :
            self ._reset_log_state ()
            return 
        if self .buf .size <self .batch :
            return 

        self ._reset_log_state ()

        s ,s_ ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes =self .buf .sample (self .batch )
        del actual_station_powers 

        global_obs =s .reshape (s .size (0 ),-1 )
        next_global_obs =s_ .reshape (s_ .size (0 ),-1 )
        joint_action =torch .clamp (actual_ev_soc_changes /float (MAX_EV_POWER_KW ),-1.0 ,1.0 )

        reward_joint =r_local .mean (dim =1 )+self .global_reward_weight *r_global .squeeze (-1 )
        done_joint =d .max (dim =1 ).values 

        with torch .no_grad ():
            next_action =self ._target_actor_forward (s_ )
            q_target =self .t_critic (next_global_obs ,next_action ).squeeze (-1 )
            y =reward_joint +self .gamma *(1.0 -done_joint )*q_target 

        q_current =self .critic (global_obs ,joint_action ).squeeze (-1 )
        critic_loss =F .smooth_l1_loss (q_current ,y ,beta =self .loss_beta )

        self .opt_c .zero_grad ()
        critic_loss .backward ()
        critic_grad_norm =torch .nn .utils .clip_grad_norm_ (self .critic .parameters (),GRAD_CLIP_MAX )
        critic_grad_norm_value =float (critic_grad_norm )
        if math .isfinite (critic_grad_norm_value ):
            self .opt_c .step ()

        pred_action ,_ =self ._actor_forward (s )
        actor_loss =-self .critic (global_obs ,pred_action ).mean ()

        self .opt_a .zero_grad ()
        actor_loss .backward ()
        actor_grad_norm =torch .nn .utils .clip_grad_norm_ (self .actor .parameters (),GRAD_CLIP_MAX )
        actor_grad_norm_value =float (actor_grad_norm )
        if math .isfinite (actor_grad_norm_value ):
            self .opt_a .step ()

        self ._soft_update (self .t_actor ,self .actor )
        self ._soft_update (self .t_critic ,self .critic )

        q_mean =float (q_current .detach ().mean ().item ())if torch .isfinite (q_current ).all ()else 0.0 
        critic_loss_value =float (critic_loss .item ())if torch .isfinite (critic_loss )else 0.0 
        actor_loss_value =float (actor_loss .item ())if torch .isfinite (actor_loss )else 0.0 
        critic_grad_after =min (critic_grad_norm_value ,GRAD_CLIP_MAX )if math .isfinite (critic_grad_norm_value )else 0.0 
        actor_grad_after =min (actor_grad_norm_value ,GRAD_CLIP_MAX )if math .isfinite (actor_grad_norm_value )else 0.0 

        self .last_central_q_value =q_mean 
        self .last_central_critic_loss =critic_loss_value 
        self .last_central_actor_loss =actor_loss_value 
        self .last_central_critic_grad_norm_before_clip =(
        critic_grad_norm_value if math .isfinite (critic_grad_norm_value )else 0.0 
        )
        self .last_central_actor_grad_norm_before_clip =(
        actor_grad_norm_value if math .isfinite (actor_grad_norm_value )else 0.0 
        )
        self .last_central_critic_grad_norm =critic_grad_after 
        self .last_central_actor_grad_norm =actor_grad_after 
        self .last_central_critic_clip_count =int (critic_grad_norm_value >GRAD_CLIP_MAX )
        self .last_central_actor_clip_count =int (actor_grad_norm_value >GRAD_CLIP_MAX )

        self .last_actor_loss =actor_loss_value 
        self .last_critic_loss =critic_loss_value 
        self .last_actor_grad_norm =actor_grad_after 
        self .last_local_critic_grad_norm =critic_grad_after 
        self .last_global_critic_grad_norm =critic_grad_after 
        self .last_global_critic_grad_norm_before_clip =self .last_central_critic_grad_norm_before_clip 
        self .last_global_critic_loss =critic_loss_value 
        self .last_global_critic_clip_count =self .last_central_critic_clip_count 

        self .update_step +=1 

    def _soft_update (self ,target ,source ):
        for target_param ,param in zip (target .parameters (),source .parameters ()):
            target_param .data .copy_ (
            target_param .data *(1.0 -self .tau )+param .data *self .tau 
            )

    def cache_experience (self ,s ,s_ ,delta_soc ,r_local ,r_global ,done ,actual_station_powers ):
        self .buf .cache (
        s ,
        s_ ,
        delta_soc ,
        r_local ,
        r_global ,
        done ,
        actual_station_powers =actual_station_powers ,
        actual_ev_soc_changes =delta_soc ,
        )

    def set_test_mode (self ,mode :bool ):
        self .test_mode =bool (mode )
        self .training =not self .test_mode 
        modules =[self .actor ,self .t_actor ,self .critic ,self .t_critic ]
        for module in modules :
            if self .test_mode :
                module .eval ()
            else :
                module .train ()

    def episode_start (self ):
        if not self .test_mode :
            self .current_episode +=1 

        ep =int (self .current_episode )
        self .epsilon =linear_epsilon_decay (
        ep ,
        self .epsilon_start_episode ,self .epsilon_end_episode ,
        self .epsilon_initial ,self .epsilon_final ,
        )

        if ep <self .ou_noise_start_episode :
            self .ou_noise_scale =0.0 
        elif ep >=self .ou_noise_end_episode :
            self .ou_noise_scale =self .ou_noise_scale_final 
        else :
            progress =(ep -self .ou_noise_start_episode )/max (
            1 ,self .ou_noise_end_episode -self .ou_noise_start_episode 
            )
            self .ou_noise_scale =(
            self .ou_noise_scale_initial 
            +(self .ou_noise_scale_final -self .ou_noise_scale_initial )*progress 
            )

        try :
            self .ou_noise .reset ()
        except Exception :
            pass 

    def episode_end (self ):
        pass 

    def debug_actor_output (self ,state ,station_idx =0 ):
        if not isinstance (state ,torch .Tensor ):
            state =torch .as_tensor (state ,dtype =torch .float32 ,device =device )
        else :
            state =state .to (device )

        with torch .no_grad ():
            actions ,_ =self ._actor_forward (state )
            if actions .dim ()==2 :
                station_actions =actions [station_idx ]
            else :
                station_actions =actions [0 ,station_idx ]
        return station_actions .detach ().cpu ().tolist ()

    def save_models (self ,path ,episode ):
        os .makedirs (path ,exist_ok =True )
        torch .save (self .actor .state_dict (),os .path .join (path ,f"shared_actor_ep{episode}.pth"))
        torch .save (self .critic .state_dict (),os .path .join (path ,f"shared_critic_ep{episode}.pth"))

    def load_models (self ,path ,episode ,map_location =None ):
        map_location =map_location if map_location is not None else device 
        try :
            self .actor .load_state_dict (
            torch .load (os .path .join (path ,f"shared_actor_ep{episode}.pth"),map_location =map_location )
            )
            self .critic .load_state_dict (
            torch .load (os .path .join (path ,f"shared_critic_ep{episode}.pth"),map_location =map_location )
            )
        except RuntimeError as exc :
            raise RuntimeError (
            "Centralized RL checkpoint shape mismatch. Legacy shared-observation checkpoints "
            "from the old per-station-critic design are not compatible; retrain the centralized model."
            )from exc 
        self .t_actor .load_state_dict (self .actor .state_dict ())
        self .t_critic .load_state_dict (self .critic .state_dict ())

    def save_actors (self ,path ,episode ):
        os .makedirs (path ,exist_ok =True )
        torch .save (self .actor .state_dict (),os .path .join (path ,f"shared_actor_ep{episode}.pth"))

    def load_actors (self ,path ,episode ,map_location =None ):
        map_location =map_location if map_location is not None else device 
        actor_state =torch .load (
        os .path .join (path ,f"shared_actor_ep{episode}.pth"),
        map_location =map_location ,
        )
        try :
            self .actor .load_state_dict (actor_state )
        except RuntimeError as exc :
            raise RuntimeError (
            "Centralized RL actor checkpoint shape mismatch. Legacy shared-observation actor files "
            "from the old architecture are not compatible; retrain the centralized model."
            )from exc 
        self .t_actor .load_state_dict (self .actor .state_dict ())
