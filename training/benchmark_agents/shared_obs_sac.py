
"""
Centralized SAC (Soft Actor-Critic) benchmark agent.

Single centralized Gaussian actor + twin centralized critic.
Auto-tunes entropy temperature (alpha).

Drop-in replacement for SharedObsDDPG with the same public interface:
  act / update / episode_start / episode_end / cache_experience /
  set_test_mode / save_models / load_models / save_actors / load_actors /
  update_active_evs / debug_actor_output
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
GAMMA ,
GRAD_CLIP_MAX ,
LOCAL_CRITIC_HIDDEN_SIZE ,
MAX_EV_POWER_KW ,
MEMORY_SIZE ,
SAC_ALPHA_INIT ,
SAC_GLOBAL_REWARD_WEIGHT ,
SAC_LR_ACTOR ,
SAC_LR_ALPHA ,
SAC_LR_CRITIC ,
SAC_TAU ,
SAC_TARGET_ENTROPY_SCALE ,
)
from environment.observation_config import EV_FEAT_DIM 

from training.Agent.replay_buffer import ReplayBuffer 

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")

_LEAKY_GAIN =nn .init .calculate_gain ("leaky_relu",0.1 )






class CentralizedGaussianActor (nn .Module ):
    "Documentation."

    LOG_STD_MIN =-20 
    LOG_STD_MAX =2 

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
        self .mean_head =nn .Linear (hid ,self .joint_action_dim )
        self .log_std_head =nn .Linear (hid ,self .joint_action_dim )
        self .apply (self ._init_weights )

    @staticmethod 
    def _init_weights (module ):
        if isinstance (module ,nn .Linear ):
            nn .init .xavier_uniform_ (module .weight ,gain =_LEAKY_GAIN )
            nn .init .constant_ (module .bias ,0.0 )

    def _backbone (self ,global_obs ):
        """Run shared MLP. Returns (mean, log_std, was_squeezed)."""
        squeeze =global_obs .dim ()==1 
        if squeeze :
            global_obs =global_obs .unsqueeze (0 )
        feat =self .net (global_obs )
        mean =self .mean_head (feat )
        log_std =self .log_std_head (feat ).clamp (self .LOG_STD_MIN ,self .LOG_STD_MAX )
        return mean ,log_std ,squeeze 

        

    def forward (self ,global_obs ):
        """Deterministic action: tanh(mean), shape (batch, n_agents, max_evs)."""
        mean ,_ ,squeeze =self ._backbone (global_obs )
        action =torch .tanh (mean ).view (-1 ,self .n_agents ,self .max_evs )
        if squeeze :
            action =action .squeeze (0 )
        return action 

        

    def sample (self ,global_obs ):
        "Documentation."
        mean ,log_std ,squeeze =self ._backbone (global_obs )
        std =log_std .exp ()

        eps =torch .randn_like (std )
        pre_tanh =mean +std *eps 
        action_flat =torch .tanh (pre_tanh )

        
        log_prob =(
        -0.5 *eps .pow (2 )
        -log_std 
        -0.5 *math .log (2.0 *math .pi )
        -torch .log (1.0 -action_flat .pow (2 )+1e-6 )
        ).sum (-1 )

        action =action_flat .view (-1 ,self .n_agents ,self .max_evs )
        if squeeze :
            action =action .squeeze (0 )
            log_prob =log_prob .squeeze (0 )

        return action ,log_prob 


class CentralizedTwinCritic (nn .Module ):
    """Twin Q-networks (SAC-style) that take global_obs + joint_action."""

    def __init__ (self ,global_obs_dim ,joint_action_dim ,hid =LOCAL_CRITIC_HIDDEN_SIZE ):
        super ().__init__ ()
        input_dim =int (global_obs_dim )+int (joint_action_dim )

        def _make ():
            return nn .Sequential (
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

        self .q1_net =_make ()
        self .q2_net =_make ()
        self .apply (self ._init_weights )

    @staticmethod 
    def _init_weights (module ):
        if isinstance (module ,nn .Linear ):
            nn .init .xavier_uniform_ (module .weight ,gain =_LEAKY_GAIN )
            nn .init .constant_ (module .bias ,0.0 )

    def _cat (self ,global_obs ,joint_action ):
        if global_obs .dim ()==1 :
            global_obs =global_obs .unsqueeze (0 )
        if joint_action .dim ()==2 :
            joint_action =joint_action .unsqueeze (0 )
        return torch .cat ([global_obs ,joint_action .reshape (global_obs .size (0 ),-1 )],dim =-1 )

    def forward (self ,global_obs ,joint_action ):
        """Returns (Q1, Q2), each shape (batch, 1)."""
        x =self ._cat (global_obs ,joint_action )
        return self .q1_net (x ),self .q2_net (x )


        
        
        

class SharedObsSAC :
    """
    Centralized SAC benchmark agent.

    - One CentralizedGaussianActor (stochastic, auto-entropy)
    - One CentralizedTwinCritic  (Q1 + Q2, targets)
    - Auto-tuned entropy temperature (log_alpha)
    - Same public interface as SharedObsDDPG.
    """

    visualizer_layout ="centralized_joint"

    def __init__ (
    self ,
    s_dim ,
    max_evs_per_station ,
    n_agent ,
    gamma =GAMMA ,
    tau =None ,
    batch =BATCH_SIZE ,
    lr_a =None ,
    lr_c =None ,
    lr_alpha =None ,
    num_episodes =10000 ,
    global_reward_weight =None ,
    alpha_init =None ,
    target_entropy_scale =None ,
    ):
        self .s_dim =int (s_dim )
        self .max_evs =int (max_evs_per_station )
        self .n =int (n_agent )
        self .gamma =float (gamma )
        self .tau =float (SAC_TAU if tau is None else tau )
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

        self .global_reward_weight =float (
        SAC_GLOBAL_REWARD_WEIGHT if global_reward_weight is None else global_reward_weight 
        )

        
        _alpha_init =float (SAC_ALPHA_INIT if alpha_init is None else alpha_init )
        _te_scale =float (
        SAC_TARGET_ENTROPY_SCALE if target_entropy_scale is None else target_entropy_scale 
        )
        self .target_entropy =-float (self .joint_action_dim )*_te_scale 
        self .log_alpha =torch .tensor (
        math .log (_alpha_init ),dtype =torch .float32 ,device =device ,requires_grad =True 
        )
        _lr_alpha =float (SAC_LR_ALPHA if lr_alpha is None else lr_alpha )
        self .opt_alpha =optim .Adam ([self .log_alpha ],lr =_lr_alpha )

        
        self .buf =ReplayBuffer (cap =int (MEMORY_SIZE ))
        self .buf .maddpg_ref =self 
        self .active_evs =[0 ]*self .n 

        
        _lr_a =float (SAC_LR_ACTOR if lr_a is None else lr_a )
        _lr_c =float (SAC_LR_CRITIC if lr_c is None else lr_c )

        self .actor =CentralizedGaussianActor (
        global_obs_dim =self .global_obs_dim ,
        n_agents =self .n ,
        max_evs_per_station =self .max_evs ,
        ).to (device )

        self .critic =CentralizedTwinCritic (
        global_obs_dim =self .global_obs_dim ,
        joint_action_dim =self .joint_action_dim ,
        ).to (device )
        self .t_critic =copy .deepcopy (self .critic )
        
        for p in self .t_critic .parameters ():
            p .requires_grad_ (False )

        self .opt_a =optim .Adam (self .actor .parameters (),lr =_lr_a )
        self .opt_c =optim .Adam (self .critic .parameters (),lr =_lr_c )

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
        self .last_sac_alpha =float (self .log_alpha .exp ().item ())
        self .last_sac_entropy =0.0 
        self .last_sac_alpha_loss =0.0 

        
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
        """Returns float mask (batch, n, max_evs): 1 where EV present, 0 elsewhere."""
        batch_state =state 
        squeeze_out =False 
        if batch_state .dim ()==2 :
            batch_state =batch_state .unsqueeze (0 )
            squeeze_out =True 

        presence =batch_state [:,:,:self .max_evs *EV_FEAT_DIM ]
        presence =presence .reshape (
        batch_state .size (0 ),self .n ,self .max_evs ,EV_FEAT_DIM 
        )[...,0 ]
        mask =(presence >0.0 ).to (batch_state .dtype )

        if squeeze_out :
            mask =mask .squeeze (0 )
        return mask 

        
        
        

    def update_active_evs (self ,env ):
        self .active_evs =[0 ]*self .n 
        for idx in range (min (self .n ,env .num_stations )):
            self .active_evs [idx ]=int (env .ev_mask [idx ].sum ().item ())

            
            
            

    def act (self ,state ,env =None ,noise =True ):
        "Documentation."
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

        global_obs =self ._reshape_global_obs (state )
        action_mask =self ._action_mask_from_state (state )

        with torch .no_grad ():
            if noise :
                actions ,_ =self .actor .sample (global_obs )
            else :
                actions =self .actor (global_obs )

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

        s ,s_ ,a ,r_local ,r_global ,d ,_station_powers ,actual_ev_soc_changes =self .buf .sample (
        self .batch 
        )

        global_obs =s .reshape (s .size (0 ),-1 )
        next_global_obs =s_ .reshape (s_ .size (0 ),-1 )
        joint_action =torch .clamp (actual_ev_soc_changes /float (MAX_EV_POWER_KW ),-1.0 ,1.0 )

        reward_joint =r_local .mean (dim =1 )+self .global_reward_weight *r_global .squeeze (-1 )
        done_joint =d .max (dim =1 ).values 

        alpha =self .log_alpha .exp ().detach ()

        
        with torch .no_grad ():
            next_action ,next_log_pi =self .actor .sample (next_global_obs )
            next_mask =self ._action_mask_from_state (s_ )
            next_action =next_action *next_mask 
            q1_t ,q2_t =self .t_critic (next_global_obs ,next_action )
            q_target =torch .min (q1_t ,q2_t ).squeeze (-1 )-alpha *next_log_pi 
            y =reward_joint +self .gamma *(1.0 -done_joint )*q_target 

        q1 ,q2 =self .critic (global_obs ,joint_action )
        critic_loss =F .mse_loss (q1 .squeeze (-1 ),y )+F .mse_loss (q2 .squeeze (-1 ),y )

        self .opt_c .zero_grad ()
        critic_loss .backward ()
        critic_grad_norm =torch .nn .utils .clip_grad_norm_ (self .critic .parameters (),GRAD_CLIP_MAX )
        critic_gn =float (critic_grad_norm )
        if math .isfinite (critic_gn ):
            self .opt_c .step ()

            
        new_action ,log_pi =self .actor .sample (global_obs )
        cur_mask =self ._action_mask_from_state (s )
        new_action =new_action *cur_mask 
        q1_new ,q2_new =self .critic (global_obs ,new_action )
        actor_loss =(alpha *log_pi -torch .min (q1_new ,q2_new ).squeeze (-1 )).mean ()

        self .opt_a .zero_grad ()
        actor_loss .backward ()
        actor_grad_norm =torch .nn .utils .clip_grad_norm_ (self .actor .parameters (),GRAD_CLIP_MAX )
        actor_gn =float (actor_grad_norm )
        if math .isfinite (actor_gn ):
            self .opt_a .step ()

            
        alpha_loss =-(self .log_alpha *(log_pi .detach ()+self .target_entropy )).mean ()
        self .opt_alpha .zero_grad ()
        alpha_loss .backward ()
        self .opt_alpha .step ()

        
        self ._soft_update (self .t_critic ,self .critic )

        
        q_mean =float (q1 .detach ().mean ().item ())if torch .isfinite (q1 ).all ()else 0.0 
        critic_loss_v =float (critic_loss .item ())if torch .isfinite (critic_loss )else 0.0 
        actor_loss_v =float (actor_loss .item ())if torch .isfinite (actor_loss )else 0.0 
        alpha_loss_v =float (alpha_loss .item ())if torch .isfinite (alpha_loss )else 0.0 
        entropy_v =float (-log_pi .detach ().mean ().item ())
        critic_after =min (critic_gn ,GRAD_CLIP_MAX )if math .isfinite (critic_gn )else 0.0 
        actor_after =min (actor_gn ,GRAD_CLIP_MAX )if math .isfinite (actor_gn )else 0.0 

        self .last_central_q_value =q_mean 
        self .last_central_critic_loss =critic_loss_v 
        self .last_central_actor_loss =actor_loss_v 
        self .last_central_critic_grad_norm_before_clip =critic_gn if math .isfinite (critic_gn )else 0.0 
        self .last_central_actor_grad_norm_before_clip =actor_gn if math .isfinite (actor_gn )else 0.0 
        self .last_central_critic_grad_norm =critic_after 
        self .last_central_actor_grad_norm =actor_after 
        self .last_central_critic_clip_count =int (critic_gn >GRAD_CLIP_MAX )
        self .last_central_actor_clip_count =int (actor_gn >GRAD_CLIP_MAX )
        self .last_sac_alpha =float (self .log_alpha .exp ().item ())
        self .last_sac_entropy =entropy_v 
        self .last_sac_alpha_loss =alpha_loss_v 

        
        self .last_actor_loss =actor_loss_v 
        self .last_critic_loss =critic_loss_v 
        self .last_actor_grad_norm =actor_after 
        self .last_local_critic_grad_norm =critic_after 
        self .last_global_critic_grad_norm =critic_after 
        self .last_global_critic_grad_norm_before_clip =self .last_central_critic_grad_norm_before_clip 
        self .last_global_critic_loss =critic_loss_v 
        self .last_global_critic_clip_count =self .last_central_critic_clip_count 

        self .update_step +=1 

    def _soft_update (self ,target ,source ):
        for tp ,p in zip (target .parameters (),source .parameters ()):
            tp .data .copy_ (tp .data *(1.0 -self .tau )+p .data *self .tau )

            
            
            

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

        
        
        

    def episode_start (self ):
        if not self .test_mode :
            self .current_episode +=1 

    def episode_end (self ):
        pass 

        
        
        

    def set_test_mode (self ,mode :bool ):
        self .test_mode =bool (mode )
        self .training =not self .test_mode 
        modules =[self .actor ,self .critic ,self .t_critic ]
        for m in modules :
            m .eval ()if self .test_mode else m .train ()

            
            
            

    def save_models (self ,path ,episode ):
        os .makedirs (path ,exist_ok =True )
        torch .save (self .actor .state_dict (),os .path .join (path ,f"sac_actor_ep{episode}.pth"))
        torch .save (self .critic .state_dict (),os .path .join (path ,f"sac_critic_ep{episode}.pth"))
        torch .save (
        {"log_alpha":self .log_alpha .detach ().cpu ()},
        os .path .join (path ,f"sac_alpha_ep{episode}.pth"),
        )

    def load_models (self ,path ,episode ,map_location =None ):
        ml =map_location if map_location is not None else device 
        self .actor .load_state_dict (
        torch .load (os .path .join (path ,f"sac_actor_ep{episode}.pth"),map_location =ml )
        )
        self .critic .load_state_dict (
        torch .load (os .path .join (path ,f"sac_critic_ep{episode}.pth"),map_location =ml )
        )
        alpha_ckpt =os .path .join (path ,f"sac_alpha_ep{episode}.pth")
        if os .path .isfile (alpha_ckpt ):
            data =torch .load (alpha_ckpt ,map_location =ml )
            with torch .no_grad ():
                self .log_alpha .copy_ (data ["log_alpha"].to (device ))
        self .t_critic .load_state_dict (self .critic .state_dict ())

    def save_actors (self ,path ,episode ):
        os .makedirs (path ,exist_ok =True )
        torch .save (self .actor .state_dict (),os .path .join (path ,f"sac_actor_ep{episode}.pth"))

    def load_actors (self ,path ,episode ,map_location =None ):
        ml =map_location if map_location is not None else device 
        self .actor .load_state_dict (
        torch .load (os .path .join (path ,f"sac_actor_ep{episode}.pth"),map_location =ml )
        )

        
        
        

    def debug_actor_output (self ,state ,station_idx =0 ):
        if not isinstance (state ,torch .Tensor ):
            state =torch .as_tensor (state ,dtype =torch .float32 ,device =device )
        else :
            state =state .to (device )

        global_obs =self ._reshape_global_obs (state )
        with torch .no_grad ():
            actions =self .actor (global_obs )
        if actions .dim ()==3 :
            return actions [0 ,station_idx ].detach ().cpu ().tolist ()
        return actions [station_idx ].detach ().cpu ().tolist ()
