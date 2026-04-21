
"Documentation."

import os 
import time 
import torch ._dynamo 
torch ._dynamo .config .suppress_errors =True 
import copy 
import torch 
import torch .nn .functional as F 
import torch .nn as nn 
import torch .optim as optim 

from Config import (
BATCH_SIZE ,GAMMA ,TAU ,
ACTOR_HIDDEN_SIZE ,LOCAL_CRITIC_HIDDEN_SIZE ,
LR_ACTOR ,LR_CRITIC_LOCAL ,
RANDOM_ACTION_RANGE ,
SMOOTHL1_BETA ,
EPSILON_START_EPISODE ,EPSILON_END_EPISODE ,EPSILON_INITIAL ,EPSILON_FINAL ,
OU_NOISE_START_EPISODE ,OU_NOISE_END_EPISODE ,OU_NOISE_SCALE_INITIAL ,OU_NOISE_SCALE_FINAL ,
GRAD_CLIP_MAX ,
GLOBAL_REWARD_WEIGHT ,
MEMORY_SIZE ,
MAX_EV_PER_STATION ,
)
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_USE_STATION_POWER ,
LOCAL_DEMAND_STEPS ,
LOCAL_USE_STEP ,
)

from training.Agent.actor import Actor 
from training.Agent.critic import LocalEvMLPCritic 
from training.Agent.replay_buffer import ReplayBuffer 
from training.Agent.noise import OUNoise ,linear_epsilon_decay 

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


class IndependentDDPG :
    "Documentation."

    def __init__ (self ,s_dim ,max_evs_per_station ,n_agent ,
    gamma =GAMMA ,tau =TAU ,batch =BATCH_SIZE ,
    lr_a =LR_ACTOR ,lr_c =LR_CRITIC_LOCAL ,
    num_episodes =10000 ,
    smoothl1_beta =SMOOTHL1_BETA ):
        "Documentation."
        if max_evs_per_station !=MAX_EV_PER_STATION :
            raise AssertionError (
            f"max_evs_per_station={max_evs_per_station} != Config.MAX_EV_PER_STATION={MAX_EV_PER_STATION}. "
            "LocalEvMLPCritic and normalize.py use Config.MAX_EV_PER_STATION directly; "
            "passing a different value causes silent shape mismatches."
            )
        self .s_dim ,self .a_dim ,self .n =s_dim ,max_evs_per_station ,n_agent
        self .max_ev_per_station =max_evs_per_station
        self .gamma ,self .tau ,self .batch =gamma ,tau ,batch 
        self .lr_a =lr_a 
        self .lr_c =lr_c 
        self .total_episodes =num_episodes 
        self .current_episode =0 

        self .hidden_size =ACTOR_HIDDEN_SIZE 

        
        self .actor_norms =[0 ]*n_agent 
        self .critic_norms =[0 ]*n_agent 

        
        self .actor_losses =[]
        self .critic_losses =[]

        
        self .last_actor_loss =0.0 
        self .last_critic_loss =0.0 
        self .last_actor_grad_norm =0.0 
        self .last_local_critic_grad_norm =0.0 

        
        self .last_actor_clip_count =0 
        self .last_local_critic_clip_count =0 
        self .local_critic_clip_counts =[0 ]*n_agent 
        self .actor_clip_counts =[0 ]*n_agent 

        
        self .critic_norms_before_clip =[0 ]*n_agent 
        self .actor_norms_before_clip =[0 ]*n_agent 

        
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

        
        self .ou_noise =OUNoise (n_agent ,max_evs_per_station )

        
        self .test_mode =False 

        
        self .buf =ReplayBuffer (cap =int (MEMORY_SIZE ))
        self .buf .maddpg_ref =self 
        self .max_evs =max_evs_per_station 

        
        self .active_evs =[0 ]*n_agent 

        
        
        self .ev_state_dim =EV_FEAT_DIM 
        self .max_evs =max_evs_per_station 
        
        self .local_tail_dim =(
        (1 if LOCAL_USE_STATION_POWER else 0 )
        +int (LOCAL_DEMAND_STEPS )
        +(1 if LOCAL_USE_STEP else 0 )
        )
        self .station_state_dim =self .ev_state_dim *self .max_evs +self .local_tail_dim 

        
        self .id_dim =n_agent 
        
        
        self .station_state_dim_with_id =n_agent +s_dim 

        
        
        self .actors =[Actor (s_dim +n_agent ,max_evs_per_station ,
        station_state_dim =self .station_state_dim_with_id ).to (device )
        for _ in range (n_agent )]
        self .t_actors =[copy .deepcopy (ac )for ac in self .actors ]

        
        self .critics =[
        LocalEvMLPCritic (
        ev_feat_dim =EV_FEAT_DIM ,
        a_dim =max_evs_per_station ,
        max_evs =max_evs_per_station ,
        hid =LOCAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self .station_state_dim_with_id ,
        ).to (device )
        for _ in range (n_agent )
        ]

        self .t_critics =[copy .deepcopy (cr )for cr in self .critics ]

        
        
        try :
            from Config import USE_TORCH_COMPILE 
            use_compile =bool (USE_TORCH_COMPILE )
        except Exception :
            use_compile =False 
        if os .name =="nt"and not use_compile :
            use_compile =False 

        if use_compile :
            t0 =time .perf_counter ()
            try :
                compile_options ={
                "mode":"default",
                "dynamic":False ,
                "fullgraph":False ,
                }
                for net_list in (self .actors ,self .t_actors ,self .critics ,self .t_critics ):
                    for i ,net in enumerate (net_list ):
                        net_list [i ]=torch .compile (net ,**compile_options )
                dt =time .perf_counter ()-t0 
                print (f"[Info] torch.compile enabled (IndependentDDPG). compile_time={dt:.1f}s")
            except Exception as e :
                dt =time .perf_counter ()-t0 
                print (f"[Warn] torch.compile failed/disabled (IndependentDDPG). elapsed={dt:.1f}s err={e}")

                
        self .opt_a =[optim .Adam (self .actors [i ].parameters (),lr =lr_a )for i in range (n_agent )]
        self .opt_c =[optim .Adam (self .critics [i ].parameters (),lr =lr_c )for i in range (n_agent )]

        
        self .loss_fn =nn .SmoothL1Loss (beta =smoothl1_beta )

        
        self .episode_local_q_values =[]

        
        self .update_step =0 

        
        self .global_reward_weight =GLOBAL_REWARD_WEIGHT 

    def update_active_evs (self ,env ):
        "Documentation."
        num_stations =min (self .n ,env .num_stations )
        self .active_evs =[0 ]*self .n 

        for i in range (num_stations ):
            active_count =env .ev_mask [i ].sum ().item ()
            self .active_evs [i ]=active_count 

        self .env =env 

    def _with_id (self ,state_batch ,agent_idx ):
        "Documentation."
        B =state_batch .size (0 )
        oh =torch .zeros (B ,self .id_dim ,device =state_batch .device ,dtype =state_batch .dtype )
        oh [:,agent_idx ]=1.0 
        return torch .cat ([oh ,state_batch ],dim =-1 )

    def act (self ,state ,env =None ,noise =True ):
        "Documentation."
        
        if hasattr (self ,'test_mode')and self .test_mode :
            noise =False 

            
        if env is not None :
            self .update_active_evs (env )

            
        active_agents =[i for i ,num in enumerate (self .active_evs )if num >0 ]

        
        tensor_actions =torch .zeros ((self .n ,self .max_evs ),dtype =torch .float32 ,device =device )

        with torch .no_grad ():
            for i ,agent_idx in enumerate (active_agents ):
                num_active =self .active_evs [agent_idx ]
                
                agent_state =self ._with_id (state [agent_idx :agent_idx +1 ],agent_idx )

                
                a =self .actors [agent_idx ](agent_state )

                
                if noise and hasattr (self ,'ou_noise')and getattr (self ,'ou_noise_scale',0.0 )>0.0 :
                    try :
                        if 'ou_step_noise'not in locals ():
                            ou_step_noise =self .ou_noise .sample (self .active_evs )
                        from Config import OU_NOISE_GAIN 
                        a [0 ,:num_active ]=a [0 ,:num_active ]+(OU_NOISE_GAIN *self .ou_noise_scale )*ou_step_noise [agent_idx ,:num_active ]
                    except Exception :
                        pass 

                        
                a =torch .clamp (a ,-1.0 ,1.0 )
                tensor_actions [agent_idx ,:num_active ]=a .squeeze (0 )[:num_active ]

        return tensor_actions 

    def update (self ):
        "Documentation."
        
        if hasattr (self ,'test_mode')and self .test_mode :
            self .last_actor_loss =0.0 
            self .last_critic_loss =0.0 
            self .last_actor_grad_norm =0.0 
            self .last_local_critic_grad_norm =0.0 
            return 

            
        if self .buf .size <self .batch :
            return 

            
        s ,s_ ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes =self .buf .sample (self .batch )

        
        

        
        total_critic_loss =0.0 
        total_actor_loss =0.0 

        
        prio_accum =torch .zeros ((s .size (0 ),),device =s .device )

        for i in range (self .n ):
        
            s_i =self ._with_id (s [:,i ,:],i )
            a_i =a [:,i ,:]
            
            r_i =r_local [:,i ]+self .global_reward_weight *r_global .squeeze (-1 )
            s_i_ =self ._with_id (s_ [:,i ,:],i )
            d_i =d [:,i ]

            
            with torch .no_grad ():
            
                num_active_i =self .active_evs [i ]if i <len (self .active_evs )else self .max_evs 
                a_i_ =self .t_actors [i ](s_i_ )

                
                q_target =self .t_critics [i ](s_i_ ,a_i_ ,actual_station_powers =actual_station_powers [:,i ])
                y_i =r_i +self .gamma *(1.0 -d_i )*q_target .squeeze (-1 )

                
            q_current =self .critics [i ](s_i ,a_i ,actual_station_powers =actual_station_powers [:,i ]).squeeze (-1 )

            
            td_i =(q_current -y_i ).detach ().abs ()
            prio_accum +=td_i 

            critic_loss_i =F .smooth_l1_loss (q_current ,y_i ,beta =SMOOTHL1_BETA )

            
            self .opt_c [i ].zero_grad ()
            critic_loss_i .backward ()

            
            critic_grad_norm =torch .nn .utils .clip_grad_norm_ (self .critics [i ].parameters (),GRAD_CLIP_MAX )
            self .critic_norms [i ]=float (critic_grad_norm )

            self .opt_c [i ].step ()

            total_critic_loss +=critic_loss_i .item ()

            
            
            a_i_pred =self .actors [i ](s_i )
            q_i_pred =self .critics [i ](s_i ,a_i_pred ,actual_station_powers =actual_station_powers [:,i ])
            actor_loss_i =-q_i_pred .mean ()

            
            self .opt_a [i ].zero_grad ()
            actor_loss_i .backward ()

            
            actor_grad_norm =torch .nn .utils .clip_grad_norm_ (self .actors [i ].parameters (),GRAD_CLIP_MAX )
            self .actor_norms [i ]=float (actor_grad_norm )

            self .opt_a [i ].step ()

            total_actor_loss +=actor_loss_i .item ()

            

            
        for i in range (self .n ):
            self ._soft_update (self .t_actors [i ],self .actors [i ])
            self ._soft_update (self .t_critics [i ],self .critics [i ])

            
        self .last_critic_loss =total_critic_loss /self .n 
        self .last_actor_loss =total_actor_loss /self .n 
        self .last_local_critic_grad_norm =sum (self .critic_norms )/self .n 
        self .last_actor_grad_norm =sum (self .actor_norms )/self .n 

        self .update_step +=1 

    def _soft_update (self ,target ,source ):
        "Documentation."
        for target_param ,param in zip (target .parameters (),source .parameters ()):
            target_param .data .copy_ (
            target_param .data *(1.0 -self .tau )+param .data *self .tau 
            )

    def cache_experience (self ,s ,s_ ,delta_soc ,r_local ,r_global ,done ,actual_station_powers ):
        "Documentation."
        self .buf .cache (s ,s_ ,delta_soc ,r_local ,r_global ,done ,actual_station_powers )

    def set_test_mode (self ,mode :bool ):
        "Documentation."
        self .test_mode =bool (mode )

    def episode_start (self ):
        "Documentation."
        self .episode_local_q_values =[]
        
        if not self .test_mode :
            self .current_episode +=1 

            
        self .epsilon =linear_epsilon_decay (
        self .current_episode ,
        self .epsilon_start_episode ,self .epsilon_end_episode ,
        self .epsilon_initial ,self .epsilon_final ,
        )

        
        if self .current_episode <self .ou_noise_start_episode :
            self .ou_noise_scale =0.0 
        elif self .current_episode >=self .ou_noise_end_episode :
            self .ou_noise_scale =self .ou_noise_scale_final 
        else :
            progress =(self .current_episode -self .ou_noise_start_episode )/max (1 ,(self .ou_noise_end_episode -self .ou_noise_start_episode ))
            self .ou_noise_scale =self .ou_noise_scale_initial +(self .ou_noise_scale_final -self .ou_noise_scale_initial )*progress 

            
        if hasattr (self ,'ou_noise'):
            try :
                self .ou_noise .reset ()
            except Exception :
                pass 

    def episode_end (self ):
        "Documentation."
        pass 

    def save_models (self ,path ,episode ):
        "Documentation."
        import os 
        os .makedirs (path ,exist_ok =True )

        for i in range (self .n ):
            torch .save (self .actors [i ].state_dict (),
            os .path .join (path ,f"actor_{i}_ep{episode}.pth"))
            torch .save (self .critics [i ].state_dict (),
            os .path .join (path ,f"critic_{i}_ep{episode}.pth"))

    def load_models (self ,path ,episode ):
        "Documentation."
        for i in range (self .n ):
            self .actors [i ].load_state_dict (
            torch .load (os .path .join (path ,f"actor_{i}_ep{episode}.pth")))
            self .critics [i ].load_state_dict (
            torch .load (os .path .join (path ,f"critic_{i}_ep{episode}.pth")))

            
            self .t_actors [i ].load_state_dict (self .actors [i ].state_dict ())
            self .t_critics [i ].load_state_dict (self .critics [i ].state_dict ())

    def save_actors (self ,path ,episode ):
        "Documentation."
        import os ,torch 
        os .makedirs (path ,exist_ok =True )
        for i in range (self .n ):
            torch .save (self .actors [i ].state_dict (),
            os .path .join (path ,f"actor_{i}_ep{episode}.pth"))

    def load_actors (self ,path ,episode ,map_location =None ):
        "Documentation."
        import os ,torch 
        for i in range (self .n ):
            sd =torch .load (os .path .join (path ,f"actor_{i}_ep{episode}.pth"),
            map_location =map_location )
            self .actors [i ].load_state_dict (sd )
            if i <len (self .t_actors ):
                self .t_actors [i ].load_state_dict (self .actors [i ].state_dict ())
