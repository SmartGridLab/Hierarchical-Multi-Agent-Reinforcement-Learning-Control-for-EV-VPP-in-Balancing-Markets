"""


- LocalEvMLPCritic:

- GlobalMLPCritic:

"""
import torch
import torch .nn as nn
from Config import (
LOCAL_CRITIC_HIDDEN_SIZE ,
GLOBAL_CRITIC_HIDDEN_SIZE ,
MAX_EV_PER_STATION ,
MIXER_B_MAX ,
MIXER_B_MAX_ENABLE ,
)
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_DEMAND_STEPS ,
LOCAL_TAIL_DIM ,
GLOBAL_DEMAND_STEPS ,
)


class LocalEvMLPCritic (nn .Module ):
    def __init__ (self ,ev_feat_dim ,a_dim ,max_evs ,hid =LOCAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        super ().__init__ ()
        self .ev_feat_dim =ev_feat_dim 
        self .a_dim =a_dim 
        self .max_evs =max_evs 
        self .hid =hid 
        self .station_state_dim =station_state_dim if station_state_dim is not None else (ev_feat_dim *max_evs +LOCAL_TAIL_DIM )
        self .init_gain =init_gain 

        additional_features =1 +1 +int (LOCAL_DEMAND_STEPS )
        set_hid =max (hid //2 ,32 )

        
        self .token_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim +1 ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )

        
        self .q_head =nn .Sequential (
        nn .Linear (set_hid +additional_features ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,1 ),
        )

        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s_flat ,a_flat ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ):
        if s_flat .dim ()==1 :
            s_flat =s_flat .unsqueeze (0 )
        if a_flat .dim ()==1 :
            a_flat =a_flat .unsqueeze (0 )

        B =s_flat .size (0 )

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for LocalEvMLPCritic")
        if actual_station_powers .dim ()>1 :
            total_ev_power =actual_station_powers .sum (dim =1 ,keepdim =True )
        else :
            total_ev_power =actual_station_powers .unsqueeze (-1 )

            
        ev_dim =self .max_evs *self .ev_feat_dim 
        ev_flat =s_flat [:,:ev_dim ]
        ev_tokens =ev_flat .view (B ,self .max_evs ,self .ev_feat_dim )
        presence =(ev_tokens [...,0 :1 ]>0.5 ).float ()

        if a_flat .size (1 )!=self .max_evs :
            raise ValueError (f"Invalid local action shape: {tuple(a_flat.shape)} expected second dim {self.max_evs}")
        a_tokens =torch .clamp (a_flat ,-1.0 ,1.0 ).unsqueeze (-1 )

        token_input =torch .cat ([ev_tokens ,a_tokens ],dim =-1 )
        token_feat =self .token_encoder (token_input )

        denom =presence .sum (dim =1 ,keepdim =False ).clamp (min =1.0 )
        pooled =(token_feat *presence ).sum (dim =1 )/denom 

        L =int (LOCAL_DEMAND_STEPS )
        tail =L +1 
        if s_flat .size (1 )>=(self .max_evs *self .ev_feat_dim +tail ):
            current_step =s_flat [:,-1 :]
            ag_lookahead =s_flat [:,-tail :-1 ]
        else :
            current_step =s_flat .new_zeros (B ,1 )
            ag_lookahead =s_flat .new_zeros (B ,L )if L >0 else s_flat .new_zeros (B ,0 )

        scaled_total_ev_power =torch .clamp (total_ev_power ,-1.0 ,1.0 )
        feat =torch .cat ([pooled ,scaled_total_ev_power ,current_step ,ag_lookahead ],dim =1 )
        out =self .q_head (feat )
        out =torch .nan_to_num_ (out ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        return out 


class GlobalMLPCritic (nn .Module ):
    def __init__ (self ,s_dim ,a_dim ,n_agent ,hid =GLOBAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        super ().__init__ ()
        self .s_dim =s_dim 
        self .a_dim =a_dim 
        self .n_agent =n_agent 
        self .hid =hid 
        self .ev_features_per_station =EV_FEAT_DIM *MAX_EV_PER_STATION
        self .station_state_dim =self .ev_features_per_station 
        self .init_gain =init_gain 

        self .station_emb_dim =hid //2 
        self .per_station_sa =nn .Sequential (
        nn .Linear (self .ev_features_per_station +a_dim ,self .station_emb_dim ),
        nn .LayerNorm (self .station_emb_dim ),
        nn .LeakyReLU (0.1 ),
        )

        self .per_agent_head =nn .Sequential (
        nn .Linear (self .station_emb_dim ,self .station_emb_dim //2 ),
        nn .LayerNorm (self .station_emb_dim //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (self .station_emb_dim //2 ,1 ),
        )

        
        
        
        
        
        common_in_dim =1 +1 +int (GLOBAL_DEMAND_STEPS )+n_agent 
        self .mixer_w =nn .Sequential (
        nn .Linear (common_in_dim ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,n_agent ),
        )
        self .mixer_b =nn .Sequential (
        nn .Linear (common_in_dim ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,1 ),
        )
        self .softplus =nn .Softplus ()

        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s ,a ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ,attn_per_head =False ):
        if s .dim ()==1 :
            s =s .unsqueeze (0 )
        B =s .size (0 )

        ev_features_total =self .n_agent *self .ev_features_per_station 
        ev_features_flat =s [:,:ev_features_total ]

        common_info_start =ev_features_total 
        total_ev_power =s [:,common_info_start :common_info_start +1 ]
        current_step =s [:,common_info_start +1 :common_info_start +2 ]
        L =int (GLOBAL_DEMAND_STEPS )
        ag_lookahead =s [:,common_info_start +2 :common_info_start +2 +L ]if L >0 else s .new_zeros (B ,0 )

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for GlobalMLPCritic")
        station_power_vec =torch .clamp (actual_station_powers ,-1.0 ,1.0 )
        if station_power_vec .dim ()==1 :
            station_power_vec =station_power_vec .unsqueeze (0 )
        if station_power_vec .dim ()!=2 or station_power_vec .size (1 )!=self .n_agent :
            raise ValueError (
            f"Invalid actual_station_powers shape: {tuple(station_power_vec.shape)} "
            f"expected (B, {self.n_agent})"
            )

        if a is None :
            raise ValueError ("action tensor 'a' must be provided for GlobalMLPCritic")
        actions_vec =torch .clamp (a ,-1.0 ,1.0 )
        if actions_vec .dim ()!=3 or actions_vec .size (1 )!=self .n_agent or actions_vec .size (2 )!=self .a_dim :
            raise ValueError (f"Invalid action shape: {tuple(actions_vec.shape)} expected (B, n, {self.a_dim})")

        ev_per_station =ev_features_flat .reshape (B ,self .n_agent ,self .ev_features_per_station )
        station_input =torch .cat ([ev_per_station ,actions_vec ],dim =2 )
        station_feats =self .per_station_sa (
        station_input .reshape (B *self .n_agent ,-1 )
        ).reshape (B ,self .n_agent ,self .station_emb_dim )

        u_tokens =self .per_agent_head (station_feats )
        u =u_tokens .squeeze (-1 )
        if key_padding_mask is not None :
            u =u .masked_fill (key_padding_mask ,0.0 )

        common_features =torch .cat ([total_ev_power ,current_step ,ag_lookahead ,station_power_vec ],dim =1 )

        w =self .mixer_w (common_features )
        w =self .softplus (w )
        b_raw =self .mixer_b (common_features )
        # Scaled tanh: b = K * tanh(b_raw / K). Same ±K output range as the
        # naive form but ∂b/∂b_raw ≈ 1 near zero (no gradient amplification).
        if MIXER_B_MAX_ENABLE :
            b =MIXER_B_MAX *torch .tanh (b_raw /MIXER_B_MAX )
        else :
            b =b_raw

        q_global =(w *u ).mean (dim =1 ,keepdim =True )+b

        if not torch .isfinite (q_global ).all ():
            q_global =torch .nan_to_num (q_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        if not torch .isfinite (u ).all ():
            u =torch .nan_to_num (u ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        if return_attn :
            return q_global ,u ,None 

        return q_global ,u 
