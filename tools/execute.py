
import argparse 
import csv 
import glob 
import json 
import os 
import re 
import subprocess 
import sys 
import time 
from datetime import datetime 
from pathlib import Path 

import matplotlib .pyplot as plt 
import numpy as np 
import pandas as pd 
import torch 

from training.Agent import MADDPG
from training.benchmark_agents.independent_ddpg import IndependentDDPG
from training.benchmark_agents.shared_obs_ddpg import SharedObsDDPG
from training.benchmark_agents.shared_obs_sac import SharedObsSAC 
from Config import USE_INDEPENDENT_DDPG as CFG_USE_IDDPG 
from Config import USE_SHARED_OBS_DDPG as CFG_USE_SHARED_OBS_DDPG 
from Config import USE_SHARED_OBS_SAC as CFG_USE_SHARED_OBS_SAC 
from Config import NUM_STATIONS ,NUM_EVS ,EPISODE_STEPS 
from Config import LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC 
from Config import BATCH_SIZE ,GAMMA ,TAU ,TAU_GLOBAL 
from Config import TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,TD3_SIGMA_LOCAL ,TD3_CLIP_LOCAL 
from Config import MAX_EV_POWER_KW ,POWER_TO_ENERGY ,ENV_SEED 
from Config import (
SHARED_OBS_LR_ACTOR ,
SHARED_OBS_LR_CRITIC_LOCAL ,
SHARED_OBS_GLOBAL_REWARD_WEIGHT ,
SHARED_OBS_EPSILON_END_EPISODE ,
SHARED_OBS_OU_NOISE_END_EPISODE ,
SHARED_OBS_OU_NOISE_SCALE_FINAL ,
SAC_LR_ACTOR ,SAC_LR_CRITIC ,SAC_LR_ALPHA ,
SAC_TAU ,SAC_ALPHA_INIT ,SAC_TARGET_ENTROPY_SCALE ,
SAC_GLOBAL_REWARD_WEIGHT ,
)
from environment.EVEnv import EVEnv 
from tools.evaluator import set_env_seed 
from tools.Utils import (
plot_arrival_counts ,
plot_daily_rewards ,
plot_performance_metrics ,
plot_power_mismatch_analysis ,
plot_reward_breakdown ,
plot_station_cooperation_full ,
)
from environment.normalize import normalize_observation 

DEMAND_TARGET_MIN =-200.0 
DEMAND_TARGET_MAX =800.0 
ACTOR_PATTERNS =(
re .compile (r"actor_\d+_ep(\d+)\.pth$"),
re .compile (r"shared_actor_ep(\d+)\.pth$"),
)


def extract_actor_episode (filename ):
    for pattern in ACTOR_PATTERNS :
        match =pattern .fullmatch (filename )
        if match is not None :
            return int (match .group (1 ))
    return None 


def resolve_path (project_root ,path_value ):
    if os .path .isabs (path_value ):
        return path_value 
    return os .path .abspath (os .path .join (project_root ,path_value ))


def run_splitter (split_script_path ,xlsx_path ,out_dir ):
    cmd =[sys .executable ,split_script_path ,"--xlsx",xlsx_path ,"--out-dir",out_dir ]
    subprocess .run (cmd ,check =True )


def extract_day_label (csv_path ):
    base =os .path .basename (csv_path )
    m =re .search (r"day_(\d{4}-\d{2}-\d{2})\.csv$",base )
    if m is None :
        return base 
    return m .group (1 )


def infer_year_month_from_xlsx (xlsx_path ):
    base =os .path .basename (xlsx_path )
    m =re .search (r"(\d{2})\s+(\d{4})\.xlsx$",base )
    if m is None :
        return None 
    month =m .group (1 )
    year =m .group (2 )
    return f"{year}-{month}"


def load_scaled_episode (csv_path ):
    df =pd .read_csv (csv_path )
    if "demand_adjustment"in df .columns :
        series =pd .to_numeric (df ["demand_adjustment"],errors ="coerce").fillna (0.0 ).to_numpy (float )
    else :
        numeric_cols =df .select_dtypes (include =["number"]).columns .tolist ()
        if len (numeric_cols )==0 :
            series =np .zeros (EPISODE_STEPS ,dtype =float )
        else :
            series =pd .to_numeric (df [numeric_cols [0 ]],errors ="coerce").fillna (0.0 ).to_numpy (float )
    if series .size ==0 :
        series =np .zeros (EPISODE_STEPS ,dtype =float )
    data =np .zeros (EPISODE_STEPS ,dtype =float )
    usable =series [:EPISODE_STEPS ].astype (float )
    data [:usable .size ]=usable 
    a =float (np .min (data ))
    b =float (np .max (data ))
    if np .isclose (a ,b ):
        return np .full_like (data ,(DEMAND_TARGET_MIN +DEMAND_TARGET_MAX )/2.0 )
    scale =(DEMAND_TARGET_MAX -DEMAND_TARGET_MIN )/(b -a )
    offset =DEMAND_TARGET_MIN /scale -a 
    return scale *(data +offset )



def find_model_path_and_episode (base_dir ,requested_episode =None ):
    search_dirs =[base_dir ]
    results_dir =os .path .join (base_dir ,"results")
    if os .path .isdir (results_dir ):
        for sub in os .listdir (results_dir ):
            if sub .startswith ("TEST"):
                search_dirs .append (os .path .join (results_dir ,sub ))

    candidates =[]
    for directory in search_dirs :
        if not os .path .isdir (directory ):
            continue 
        for name in os .listdir (directory ):
            ep =extract_actor_episode (name )
            if ep is None :
                continue 
            if requested_episode is not None and ep !=int (requested_episode ):
                continue 
            actor_path =os .path .join (directory ,name )
            mtime =os .path .getmtime (actor_path )
            candidates .append ((ep ,mtime ,directory ))

    if len (candidates )==0 :
        raise FileNotFoundError (f"No actor files were found under: {base_dir}")

    if requested_episode is None :
        candidates .sort (key =lambda x :(x [0 ],x [1 ]),reverse =True )
    else :
        candidates .sort (key =lambda x :x [1 ],reverse =True )

    chosen =candidates [0 ]
    return chosen [2 ],int (chosen [0 ])


def build_agent (env ):
    state_dim =env ._get_obs ().shape [1 ]
    if CFG_USE_IDDPG :
        return IndependentDDPG (
        state_dim ,
        env .max_ev_per_station ,
        n_agent =env .num_stations ,
        num_episodes =1 ,
        batch =BATCH_SIZE ,
        gamma =GAMMA ,
        tau =TAU ,
        lr_a =LR_ACTOR ,
        lr_c =LR_CRITIC_LOCAL ,
        )
    if CFG_USE_SHARED_OBS_DDPG :
        return SharedObsDDPG (
        state_dim ,
        env .max_ev_per_station ,
        n_agent =env .num_stations ,
        num_episodes =1 ,
        batch =BATCH_SIZE ,
        gamma =GAMMA ,
        tau =TAU ,
        lr_a =SHARED_OBS_LR_ACTOR ,
        lr_c =SHARED_OBS_LR_CRITIC_LOCAL ,
        global_reward_weight =SHARED_OBS_GLOBAL_REWARD_WEIGHT ,
        epsilon_end_episode =SHARED_OBS_EPSILON_END_EPISODE ,
        ou_noise_end_episode =SHARED_OBS_OU_NOISE_END_EPISODE ,
        ou_noise_scale_final =SHARED_OBS_OU_NOISE_SCALE_FINAL ,
        )
    if CFG_USE_SHARED_OBS_SAC :
        return SharedObsSAC (
        state_dim ,
        env .max_ev_per_station ,
        n_agent =env .num_stations ,
        num_episodes =1 ,
        batch =BATCH_SIZE ,
        gamma =GAMMA ,
        tau =SAC_TAU ,
        lr_a =SAC_LR_ACTOR ,
        lr_c =SAC_LR_CRITIC ,
        lr_alpha =SAC_LR_ALPHA ,
        global_reward_weight =SAC_GLOBAL_REWARD_WEIGHT ,
        alpha_init =SAC_ALPHA_INIT ,
        target_entropy_scale =SAC_TARGET_ENTROPY_SCALE ,
        )
    return MADDPG (
    s_dim =state_dim ,
    max_evs_per_station =env .max_ev_per_station ,
    n_agent =env .num_stations ,
    num_episodes =1 ,
    batch =BATCH_SIZE ,
    gamma =GAMMA ,
    tau =TAU ,
    lr_a =LR_ACTOR ,
    lr_c =LR_CRITIC_LOCAL ,
    lr_global_c =LR_GLOBAL_CRITIC ,
    tau_global =TAU_GLOBAL ,
    td3_sigma =TD3_SIGMA_GLOBAL ,
    td3_clip =TD3_CLIP_GLOBAL ,
    td3_sigma_local =TD3_SIGMA_LOCAL ,
    td3_clip_local =TD3_CLIP_LOCAL ,
    smoothl1_beta =0.01 ,
    )


def ensure_tensor_actions (actions ,env ):
    if isinstance (actions ,torch .Tensor ):
        return actions .clone ()
    return torch .as_tensor (actions ,dtype =torch .float32 ,device =env .soc .device )


def apply_force_charging (actions ,env ,trigger_ratio ,slack_kwh ,ev_force_state =None ):
    "Documentation."
    if ev_force_state is None :
        ev_force_state ={}
    actions =ensure_tensor_actions (actions ,env )
    forced =0 
    forced_by_station =[0 ]*env .num_stations 
    for st in range (env .num_stations ):
        active_evs =torch .nonzero (env .ev_mask [st ],as_tuple =False ).squeeze (-1 )
        if active_evs .numel ()==0 :
            continue 
        sorted_active =env ._sort_active_evs (st ,active_evs )
        for order_idx in range (len (sorted_active )):
            ev_idx =int (sorted_active [order_idx ].item ())
            need_kwh =float ((env .target [st ,ev_idx ]-env .soc [st ,ev_idx ]).item ())
            
            ev_key =(st ,ev_idx ,int (env .depart [st ,ev_idx ].item ()))

            
            if need_kwh <=0 :
                ev_force_state .pop (ev_key ,None )
                continue 

            remaining_steps =int (env .depart [st ,ev_idx ].item ())-int (env .step_count )+1 
            remaining_steps =max (remaining_steps ,1 )
            max_possible_kwh =remaining_steps *MAX_EV_POWER_KW *POWER_TO_ENERGY 

            
            if ev_key not in ev_force_state :
                safe_possible_kwh =max_possible_kwh *trigger_ratio +slack_kwh 
                if need_kwh <safe_possible_kwh :
                    continue 
                ev_force_state [ev_key ]=True 

                
            current_action =float (actions [st ,order_idx ].item ())
            if current_action >=1.0 -1e-6 :
                continue 
            actions [st ,order_idx ]=1.0 
            forced +=1 
            forced_by_station [st ]+=1 
    return actions ,forced ,forced_by_station 


def normalize_int_list (values ,size ):
    out =[0 ]*size 
    if isinstance (values ,dict ):
        for idx in range (size ):
            out [idx ]=int (values .get (idx ,values .get (str (idx ),0 )))
        return out 
    if isinstance (values ,(list ,tuple ,np .ndarray )):
        for idx in range (min (size ,len (values ))):
            out [idx ]=int (values [idx ])
    return out 


def normalize_float_list (values ,size ):
    out =[0.0 ]*size 
    if isinstance (values ,dict ):
        for idx in range (size ):
            out [idx ]=float (values .get (idx ,values .get (str (idx ),0.0 )))
        return out 
    if isinstance (values ,(list ,tuple ,np .ndarray )):
        for idx in range (min (size ,len (values ))):
            out [idx ]=float (values [idx ])
    return out 


def build_episode_data ():
    episode_data ={
    "ag_requests":[],
    "total_ev_transport":[],
    "power_mismatch":[],
    "arrivals_per_step":[],
    "rewards_global_balance":[],
    "rewards_local_shaping":[],
    "rewards_local_departure":[],
    "rewards_local_discharge_penalty":[],
    "force_ev_step_overrides":[],
    "force_ev_step_overrides_by_station":[],
    }
    for station_idx in range (NUM_STATIONS ):
        episode_data [f"actual_ev{station_idx + 1}"]=[]
    return episode_data 


def build_step_row (step_count ,info ,forced_now ,forced_by_station ,local_reward_mean ):
    station_powers =normalize_float_list (info .get ("station_powers",[]),NUM_STATIONS )
    arrivals_by_station =normalize_int_list (info .get ("arrivals_by_station",[]),NUM_STATIONS )
    active_by_station =normalize_int_list (info .get ("active_evs",{}),NUM_STATIONS )
    reward_breakdown =info .get ("reward_breakdown",{})
    station_breakdown =reward_breakdown .get ("per_station",[])
    global_breakdown =reward_breakdown .get ("global",{})

    local_shaping_total =0.0 
    local_departure_total =0.0 
    local_discharge_penalty_total =0.0 
    local_switch_penalty_total =0.0 
    for item in station_breakdown :
        local_shaping_total +=float (item .get ("progress_shaping",0.0 ))
        local_departure_total +=float (item .get ("departure_reward",0.0 ))
        local_discharge_penalty_total +=float (item .get ("discharge_penalty",0.0 ))
        local_switch_penalty_total +=float (item .get ("switch_penalty",0.0 ))

    net_demand =float (info .get ("net_demand",0.0 ))
    total_ev_transport =float (info .get ("total_ev_transport",0.0 ))
    power_mismatch =net_demand -total_ev_transport 

    active_evs_total =int (sum (active_by_station ))
    force_ev_step_ratio_pct =0.0 
    if active_evs_total !=0 :
        force_ev_step_ratio_pct =float (forced_now )/float (active_evs_total )*100.0 

    row ={
    "step":int (step_count ),
    "net_demand_kw":net_demand ,
    "total_ev_transport_kw":total_ev_transport ,
    "power_mismatch_kw":power_mismatch ,
    "abs_power_mismatch_kw":abs (power_mismatch ),
    "global_reward":float (info .get ("global_reward",0.0 )),
    "global_balance_reward":float (global_breakdown .get ("balance_reward",info .get ("global_reward",0.0 ))),
    "local_reward_mean":float (local_reward_mean ),
    "local_shaping_total":local_shaping_total ,
    "local_departure_total":local_departure_total ,
    "local_discharge_penalty_total":local_discharge_penalty_total ,
    "local_switch_penalty_total":local_switch_penalty_total ,
    "arrivals_total":int (sum (arrivals_by_station )),
    "active_evs_total":active_evs_total ,
    "force_ev_step_overrides":int (forced_now ),
    "force_active_step":int (forced_now >0 ),
    "force_ev_step_ratio_pct":force_ev_step_ratio_pct ,
    }

    for station_idx in range (NUM_STATIONS ):
        station_num =station_idx +1 
        row [f"station_{station_num}_power_kw"]=float (station_powers [station_idx ])
        row [f"station_{station_num}_arrivals"]=int (arrivals_by_station [station_idx ])
        row [f"station_{station_num}_active_evs"]=int (active_by_station [station_idx ])
        row [f"station_{station_num}_force_ev_step_overrides"]=int (forced_by_station [station_idx ])

    return row 


def run_single_episode (agent ,day_label ,demand_series ,force_enabled ,trigger_ratio ,slack_kwh ):
    env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )
    env .reset (net_demand_series =demand_series )
    agent .episode_start ()
    agent .update_active_evs (env )

    ep_local =0.0 
    ep_global =0.0 
    forced_overrides =0 
    forced_steps =0 
    step_rows =[]
    episode_data =build_episode_data ()
    ev_force_state ={}

    while True :
        obs =env .begin_step ()
        agent .update_active_evs (env )
        obs =normalize_observation (obs )
        actions =agent .act (obs ,env =env ,noise =False )
        forced_now =0 
        forced_by_station =[0 ]*NUM_STATIONS 
        if force_enabled :
            actions ,forced_now ,forced_by_station =apply_force_charging (
            actions ,
            env ,
            trigger_ratio ,
            slack_kwh ,
            ev_force_state ,
            )
            forced_overrides +=int (forced_now )
            if int (forced_now )!=0 :
                forced_steps +=1 
        _ ,r_local ,r_global ,done ,info =env .apply_action (actions )
        local_reward_mean =float (np .mean (r_local ))
        ep_local +=local_reward_mean 
        ep_global +=float (r_global )

        step_count =int (info .get ("step_count",env .step_count ))
        step_row =build_step_row (step_count ,info ,forced_now ,forced_by_station ,local_reward_mean )
        step_rows .append (step_row )

        episode_data ["ag_requests"].append (step_row ["net_demand_kw"])
        episode_data ["total_ev_transport"].append (step_row ["total_ev_transport_kw"])
        episode_data ["power_mismatch"].append (step_row ["power_mismatch_kw"])
        episode_data ["arrivals_per_step"].append ([
        step_row [f"station_{station_idx + 1}_arrivals"]for station_idx in range (NUM_STATIONS )
        ])
        episode_data ["rewards_global_balance"].append (step_row ["global_balance_reward"])
        episode_data ["rewards_local_shaping"].append (step_row ["local_shaping_total"])
        episode_data ["rewards_local_departure"].append (step_row ["local_departure_total"])
        episode_data ["rewards_local_discharge_penalty"].append (
        step_row ["local_discharge_penalty_total"]+step_row ["local_switch_penalty_total"]
        )
        episode_data ["force_ev_step_overrides"].append (step_row ["force_ev_step_overrides"])
        episode_data ["force_ev_step_overrides_by_station"].append ([
        step_row [f"station_{station_idx + 1}_force_ev_step_overrides"]for station_idx in range (NUM_STATIONS )
        ])
        for station_idx in range (NUM_STATIONS ):
            episode_data [f"actual_ev{station_idx + 1}"].append (
            step_row [f"station_{station_idx + 1}_power_kw"]
            )

        if all (done ):
            break 

    agent .episode_end ()
    metrics =env .get_metrics ()
    step_count =int (env .step_count )
    local_avg =ep_local /max (step_count ,1 )
    global_avg =ep_global /max (step_count ,1 )

    surplus_steps =int (metrics .get ("surplus_steps",0 ))
    shortage_steps =int (metrics .get ("shortage_steps",0 ))
    surplus_within =int (metrics .get ("surplus_within_narrow",0 ))
    shortage_within =int (metrics .get ("shortage_within_narrow",0 ))
    dispatch_success =surplus_within +shortage_within 
    dispatch_total =surplus_steps +shortage_steps 
    dispatch_tracking =0.0 
    if dispatch_total !=0 :
        dispatch_tracking =dispatch_success /dispatch_total *100.0 

    step_df =pd .DataFrame (step_rows )
    controllable_ev_steps =0 
    if not step_df .empty and "active_evs_total"in step_df .columns :
        controllable_ev_steps =int (step_df ["active_evs_total"].sum ())
    forced_ev_step_ratio_pct =0.0 
    if controllable_ev_steps !=0 :
        forced_ev_step_ratio_pct =float (forced_overrides )/float (controllable_ev_steps )*100.0 

    row ={
    "day":day_label ,
    "local_avg_reward":local_avg ,
    "global_avg_reward":global_avg ,
    "soc_miss_rate":float (metrics .get ("soc_miss_rate",0.0 )),
    "soc_hit_rate":100.0 -float (metrics .get ("soc_miss_rate",0.0 )),
    "avg_soc_deficit":float (metrics .get ("avg_soc_deficit",0.0 )),
    "avg_switches":float (metrics .get ("avg_switches",0.0 )),
    "departing_evs":int (metrics .get ("departing_evs",0 )),
    "departing_evs_soc_met":int (metrics .get ("departing_evs_soc_met",0 )),
    "surplus_absorption_rate":float (metrics .get ("surplus_absorption_rate",0.0 )),
    "supply_cooperation_rate":float (metrics .get ("supply_cooperation_rate",0.0 )),
    "surplus_steps":surplus_steps ,
    "surplus_within_narrow":surplus_within ,
    "shortage_steps":shortage_steps ,
    "shortage_within_narrow":shortage_within ,
    "dispatch_success_steps":dispatch_success ,
    "dispatch_total_steps":dispatch_total ,
    "dispatch_tracking_rate":dispatch_tracking ,
    "controllable_ev_steps":int (controllable_ev_steps ),
    "forced_ev_step_overrides":int (forced_overrides ),
    "forced_ev_step_ratio_pct":float (forced_ev_step_ratio_pct ),
    "forced_active_steps":int (forced_steps ),
    "forced_overrides":int (forced_overrides ),
    "forced_steps":int (forced_steps ),
    "episode_steps":step_count ,
    }
    return row ,step_df ,episode_data 



def build_overall_summary (df ,repeats ,day_count ):
    depart_total =int (df ["departing_evs"].sum ())
    soc_met_total =int (df ["departing_evs_soc_met"].sum ())
    dispatch_success_total =int (df ["dispatch_success_steps"].sum ())
    dispatch_steps_total =int (df ["dispatch_total_steps"].sum ())
    controllable_ev_steps_total =int (df ["controllable_ev_steps"].sum ())

    soc_hit_weighted =0.0 
    if depart_total !=0 :
        soc_hit_weighted =soc_met_total /depart_total *100.0 

    dispatch_weighted =0.0 
    if dispatch_steps_total !=0 :
        dispatch_weighted =dispatch_success_total /dispatch_steps_total *100.0 

    forced_ev_step_ratio_pct =0.0 
    if controllable_ev_steps_total !=0 :
        forced_ev_step_ratio_pct =float (df ["forced_ev_step_overrides"].sum ())/float (controllable_ev_steps_total )*100.0 

    summary ={
    "episodes":int (len (df )),
    "repeats":int (repeats ),
    "days":int (day_count ),
    "local_avg_reward_mean":float (df ["local_avg_reward"].mean ()),
    "global_avg_reward_mean":float (df ["global_avg_reward"].mean ()),
    "soc_hit_rate_mean":float (df ["soc_hit_rate"].mean ()),
    "soc_hit_rate_weighted":float (soc_hit_weighted ),
    "dispatch_tracking_rate_mean":float (df ["dispatch_tracking_rate"].mean ()),
    "dispatch_tracking_rate_weighted":float (dispatch_weighted ),
    "departing_evs":depart_total ,
    "departing_evs_soc_met":soc_met_total ,
    "dispatch_success_steps":dispatch_success_total ,
    "dispatch_total_steps":dispatch_steps_total ,
    "controllable_ev_steps":int (controllable_ev_steps_total ),
    "forced_ev_step_overrides":int (df ["forced_ev_step_overrides"].sum ()),
    "forced_ev_step_ratio_pct":float (forced_ev_step_ratio_pct ),
    "forced_active_steps":int (df ["forced_active_steps"].sum ()),
    "forced_overrides":int (df ["forced_overrides"].sum ()),
    "forced_steps":int (df ["forced_steps"].sum ()),
    }
    return summary 


_MODEL_DIR_PATTERN =re .compile (r"model_\d{8}_\d{6}$")
_TEST_DIR_PATTERN =re .compile (r"TEST(\d+)$")


def find_latest_model_dir (archive_dir ):
    """Find the latest model_YYYYMMDD_HHMMSS directory by name (alphabetically = newest timestamp)."""
    archive_dir =Path (archive_dir )
    latest =None 
    for candidate in archive_dir .iterdir ():
        if not candidate .is_dir ()or not _MODEL_DIR_PATTERN .fullmatch (candidate .name ):
            continue 
        if not (candidate /"results").is_dir ():
            continue 
        if latest is None or candidate .name >latest .name :
            latest =candidate 
    if latest is None :
        raise FileNotFoundError (f"No archive/model_* directories with results found under: {archive_dir}")
    return latest 


def build_test_dir_map (results_dir ):
    """Return dict of {episode_int: path} for TEST* subdirs."""
    results_dir =Path (results_dir )
    test_dirs ={}
    for entry in results_dir .iterdir ():
        if not entry .is_dir ():
            continue 
        m =_TEST_DIR_PATTERN .fullmatch (entry .name )
        if m :
            test_dirs [int (m .group (1 ))]=entry 
    return test_dirs 


def load_test_history_episodes (results_dir ):
    """Read test_history.json, return list of episode ints (or None if unavailable)."""
    history_path =Path (results_dir )/"test_history.json"
    if not history_path .is_file ():
        return None 
    with history_path .open ("r",encoding ="utf-8")as fh :
        data =json .load (fh )
    episodes =data .get ("episodes")
    if not isinstance (episodes ,list ):
        return None 
    parsed =[]
    for value in episodes :
        try :
            parsed .append (int (value ))
        except (TypeError ,ValueError ):
            return None 
    return parsed 


def choose_best_episode (model_dir ):
    """Find best episode by SoC+Dispatch score from test_performance_metrics.csv.

    Returns the episode int for the best-scoring TEST* dir, or the highest
    episode number as a fallback.
    """
    results_dir =Path (model_dir )/"results"
    test_dirs =build_test_dir_map (results_dir )
    if not test_dirs :
        raise FileNotFoundError (f"No TEST* directories found under: {results_dir}")

    metrics_csv =results_dir /"test_performance_metrics.csv"
    history_episodes =load_test_history_episodes (results_dir )

    best_choice =None 
    if metrics_csv .is_file ():
        with metrics_csv .open ("r",encoding ="utf-8-sig",newline ="")as fh :
            rows =list (csv .DictReader (fh ))

        for idx ,row in enumerate (rows ):
            try :
                soc =float (row ["SoC_Hit_Rate_%"])
                dispatch =float (row ["Dispatch_Tracking_Rate_%"])
            except (KeyError ,TypeError ,ValueError ):
                continue 

            episode =None 
            if history_episodes is not None and idx <len (history_episodes ):
                episode =history_episodes [idx ]
            else :
                raw_episode =row .get ("Episode","")
                try :
                    csv_episode =int (float (raw_episode ))
                except (TypeError ,ValueError ):
                    csv_episode =None 

                if csv_episode is not None :
                    if csv_episode in test_dirs :
                        episode =csv_episode 
                    elif (csv_episode *10 )in test_dirs :
                        episode =csv_episode *10 

            if episode is None or episode not in test_dirs :
                continue 

            score =soc +dispatch 
            if (
            best_choice is None 
            or score >best_choice [1 ]
            or (score ==best_choice [1 ]and episode >best_choice [0 ])
            ):
                best_choice =(episode ,score )

    if best_choice is not None :
        return best_choice [0 ]

    return max (test_dirs )


def pick_default_model_dir (project_root ):
    """Auto-detect the latest model dir under archive/. Falls back to archive/ root."""
    archive_dir =os .path .join (project_root ,"archive")
    try :
        return str (find_latest_model_dir (archive_dir ))
    except FileNotFoundError :
        return archive_dir 


def sanitize_label (value ):
    cleaned =re .sub (r"[^A-Za-z0-9._-]+","_",str (value ))
    cleaned =cleaned .strip ("._-")
    return cleaned or "item"


def build_plot_metrics_from_df (df ):
    return {
    "soc_miss_count":(100.0 -df ["soc_hit_rate"]).tolist (),
    "surplus_absorption_rate":df ["surplus_absorption_rate"].tolist (),
    "supply_cooperation_rate":df ["supply_cooperation_rate"].tolist (),
    "departing_evs":df ["departing_evs"].astype (int ).tolist (),
    "departing_evs_soc_met":df ["departing_evs_soc_met"].astype (int ).tolist (),
    "surplus_steps":df ["surplus_steps"].astype (int ).tolist (),
    "surplus_within_narrow":df ["surplus_within_narrow"].astype (int ).tolist (),
    "shortage_steps":df ["shortage_steps"].astype (int ).tolist (),
    "shortage_within_narrow":df ["shortage_within_narrow"].astype (int ).tolist (),
    "avg_switches":df ["avg_switches"].tolist (),
    "avg_soc_deficit":df ["avg_soc_deficit"].tolist (),
    }


def save_force_timeline_plot (step_df ,run_dir ,episode_index ):
    if step_df .empty :
        return 

    x =step_df ["step"].to_numpy ()
    force_counts =step_df ["force_ev_step_overrides"].to_numpy ()
    controllable_counts =step_df ["active_evs_total"].to_numpy ()
    cumulative =np .cumsum (force_counts )
    cumulative_controllable =np .cumsum (controllable_counts )
    step_ratio =np .where (controllable_counts !=0 ,force_counts /controllable_counts *100.0 ,0.0 )
    cumulative_ratio =np .where (cumulative_controllable !=0 ,cumulative /cumulative_controllable *100.0 ,0.0 )

    fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(16 ,10 ),sharex =True )
    ax1 .bar (x ,force_counts ,color ="tab:red",alpha =0.75 )
    ax1 .set_ylabel ("Forced EV-step count")
    ax1 .grid (alpha =0.3 )

    ax2 .plot (x ,step_ratio ,color ="tab:orange",linewidth =2 ,alpha =0.7 ,label ="Step ratio")
    ax2 .plot (x ,cumulative_ratio ,color ="tab:blue",linewidth =3 ,label ="Cumulative ratio")
    ax2 .set_xlabel ("Step")
    ax2 .set_ylabel ("Forced EV-step ratio (%)")
    ax2 .grid (alpha =0.3 )
    ax2 .legend ()

    fig .tight_layout ()
    out_path =os .path .join (run_dir ,f"execute_force_timeline_episode_{episode_index}.png")
    fig .savefig (out_path ,dpi =300 ,bbox_inches ="tight")
    plt .close (fig )


def save_day_summary_plot (day_summary ,output_dir ):
    if day_summary .empty :
        return 

    x =np .arange (len (day_summary ))
    labels =day_summary ["day"].astype (str ).tolist ()

    fig ,axes =plt .subplots (3 ,1 ,figsize =(22 ,16 ),sharex =True )
    axes [0 ].plot (x ,day_summary ["soc_hit_rate_weighted"],color ="tab:blue",linewidth =3 ,label ="SoC hit")
    axes [0 ].plot (
    x ,
    day_summary ["dispatch_tracking_rate_weighted"],
    color ="tab:red",
    linewidth =3 ,
    label ="Dispatch tracking",
    )
    axes [0 ].set_ylabel ("Rate (%)")
    axes [0 ].set_ylim (0 ,105 )
    axes [0 ].grid (alpha =0.3 )
    axes [0 ].legend ()

    axes [1 ].bar (x ,day_summary ["forced_ev_step_overrides"],color ="tab:orange",alpha =0.8 ,label ="Forced EV-step")
    axes [1 ].set_ylabel ("Forced EV-step")
    axes [1 ].grid (alpha =0.3 )
    ax_force_ratio =axes [1 ].twinx ()
    ax_force_ratio .plot (
    x ,
    day_summary ["forced_ev_step_ratio_pct"],
    color ="tab:red",
    linewidth =3 ,
    label ="Forced ratio",
    )
    ax_force_ratio .set_ylabel ("Forced ratio (%)")
    ax_force_ratio .set_ylim (bottom =0 )

    axes [2 ].bar (x ,day_summary ["departing_evs"],color ="lightgray",alpha =0.9 ,label ="Departing EVs")
    axes [2 ].bar (
    x ,
    day_summary ["departing_evs_soc_met"],
    color ="tab:green",
    alpha =0.8 ,
    label ="Departing EVs met",
    )
    axes [2 ].set_ylabel ("EV count")
    axes [2 ].set_xlabel ("Day")
    axes [2 ].grid (alpha =0.3 )
    axes [2 ].legend ()
    axes [2 ].set_xticks (x )
    axes [2 ].set_xticklabels (labels ,rotation =60 ,ha ="right")

    handles1 ,labels1 =axes [1 ].get_legend_handles_labels ()
    handles2 ,labels2 =ax_force_ratio .get_legend_handles_labels ()
    axes [1 ].legend (handles1 +handles2 ,labels1 +labels2 ,loc ="upper left")

    fig .tight_layout ()
    out_path =os .path .join (output_dir ,"execute_day_summary.png")
    fig .savefig (out_path ,dpi =300 ,bbox_inches ="tight")
    plt .close (fig )


def save_force_overview_plot (df ,output_dir ):
    if df .empty :
        return 

    x =np .arange (len (df ))
    labels =[f"{day}\nR{int(repeat):02d}"for day ,repeat in zip (df ["day"],df ["repeat"])]

    fig ,ax1 =plt .subplots (figsize =(22 ,8 ))
    ax1 .bar (x ,df ["forced_ev_step_overrides"],color ="tab:orange",alpha =0.8 ,label ="Forced EV-step")
    ax1 .set_ylabel ("Forced EV-step")
    ax1 .grid (alpha =0.3 )

    ax2 =ax1 .twinx ()
    ax2 .plot (x ,df ["forced_ev_step_ratio_pct"],color ="tab:red",linewidth =3 ,label ="Forced ratio")
    ax2 .set_ylabel ("Forced ratio (%)")
    ax2 .set_ylim (bottom =0 )

    ax1 .set_xticks (x )
    ax1 .set_xticklabels (labels ,rotation =60 ,ha ="right")
    ax1 .set_xlabel ("Episode run")

    handles1 ,labels1 =ax1 .get_legend_handles_labels ()
    handles2 ,labels2 =ax2 .get_legend_handles_labels ()
    ax1 .legend (handles1 +handles2 ,labels1 +labels2 ,loc ="upper left")

    fig .tight_layout ()
    out_path =os .path .join (output_dir ,"execute_force_summary.png")
    fig .savefig (out_path ,dpi =300 ,bbox_inches ="tight")
    plt .close (fig )


def save_episode_artifacts (run_dir ,row ,step_df ,episode_data ):
    os .makedirs (run_dir ,exist_ok =True )
    step_df .to_csv (os .path .join (run_dir ,"step_trace.csv"),index =False )

    episode_key =int (row ["episode_index"])
    all_episode_data ={episode_key :episode_data }
    try :
        plot_station_cooperation_full (all_episode_data ,run_dir ,random_window =False ,title_prefix ="Test Results")
    except Exception as exc :
        print (f"[WARN] station cooperation plot skipped: {exc}")
    try :
        plot_power_mismatch_analysis (all_episode_data ,run_dir ,title_prefix ="Test Results")
    except Exception as exc :
        print (f"[WARN] power mismatch plot skipped: {exc}")
    try :
        plot_reward_breakdown (all_episode_data ,run_dir ,title_prefix ="Test Results")
    except Exception as exc :
        print (f"[WARN] reward breakdown plot skipped: {exc}")
    try :
        plot_arrival_counts (all_episode_data ,run_dir ,title_prefix ="Test Results")
    except Exception as exc :
        print (f"[WARN] arrival plot skipped: {exc}")
    save_force_timeline_plot (step_df ,run_dir ,episode_key )


def save_summary_artifacts (output_dir ,episode_df ,day_summary_df ):
    if episode_df .empty :
        return 

    metrics =build_plot_metrics_from_df (episode_df )
    try :
        plot_daily_rewards (
        episode_df ["local_avg_reward"].tolist (),
        episode_df ["global_avg_reward"].tolist (),
        output_dir ,
        episode_num =len (episode_df ),
        performance_metrics =metrics ,
        title_prefix ="Test Results",
        )
    except Exception as exc :
        print (f"[WARN] reward summary plot skipped: {exc}")
    try :
        plot_performance_metrics (metrics ,output_dir ,title_prefix ="Test Results")
    except Exception as exc :
        print (f"[WARN] performance summary plot skipped: {exc}")
    try :
        save_day_summary_plot (day_summary_df ,output_dir )
    except Exception as exc :
        print (f"[WARN] day summary plot skipped: {exc}")
    try :
        save_force_overview_plot (episode_df ,output_dir )
    except Exception as exc :
        print (f"[WARN] force summary plot skipped: {exc}")


def parse_args (project_root ):
    default_model_dir =pick_default_model_dir (project_root )
    parser =argparse .ArgumentParser (
    description ="Runtime evaluation with optional force charging override (inference only)."
    )
    parser .add_argument (
    "--model-dir",
    default =default_model_dir ,
    help ="Model root or TEST folder containing actor_*.pth",
    )
    parser .add_argument (
    "--episode",
    type =int ,
    default =None ,
    help ="Episode number to load. Default: None (auto-detect best TEST* by SoC+Dispatch score).",
    )
    parser .add_argument (
    "--xlsx",
    default =os .path .join (project_root ,"data","input.demand_fromPJM","08 2024.xlsx"),
    help ="Workbook path used for test demand creation",
    )
    parser .add_argument (
    "--split-script",
    default =os .path .join (project_root ,"data","input.demand_fromPJM","5mindivide.py"),
    help ="Splitter script path",
    )
    parser .add_argument (
    "--split-out-dir",
    default =os .path .join (project_root ,"data","input.demand_fromPJM","output_5min"),
    help ="Directory where day_YYYY-MM-DD.csv files are stored",
    )
    parser .add_argument ("--skip-split",action ="store_true",help ="Skip xlsx split and use existing day files")
    parser .add_argument ("--repeats",type =int ,default =1 ,help ="Number of repeats per day file. Default: 1")
    parser .add_argument ("--max-days",type =int ,default =0 ,help ="Use first N day files only. 0 means all")
    parser .add_argument ("--seed",type =int ,default =ENV_SEED ,help ="Random seed for runtime")
    parser .add_argument ("--disable-force",action ="store_true",help ="Disable force charging override")
    parser .add_argument (
    "--force-trigger-ratio",
    type =float ,
    default =0.5 ,
    help ="Force full charge when remaining need >= trigger_ratio * remaining max charge. Default 0.5.",
    )
    parser .add_argument (
    "--force-slack-kwh",
    type =float ,
    default =0.0 ,
    help ="Additional kWh margin for the full-charge trigger.",
    )
    parser .add_argument (
    "--output-dir",
    default =os .path .join (project_root ,"execute_results",datetime .now ().strftime ("%Y%m%d_%H%M%S")),
    help ="Directory for CSV, JSON, and graph outputs",
    )
    return parser .parse_args ()



def main ():
    project_root =os .path .abspath (os .path .dirname (__file__ ))
    args =parse_args (project_root )

    model_dir =resolve_path (project_root ,args .model_dir )
    xlsx_path =resolve_path (project_root ,args .xlsx )
    split_script =resolve_path (project_root ,args .split_script )
    split_out_dir =resolve_path (project_root ,args .split_out_dir )
    output_dir =resolve_path (project_root ,args .output_dir )

    if not args .skip_split :
        run_splitter (split_script ,xlsx_path ,split_out_dir )

    month_key =infer_year_month_from_xlsx (xlsx_path )
    if month_key is None :
        day_files =sorted (glob .glob (os .path .join (split_out_dir ,"day_*.csv")))
    else :
        month_pattern =os .path .join (split_out_dir ,f"day_{month_key}-*.csv")
        day_files =sorted (glob .glob (month_pattern ))
        if len (day_files )==0 :
            print (f"[WARN] No files matched {month_pattern}. Falling back to all day files.")
            day_files =sorted (glob .glob (os .path .join (split_out_dir ,"day_*.csv")))

    if int (args .max_days )!=0 :
        day_files =day_files [:int (args .max_days )]

    if len (day_files )==0 :
        raise FileNotFoundError (f"No day CSV files found in: {split_out_dir}")

    if int (args .max_days )==0 and len (day_files )!=31 :
        print (f"[WARN] Expected 31 days but found {len(day_files)} day files.")

    day_payloads =[]
    for csv_path in day_files :
        day_label =extract_day_label (csv_path )
        demand_series =load_scaled_episode (csv_path )
        day_payloads .append ((day_label ,demand_series ,csv_path ))
    seed_value =None if args .seed is None else int (args .seed )
    set_env_seed (seed_value )
    bootstrap_env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )
    bootstrap_env .reset (net_demand_series =day_payloads [0 ][1 ])
    agent =build_agent (bootstrap_env )

    requested_episode =args .episode 
    if requested_episode is None :
        requested_episode =choose_best_episode (model_dir )
        print (f"[execute] auto-detected best episode: {requested_episode}")
    model_path ,model_episode =find_model_path_and_episode (model_dir ,requested_episode )
    map_loc =None if torch .cuda .is_available ()else "cpu"
    if hasattr (agent ,"load_actors"):
        agent .load_actors (model_path ,episode =model_episode ,map_location =map_loc )
    else :
        agent .load_models (model_path ,episode =model_episode ,map_location =map_loc )
    agent .set_test_mode (True )

    rows =[]
    total_runs =int (args .repeats )*len (day_payloads )
    run_idx =0 
    started =time .time ()

    force_enabled =not bool (args .disable_force )
    force_ratio =float (args .force_trigger_ratio )
    force_slack =float (args .force_slack_kwh )
    os .makedirs (output_dir ,exist_ok =True )
    runs_dir =os .path .join (output_dir ,"runs")
    os .makedirs (runs_dir ,exist_ok =True )

    for repeat in range (1 ,int (args .repeats )+1 ):
        for day_idx ,payload in enumerate (day_payloads ,start =1 ):
            run_idx +=1 
            day_label ,demand_series ,source_csv =payload 
            t0 =time .time ()
            row ,step_df ,episode_data =run_single_episode (
            agent ,
            day_label ,
            demand_series ,
            force_enabled =force_enabled ,
            trigger_ratio =force_ratio ,
            slack_kwh =force_slack ,
            )
            row ["episode_index"]=run_idx 
            row ["repeat"]=repeat 
            row ["day_index"]=day_idx 
            row ["source_csv"]=source_csv 
            row ["elapsed_sec"]=time .time ()-t0 
            rows .append (row )

            run_dir_name =f"run_{run_idx:03d}_{sanitize_label(day_label)}_rep{repeat:02d}"
            run_dir =os .path .join (runs_dir ,run_dir_name )
            save_episode_artifacts (run_dir ,row ,step_df ,episode_data )

            print (
            "[{}/{}] rep={:02d} day={} SoC={:.2f}% Dispatch={:.2f}% "
            "ForceEVStep={} ({:.2f}%) ForceSteps={} Elapsed={:.1f}s".format (
            run_idx ,
            total_runs ,
            repeat ,
            day_label ,
            row ["soc_hit_rate"],
            row ["dispatch_tracking_rate"],
            row ["forced_ev_step_overrides"],
            row ["forced_ev_step_ratio_pct"],
            row ["forced_active_steps"],
            row ["elapsed_sec"],
            )
            )

    agent .set_test_mode (False )
    total_elapsed =time .time ()-started 

    df =pd .DataFrame (rows )

    episode_csv =os .path .join (output_dir ,"episode_results.csv")
    df .to_csv (episode_csv ,index =False )
    day_summary_csv =os .path .join (output_dir ,"summary_by_day.csv")
    overall_csv =os .path .join (output_dir ,"summary_overall.csv")
    run_config_json =os .path .join (output_dir ,"run_config.json")

    day_summary =df .groupby ("day",as_index =False ).agg (
    episodes =("episode_index","count"),
    local_avg_reward =("local_avg_reward","mean"),
    global_avg_reward =("global_avg_reward","mean"),
    soc_hit_rate_mean =("soc_hit_rate","mean"),
    departing_evs =("departing_evs","sum"),
    departing_evs_soc_met =("departing_evs_soc_met","sum"),
    dispatch_success_steps =("dispatch_success_steps","sum"),
    dispatch_total_steps =("dispatch_total_steps","sum"),
    controllable_ev_steps =("controllable_ev_steps","sum"),
    forced_ev_step_overrides =("forced_ev_step_overrides","sum"),
    forced_active_steps =("forced_active_steps","sum"),
    )
    day_summary ["soc_hit_rate_weighted"]=np .where (
    day_summary ["departing_evs"]!=0 ,
    day_summary ["departing_evs_soc_met"]/day_summary ["departing_evs"]*100.0 ,
    0.0 ,
    )
    day_summary ["dispatch_tracking_rate_weighted"]=np .where (
    day_summary ["dispatch_total_steps"]!=0 ,
    day_summary ["dispatch_success_steps"]/day_summary ["dispatch_total_steps"]*100.0 ,
    0.0 ,
    )
    day_summary ["forced_ev_step_ratio_pct"]=np .where (
    day_summary ["controllable_ev_steps"]!=0 ,
    day_summary ["forced_ev_step_overrides"]/day_summary ["controllable_ev_steps"]*100.0 ,
    0.0 ,
    )
    day_summary ["forced_overrides"]=day_summary ["forced_ev_step_overrides"]
    day_summary ["forced_steps"]=day_summary ["forced_active_steps"]
    day_summary .to_csv (day_summary_csv ,index =False )

    overall =build_overall_summary (df ,repeats =int (args .repeats ),day_count =len (day_payloads ))
    overall ["model_path"]=model_path 
    overall ["model_episode"]=int (model_episode )
    overall ["total_elapsed_sec"]=float (total_elapsed )
    pd .DataFrame ([overall ]).to_csv (overall_csv ,index =False )

    save_summary_artifacts (output_dir ,df ,day_summary )

    run_config ={
    "timestamp":datetime .now ().isoformat (),
    "model_dir":model_dir ,
    "model_path":model_path ,
    "model_episode":int (model_episode ),
    "xlsx_path":xlsx_path ,
    "split_script":split_script ,
    "split_out_dir":split_out_dir ,
    "day_files":day_files ,
    "repeats":int (args .repeats ),
    "seed":seed_value ,
    "force_enabled":bool (force_enabled ),
    "force_trigger_ratio":float (force_ratio ),
    "force_slack_kwh":float (force_slack ),
    "total_runs":int (total_runs ),
    "total_elapsed_sec":float (total_elapsed ),
    }
    with open (run_config_json ,"w",encoding ="utf-8")as f :
        json .dump (run_config ,f ,ensure_ascii =False ,indent =2 )

    print ("\n===== Execute Finished =====")
    print (f"Model: {model_path} (episode {model_episode})")
    print (f"Runs: {total_runs} Days: {len(day_payloads)} Repeats: {int(args.repeats)}")
    print ("SoC hit (weighted): {:.2f}%".format (overall ["soc_hit_rate_weighted"]))
    print ("Dispatch (weighted): {:.2f}%".format (overall ["dispatch_tracking_rate_weighted"]))
    print (
    "Force EV-step overrides: {} / {} ({:.2f}%)".format (
    overall ["forced_ev_step_overrides"],
    overall ["controllable_ev_steps"],
    overall ["forced_ev_step_ratio_pct"],
    )
    )
    print ("Force active steps: {}".format (overall ["forced_active_steps"]))
    print (f"Output dir: {output_dir}")


if __name__ =="__main__":
    main ()
