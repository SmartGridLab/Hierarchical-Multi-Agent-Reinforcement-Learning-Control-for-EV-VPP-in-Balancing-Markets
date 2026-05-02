"""
Training entry point for EV multi-agent charging control.

This module coordinates the full training experiment:
- create an archive directory and snapshot the source code;
- load training demand episodes;
- construct EVEnv and the selected agent class;
- fill replay memory during warmup;
- run online environment interaction and gradient updates;
- record TensorBoard diagnostics, CSV/PNG summaries, and checkpoint evaluations.

Training data flow:
1. A daily demand series is sampled from the training split.
2. EVEnv generates station-specific arrivals and EV profiles for the episode.
3. The agent observes normalized station states and outputs per-station EV-slot
   actions.
4. EVEnv applies actions, returns local/global rewards, and exposes physical
   power traces.
5. Agents with replay buffers cache the transition and update their networks.
6. Periodic deterministic tests run through `tools.evaluator.test()` on the held
   out demand split.

Output structure:
- `archive/{model_name}_{timestamp}/code_snapshot`: source snapshot.
- `.../performance`: TensorBoard event files for training diagnostics.
- `.../results`: train/test CSVs, plots, and checkpoint folders.
"""
import os
import numpy as np
import torch
import random
from datetime import datetime
import traceback
import time
import warnings
import logging


warnings .filterwarnings ("ignore",category =UserWarning ,module ="matplotlib")
warnings .filterwarnings ("ignore",message ="Glyph .* missing from font")
logging .getLogger ('matplotlib.font_manager').setLevel (logging .ERROR )
logging .getLogger ('matplotlib.ticker').setLevel (logging .ERROR )

from torch .utils .tensorboard import SummaryWriter
from environment.normalize import normalize_observation
from tools.Utils import (
create_tensorboard_writer ,
GradientLossVisualizer ,
snapshot_code_to_archive ,
InterruptHandler ,
launch_tensorboard ,
write_train_episode_tb_scalars ,
)
import matplotlib
matplotlib .use ('Agg')
matplotlib .rcParams ['font.family']='sans-serif'
matplotlib .rcParams ['font.sans-serif']=['Arial','Helvetica','Liberation Sans','FreeSans','sans-serif']
from tools.Utils import plot_daily_rewards ,plot_performance_metrics
from environment.readcsv import load_multiple_demand_files
from tools.evaluator import set_env_seed ,test
from Config import (
NUM_EPISODES ,EPISODE_STEPS ,NUM_EVS ,NUM_STATIONS ,
LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC ,
USE_MILP ,USE_INDEPENDENT_DDPG ,ENV_SEED ,
REGULAR_MADDPG ,
GAMMA ,TAU ,BATCH_SIZE ,SMOOTHL1_BETA ,
MEMORY_SIZE ,WARMUP_STEPS ,
TRAIN_INTERIM_CSV_INTERVAL_EPISODES ,TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES ,
USE_SWITCHING_CONSTRAINTS ,LOCAL_SWITCH_PENALTY ,
USE_STATION_TOTAL_POWER_LIMIT ,LOCAL_STATION_LIMIT_PENALTY ,
)
from Config import (
USE_SHARED_OBS_DDPG ,USE_SHARED_OBS_SAC ,
MEASURE_LP_STEP_TIME ,MEASURE_STEP_INDEX ,
)
from environment.EVEnv import EVEnv
from training.Agent import MADDPG
from training.benchmark_agents.independent_ddpg import IndependentDDPG
from training.benchmark_agents.shared_obs_ddpg import SharedObsDDPG
from training.benchmark_agents.shared_obs_sac import SharedObsSAC
from training.Agent .maddpg import device

INTERIM_TEST_INTERVAL =max (1 ,int (TRAIN_INTERIM_CSV_INTERVAL_EPISODES ))
INTERIM_PLOT_INTERVAL =max (1 ,int (TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES ))


_interrupt_handler =InterruptHandler ()

def sample_episode_demand_strict (demand_data ,episode_steps ):
    """
    Sample one demand episode and force it to exactly `episode_steps` values.

    Longer demand arrays are truncated and shorter arrays are right-padded with
    zeros. The function uses Python's global RNG so train/evaluation RNG control
    remains centralized through `set_env_seed()`.
    """
    if int (episode_steps )<=0 :
        raise ValueError (f"episode_steps must be > 0, got {episode_steps}")
    if len (demand_data )==0 :
        raise ValueError ("demand_data is empty")
    _idx =random .randrange (len (demand_data ))
    _data =np .asarray (demand_data [_idx ],dtype =float ).reshape (-1 )
    if _data .size ==0 :
        raise ValueError ("Sampled demand episode is empty.")

    if _data .size >=int (episode_steps ):
        return _data [:int (episode_steps )]
    return np .pad (_data ,(0 ,int (episode_steps )-int (_data .size )))





def create_model_directory (model_name ):
    """
    Create the archive folder for one training run.

    The timestamped directory stores results, TensorBoard files, optional input
    copies, and a source-code snapshot so later analysis can be tied to the exact
    implementation used for the run.
    """
    base_dir =os .getcwd ()

    archive_dir =os .path .join (base_dir ,"archive")
    os .makedirs (archive_dir ,exist_ok =True )

    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    model_dir_name =f"{model_name}_{timestamp}"
    model_dir =os .path .join (archive_dir ,model_dir_name )



    os .makedirs (model_dir ,exist_ok =True )

    subdirs =["results","runs","performance","input"]
    for subdir in subdirs :
        os .makedirs (os .path .join (model_dir ,subdir ),exist_ok =True )


    os .makedirs (os .path .join (model_dir ,"results","test"),exist_ok =True )


    try :
        snapshot_code_to_archive (model_dir )
    except Exception as exc :
        warnings .warn (f"Failed to snapshot source code: {exc}")

    return model_dir

def train (num_episodes =NUM_EPISODES ,random_window =True ,agent =None ,start_episode =0 ,model_name ="model",working_dir =None ,all_rewards =None ,performance_metrics =None ,all_episode_data =None ,profile_episode =None ):
    """
    Run training and return the trained agent plus collected metrics.

    When `agent` is None, the active Config flags select one implementation.
    The default path is MADDPG with local station critics and a global critic.
    `all_rewards`, `performance_metrics`, and `all_episode_data` allow resumed
    or externally managed runs to continue appending to existing containers.
    """
    del random_window, profile_episode

    if working_dir is None :
        working_dir =create_model_directory (model_name )

    set_env_seed (ENV_SEED )


    performance_dir =os .path .join (working_dir ,"performance")
    os .makedirs (performance_dir ,exist_ok =True )
    if USE_MILP :
        tb_writer =None
        print ("[Info] MILP mode: TensorBoard output is disabled.")
    else :
        tb_writer =create_tensorboard_writer (log_dir =performance_dir )
        try :
            launch_tensorboard (working_dir )
        except Exception as exc :
            warnings .warn (f"Failed to launch TensorBoard: {exc}")

    _interrupt_handler .setup ()


    env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )


    demand_data_train =None
    try :

        all_demand_data =load_multiple_demand_files (train_split =25 )
        demand_data_train =all_demand_data ['train']
    except FileNotFoundError :
        error_message ="Failed to load training demand CSV files."
        print (error_message )
        raise

    if not demand_data_train :
        raise RuntimeError ("Training demand pool is empty after CSV load.")
    episode_demand =sample_episode_demand_strict (demand_data_train ,env .episode_steps )
    env .reset (net_demand_series =episode_demand )

    state_dim =env ._get_obs ().shape [1 ]

    if agent is None :




        algo_flags ={
        "USE_MILP":bool (USE_MILP ),
        "USE_INDEPENDENT_DDPG":bool (USE_INDEPENDENT_DDPG ),
        "USE_SHARED_OBS_DDPG":bool (USE_SHARED_OBS_DDPG ),
        "USE_SHARED_OBS_SAC":bool (USE_SHARED_OBS_SAC ),
        "REGULAR_MADDPG":bool (REGULAR_MADDPG ),
        }
        enabled =[k for k ,v in algo_flags .items ()if v ]
        if len (enabled )>1 :
            raise ValueError (
            f"Algorithm selection must be exclusive, but multiple are enabled: {enabled}. "
            "Please set exactly one of USE_MILP / USE_INDEPENDENT_DDPG / USE_SHARED_OBS_DDPG / USE_SHARED_OBS_SAC / REGULAR_MADDPG to True."
            )

        # The agent implementations share the same EVEnv interface but differ in
        # how they use observations/rewards. This block keeps model construction
        # explicit so the saved archive is reproducible from Config alone.
        if USE_MILP :

            from training.benchmark_agents.milp_agent import MILPAgent
            from Config import MILP_HORIZON
            agent =MILPAgent (max_evs_per_station =env .max_ev_per_station ,horizon =MILP_HORIZON )
        elif USE_INDEPENDENT_DDPG :

            agent =IndependentDDPG (state_dim ,env .max_ev_per_station ,n_agent =env .num_stations ,
            num_episodes =num_episodes ,
            batch =BATCH_SIZE ,gamma =GAMMA ,tau =TAU ,
            lr_a =LR_ACTOR ,lr_c =LR_CRITIC_LOCAL ,
            smoothl1_beta =SMOOTHL1_BETA )

            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )
        elif USE_SHARED_OBS_DDPG :

            from Config import (
            SHARED_OBS_LR_ACTOR ,
            SHARED_OBS_LR_CRITIC_LOCAL ,
            SHARED_OBS_GLOBAL_REWARD_WEIGHT ,
            SHARED_OBS_EPSILON_END_EPISODE ,
            SHARED_OBS_OU_NOISE_END_EPISODE ,
            SHARED_OBS_OU_NOISE_SCALE_FINAL ,
            )
            agent =SharedObsDDPG (state_dim ,env .max_ev_per_station ,n_agent =env .num_stations ,
            num_episodes =num_episodes ,
            batch =BATCH_SIZE ,gamma =GAMMA ,tau =TAU ,
            lr_a =SHARED_OBS_LR_ACTOR ,lr_c =SHARED_OBS_LR_CRITIC_LOCAL ,
            smoothl1_beta =SMOOTHL1_BETA ,
            global_reward_weight =SHARED_OBS_GLOBAL_REWARD_WEIGHT ,
            epsilon_end_episode =SHARED_OBS_EPSILON_END_EPISODE ,
            ou_noise_end_episode =SHARED_OBS_OU_NOISE_END_EPISODE ,
            ou_noise_scale_final =SHARED_OBS_OU_NOISE_SCALE_FINAL )
            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )
        elif USE_SHARED_OBS_SAC :

            from Config import (
            SAC_LR_ACTOR ,SAC_LR_CRITIC ,SAC_LR_ALPHA ,
            SAC_TAU ,SAC_ALPHA_INIT ,SAC_TARGET_ENTROPY_SCALE ,
            SAC_GLOBAL_REWARD_WEIGHT ,
            )
            agent =SharedObsSAC (
            state_dim ,env .max_ev_per_station ,n_agent =env .num_stations ,
            num_episodes =num_episodes ,
            batch =BATCH_SIZE ,gamma =GAMMA ,tau =SAC_TAU ,
            lr_a =SAC_LR_ACTOR ,lr_c =SAC_LR_CRITIC ,lr_alpha =SAC_LR_ALPHA ,
            global_reward_weight =SAC_GLOBAL_REWARD_WEIGHT ,
            alpha_init =SAC_ALPHA_INIT ,
            target_entropy_scale =SAC_TARGET_ENTROPY_SCALE ,
            )
            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )
        elif REGULAR_MADDPG :


            # MADDPG training objective: single-step global critic target,
            # mean-centered global mixer, and one local critic per station.
            from Config import TAU_GLOBAL ,TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL
            agent =MADDPG (
            s_dim =state_dim ,
            max_evs_per_station =env .max_ev_per_station ,
            n_agent =env .num_stations ,
            num_episodes =num_episodes ,
            batch =BATCH_SIZE ,
            gamma =GAMMA ,
            tau =TAU ,
            lr_a =LR_ACTOR ,
            lr_c =LR_CRITIC_LOCAL ,
            lr_global_c =LR_GLOBAL_CRITIC ,
            tau_global =TAU_GLOBAL ,
            td3_sigma =TD3_SIGMA_GLOBAL ,
            td3_clip =TD3_CLIP_GLOBAL ,
            smoothl1_beta =0.01
            )
            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )
        else :

            # MADDPG training objective: single-step global critic target,
            # mean-centered global mixer, and one local critic per station.
            from Config import TAU_GLOBAL ,TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL
            agent =MADDPG (
            s_dim =state_dim ,
            max_evs_per_station =env .max_ev_per_station ,
            n_agent =env .num_stations ,
            num_episodes =num_episodes ,
            batch =BATCH_SIZE ,
            gamma =GAMMA ,
            tau =TAU ,
            lr_a =LR_ACTOR ,
            lr_c =LR_CRITIC_LOCAL ,
            lr_global_c =LR_GLOBAL_CRITIC ,

            tau_global =TAU_GLOBAL ,
            td3_sigma =TD3_SIGMA_GLOBAL ,
            td3_clip =TD3_CLIP_GLOBAL ,
            smoothl1_beta =0.01
            )

            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )


    try :
        if USE_MILP :
            agent .writer =None
            agent .use_tensorboard =False
        else :
            runs_dir =os .path .join (working_dir ,"runs")
            os .makedirs (runs_dir ,exist_ok =True )
            if hasattr (agent ,'writer'):
                if agent .writer is not None :
                    try :
                        agent .writer .close ()
                    except Exception as exc :
                        warnings .warn (f"Failed to close previous TensorBoard writer: {exc}")
            agent .writer =SummaryWriter (log_dir =runs_dir )
            agent .use_tensorboard =True


    except Exception as exc :
        warnings .warn (f"TensorBoard writer setup failed: {exc}")

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    if all_rewards is None :
        all_rewards =[]

    all_local_rewards =[]
    all_global_rewards =[]


    if all_episode_data is None :
        all_episode_data ={}

    enable_switch_metrics =bool (USE_SWITCHING_CONSTRAINTS )and (float (LOCAL_SWITCH_PENALTY )!=0.0 )
    enable_stlimit_metrics =bool (USE_STATION_TOTAL_POWER_LIMIT )and (float (LOCAL_STATION_LIMIT_PENALTY )!=0.0 )


    if performance_metrics is None :
        performance_metrics ={

        'soc_miss_count':[],
        'surplus_absorption_rate':[],
        'supply_cooperation_rate':[],

        'departing_evs':[],
        'departing_evs_soc_met':[],
        'surplus_steps':[],
        'surplus_within_narrow':[],
        'shortage_steps':[],
        'shortage_within_narrow':[],
        'avg_soc_deficit':[],
        }
        if enable_switch_metrics :
            performance_metrics ['avg_switches']=[]
        if enable_stlimit_metrics :
            performance_metrics ['station_limit_hits']=[]
            performance_metrics ['station_limit_steps']=[]
            performance_metrics ['station_charge_limit_hits']=[]
            performance_metrics ['station_discharge_limit_hits']=[]
            performance_metrics ['station_limit_penalty_total']=[]
            performance_metrics ['station_limit_penalty_per_step']=[]
            performance_metrics ['station_limit_penalty_per_hit']=[]

    try :
        for ep in range (start_episode +1 ,start_episode +num_episodes +1 ):






            # Warmup collects replay transitions using exploratory actions
            # before network updates are allowed to dominate the buffer.
            _is_warmup =hasattr (agent ,'buf')and getattr (agent .buf ,'size',0 )<WARMUP_STEPS

            if _is_warmup :
                env .record_snapshots =False
                _wu_demand =sample_episode_demand_strict (demand_data_train ,env .episode_steps )
                env .reset (net_demand_series =_wu_demand )
                agent .episode_start ()
                agent .update_active_evs (env )
                _wu_prefetch =None
                while True :
                    if _wu_prefetch is not None :
                        _wu_obs =normalize_observation (_wu_prefetch )
                        _wu_prefetch =None
                    else :
                        _wu_obs =normalize_observation (env .begin_step ())
                    agent .update_active_evs (env )
                    _wu_act =agent .act (_wu_obs ,env =env ,noise =True )
                    _ ,_wu_rl ,_wu_rg ,_wu_done ,_wu_info =env .apply_action (_wu_act ,build_info =False )
                    if all (_wu_done ):
                        _wu_next =normalize_observation (env ._get_obs ())
                    else :
                        _wu_next_raw =env .begin_step ()
                        _wu_next =normalize_observation (_wu_next_raw )
                        _wu_prefetch =_wu_next_raw
                    if hasattr (agent ,'cache_experience'):
                        _wu_sp =_wu_info ['station_powers']
                        _wu_rl_t =_wu_rl if torch .is_tensor (_wu_rl )else torch .as_tensor (_wu_rl ,dtype =torch .float32 ,device =device )
                        _wu_sp_t =_wu_sp if torch .is_tensor (_wu_sp )else torch .as_tensor (_wu_sp ,dtype =torch .float32 ,device =device )
                        agent .cache_experience (
                        torch .as_tensor (_wu_obs ,dtype =torch .float32 ,device =device ),
                        torch .as_tensor (_wu_next ,dtype =torch .float32 ,device =device ),
                        _wu_act ,
                        _wu_rl_t ,
                        torch .tensor (_wu_rg ,dtype =torch .float32 ,device =device ),
                        torch .as_tensor (_wu_done ,dtype =torch .float32 ,device =device ),
                        actual_station_powers =_wu_sp_t ,
                        actual_ev_power_kw =_wu_info .get ('actual_ev_power_kw'),
                        )
                        if not all (_wu_done ):
                            agent .update ()
                    if all (_wu_done ):
                        agent .episode_end ()
                        break
                _wu_buf =getattr (agent .buf ,'size',0 )
                if ep %5 ==1 or _wu_buf >=WARMUP_STEPS :
                    print (f"[WARMUP] ep={ep:4d}  buffer={_wu_buf}/{WARMUP_STEPS}",flush =True )
                continue


            # From this point onward, `training_ep` counts learned episodes only;
            # warmup episodes are excluded from reward/performance curves.
            if not getattr (agent ,'_learning_started_episode',None ):
                agent ._learning_started_episode =ep
                print (
                f"[Info] Warmup complete at ep={ep}. "
                f"Learning starts (training_ep=1/{num_episodes}).",
                flush =True ,
                )
            training_ep =ep -agent ._learning_started_episode +1




            env .record_snapshots =bool (getattr (agent ,'test_mode',False ))and not MEASURE_LP_STEP_TIME




            visualizer =GradientLossVisualizer (env .num_stations ,tb_writer )




            episode_demand =sample_episode_demand_strict (demand_data_train ,env .episode_steps )
            env .reset (net_demand_series =episode_demand )

            agent .episode_start ()
            ep_r =0.0

            ep_start_time =time .time ()

            agent .update_active_evs (env )

            episode_data ={
            'ag_requests':[],
            'total_ev_transport':[],
            'soc_data':{},
            'power_mismatch':[]
            }

            for i in range (1 ,env .num_stations +1 ):
                episode_data [f'actual_ev{i}']=[]

            for station_idx in range (env .num_stations ):

                episode_data ['soc_data'][f'station{station_idx+1}']={}


                for ev_idx ,ev in enumerate (env .stations_evs [station_idx ]):

                    ev_id =str (ev ['id'])

                    episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]={
                    'id':ev ['id'],
                    'station':station_idx ,
                    'depart':ev ['depart'],
                    'target':ev ['target'],
                    'times':[env .step_count ],
                    'soc':[ev ['soc']]
                    }


            ep_r =0
            ep_local_r =0.0
            ep_global_r =0.0

            ep_local_departure_r =0.0
            ep_local_progress_shaping_r =0.0
            ep_local_discharge_penalty_r =0.0
            ep_local_switch_penalty_r =0.0
            ep_local_station_limit_penalty_r =0.0

            station_local_reward_sums =[
            {
            "total":0.0 ,
            "departure":0.0 ,
            "progress_shaping":0.0 ,
            "discharge_penalty":0.0 ,
            "switch_penalty":0.0 ,
            "station_limit_penalty":0.0 ,
            }
            for _ in range (env .num_stations )
            ]

            total_surplus_available =0
            total_surplus_absorbed =0

            total_discharge_request =0
            total_discharge_fulfilled =0

            _prefetch_obs =None

            # Step loop: observe, act, apply physics/rewards, cache transition,
            # update the agent, and accumulate diagnostics for this episode.
            while True :

                MAX_RETRIES =3
                retry_count =0
                step_success =False

                def is_safe (x ):
                    if isinstance (x ,torch .Tensor ):
                        return torch .isfinite (x ).all ().item ()
                    else :
                        return np .isfinite (x ).all ()

                info =None
                act_tensor =None
                obs1 =None
                next_state =None
                r_local =None
                r_global =None
                done =None

                _step_obs_raw =_prefetch_obs
                _prefetch_obs =None

                while retry_count <MAX_RETRIES and not step_success :

                    if _step_obs_raw is not None and retry_count ==0 :
                        obs1 =normalize_observation (_step_obs_raw )
                        _step_obs_raw =None
                    else :
                        obs1 =normalize_observation (env .begin_step ())

                    agent .update_active_evs (env )


                    if MEASURE_LP_STEP_TIME :
                        # Optional MILP timing path used to estimate LP problem
                        # size and solver runtime at one selected step.
                        current_step =int (env .step_count )
                        if current_step <MEASURE_STEP_INDEX :


                            if current_step %10 ==1 or current_step ==1 :
                                print (f"[Measurement] Progress: Step {current_step}/{MEASURE_STEP_INDEX} (Random actions, {env.num_stations} stations)...")
                            act_tensor =(torch .rand ((env .num_stations ,env .max_ev_per_station ),device =device )*2.0 -1.0 )
                        elif current_step ==MEASURE_STEP_INDEX :

                            print (f"\n[Measurement] Measuring calculation time for Step {current_step}...")


                            def estimate_problem_size (agent ,env ):
                                """

                                """
                                if USE_MILP :

                                    H =getattr (agent ,'H',1 )
                                    total_active_evs =0
                                    num_stations_with_evs =0
                                    estimated_vars =0
                                    estimated_constraints =0

                                    for st in range (env .num_stations ):
                                        active =torch .nonzero (env .ev_mask [st ],as_tuple =False ).squeeze (-1 )
                                        if len (active )>0 :
                                            num_stations_with_evs +=1
                                            total_active_evs +=len (active )

                                            for ev_idx in active :
                                                r =max (0 ,int (env .depart [st ,ev_idx ].item ()-env .step_count ))

                                                estimated_vars +=min (H ,r )

                                                estimated_constraints +=min (H ,r )*2


                                    estimated_constraints +=H *2

                                    estimated_constraints +=total_active_evs

                                    estimated_vars +=H

                                    estimated_vars +=total_active_evs

                                    return {
                                    'num_variables':estimated_vars ,
                                    'num_constraints':estimated_constraints ,
                                    'total_active_evs':total_active_evs ,
                                    'num_stations_with_evs':num_stations_with_evs ,
                                    'horizon':H ,
                                    'episode_steps':None ,
                                    }
                                else :
                                    return {
                                    'num_variables':0 ,
                                    'num_constraints':0 ,
                                    'total_active_evs':0 ,
                                    'num_stations_with_evs':None ,
                                    'horizon':None ,
                                    'episode_steps':None ,
                                    }


                            problem_size =estimate_problem_size (agent ,env )
                            num_vars =problem_size ['num_variables']
                            num_constraints =problem_size ['num_constraints']
                            total_active_evs =problem_size ['total_active_evs']
                            num_stations_with_evs =problem_size ['num_stations_with_evs']
                            horizon =problem_size ['horizon']
                            episode_steps =problem_size ['episode_steps']

                            n ,m =num_vars ,num_constraints
                            flops_per_iteration =m *m *m
                            estimated_iterations =max (m ,int (n *0.1 ))
                            estimated_flops =flops_per_iteration *estimated_iterations
                            simple_flops_per_iteration =n *m
                            simple_estimated_flops =simple_flops_per_iteration *estimated_iterations

                            if n ==0 or m ==0 :
                                complexity_order ='N/A'
                            elif n ==m :
                                complexity_order =f"O({n}^3)"
                            elif n <=m :
                                complexity_order =f"O({n}^2*{m})"
                            else :
                                complexity_order =f"O({n}*{m}^2)"


                            def format_flops (flops ):
                                if flops >=1e12 :
                                    return f"{flops/1e12:.2f} TFLOP"
                                elif flops >=1e9 :
                                    return f"{flops/1e9:.2f} GFLOP"
                                elif flops >=1e6 :
                                    return f"{flops/1e6:.2f} MFLOP"
                                elif flops >=1e3 :
                                    return f"{flops/1e3:.2f} KFLOP"
                                else :
                                    return f"{flops:.0f} FLOP"

                            RTX4080_TFLOPS_FP32 =83.0
                            RTX4080_TFLOPS_FP16 =1248.0

                            gpu_time_fp32 =estimated_flops /(RTX4080_TFLOPS_FP32 *1e12 )if estimated_flops >0 else 0
                            gpu_time_fp16 =estimated_flops /(RTX4080_TFLOPS_FP16 *1e12 )if estimated_flops >0 else 0
                            gpu_time_simple_fp32 =simple_estimated_flops /(RTX4080_TFLOPS_FP32 *1e12 )if simple_estimated_flops >0 else 0
                            gpu_time_simple_fp16 =simple_estimated_flops /(RTX4080_TFLOPS_FP16 *1e12 )if simple_estimated_flops >0 else 0


                            print ("============================================================")
                            print ("[Problem Size Estimation (Before Solving)]")
                            print (f"  - Number of variables (n): {num_vars}")
                            print (f"  - Number of constraints (m): {num_constraints}")
                            if total_active_evs >0 :
                                print (f"  - Total active EVs: {total_active_evs}")
                                print (f"  - Variables per EV (approx): {num_vars / total_active_evs:.2f}")
                                print (f"  - Constraints per EV (approx): {num_constraints / total_active_evs:.2f}")
                            if num_stations_with_evs is not None :
                                print (f"  - Stations with EVs: {num_stations_with_evs}")
                            if horizon is not None :
                                print (f"  - Horizon (H): {horizon}")
                            if episode_steps is not None :
                                print (f"  - Episode steps (T): {episode_steps}")
                            print ("------------------------------------------------------------")
                            print ("[Computational Complexity Estimation]")
                            print (f"  - Complexity order: {complexity_order}")
                            print (f"  - Estimated iterations: {estimated_iterations}")
                            print (f"  - Estimated FLOPs (O(m^3) per iteration): {format_flops(estimated_flops)}")
                            print (f"  - Estimated FLOPs (O(n*m) per iteration): {format_flops(simple_estimated_flops)}")
                            print ("------------------------------------------------------------")
                            print ("[GPU Performance Estimation (RTX 4080)]")
                            print ("  Note: Current solver (CBC) runs on CPU. Below is theoretical estimate")
                            print ("        if the problem could be solved on GPU.")
                            print (f"  - RTX 4080 FP32 performance: {RTX4080_TFLOPS_FP32:.1f} TFLOPS")
                            print (f"  - RTX 4080 FP16 performance: {RTX4080_TFLOPS_FP16:.1f} TFLOPS")
                            print ("============================================================")


                            print ("\n[Executing LP solver...]")
                            t_start =time .time ()
                            act_tensor =agent .act (obs1 ,env =env ,noise =False )
                            t_end =time .time ()
                            duration =t_end -t_start

                            complexity_info =getattr (agent ,'last_complexity_info',{})
                            actual_num_vars =complexity_info .get ('num_variables',num_vars )
                            actual_num_constraints =complexity_info .get ('num_constraints',num_constraints )


                            def format_flops (flops ):
                                if flops >=1e12 :
                                    return f"{flops/1e12:.2f} TFLOP"
                                elif flops >=1e9 :
                                    return f"{flops/1e9:.2f} GFLOP"
                                elif flops >=1e6 :
                                    return f"{flops/1e6:.2f} MFLOP"
                                elif flops >=1e3 :
                                    return f"{flops/1e3:.2f} KFLOP"
                                else :
                                    return f"{flops:.0f} FLOP"



                            RTX4080_TFLOPS_FP32 =83.0
                            RTX4080_TFLOPS_FP16 =1248.0



                            gpu_time_fp32 =estimated_flops /(RTX4080_TFLOPS_FP32 *1e12 )if estimated_flops >0 else 0
                            gpu_time_fp16 =estimated_flops /(RTX4080_TFLOPS_FP16 *1e12 )if estimated_flops >0 else 0


                            gpu_time_simple_fp32 =simple_estimated_flops /(RTX4080_TFLOPS_FP32 *1e12 )if simple_estimated_flops >0 else 0
                            gpu_time_simple_fp16 =simple_estimated_flops /(RTX4080_TFLOPS_FP16 *1e12 )if simple_estimated_flops >0 else 0


                            print ("\n============================================================")
                            print (f"[Measurement Result] Step {current_step} LP solve time: {duration:.6f} seconds")
                            if actual_num_vars !=num_vars or actual_num_constraints !=num_constraints :
                                print ("  Note: Actual problem size differs from estimation:")
                                print (f"    - Estimated: n={num_vars}, m={num_constraints}")
                                print (f"    - Actual: n={actual_num_vars}, m={actual_num_constraints}")
                            if duration >0 :
                                flops_per_sec =estimated_flops /duration
                                print (f"  - Effective FLOP/s (CPU, based on O(m^3)): {format_flops(flops_per_sec)}/s")
                            if duration >0 and gpu_time_fp32 >0 :
                                speedup_fp32 =duration /gpu_time_fp32
                                speedup_fp16 =duration /gpu_time_fp16
                                print (f"  - Theoretical speedup vs CPU (FP32): {speedup_fp32:.1f}x")
                                print (f"  - Theoretical speedup vs CPU (FP16): {speedup_fp16:.1f}x")
                            print ("============================================================")

                            try :
                                with open ("lp_step_time_measurement.csv","a",encoding ="utf-8")as f :

                                    file_exists =os .path .exists ("lp_step_time_measurement.csv")and os .path .getsize ("lp_step_time_measurement.csv")>0
                                    if not file_exists :
                                        f .write ("Timestamp,Step,CPUTime(sec),NumVariables,NumConstraints,TotalActiveEVs,StationsWithEVs,Horizon,EpisodeSteps,EstimatedIterations,EstimatedFLOPs_O(m3),SimpleEstimatedFLOPs_O(nm),ComplexityOrder,GPUTime_FP32_O(m3)(sec),GPUTime_FP16_O(m3)(sec),GPUTime_FP32_O(nm)(sec),GPUTime_FP16_O(nm)(sec),Speedup_FP32,Speedup_FP16\n")
                                    stations_str =str (num_stations_with_evs )if num_stations_with_evs is not None else ""
                                    horizon_str =str (horizon )if horizon is not None else ""
                                    episode_steps_str =str (episode_steps )if episode_steps is not None else ""
                                    gpu_time_fp32_str =f"{gpu_time_fp32:.9f}"if gpu_time_fp32 >0 else ""
                                    gpu_time_fp16_str =f"{gpu_time_fp16:.9f}"if gpu_time_fp16 >0 else ""
                                    gpu_time_simple_fp32_str =f"{gpu_time_simple_fp32:.9f}"if gpu_time_simple_fp32 >0 else ""
                                    gpu_time_simple_fp16_str =f"{gpu_time_simple_fp16:.9f}"if gpu_time_simple_fp16 >0 else ""
                                    speedup_fp32_str =f"{duration / gpu_time_fp32:.2f}"if duration >0 and gpu_time_fp32 >0 else ""
                                    speedup_fp16_str =f"{duration / gpu_time_fp16:.2f}"if duration >0 and gpu_time_fp16 >0 else ""
                                    f .write (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{current_step},{duration:.6f},{num_vars},{num_constraints},{total_active_evs},{stations_str},{horizon_str},{episode_steps_str},{estimated_iterations},{estimated_flops:.0f},{simple_estimated_flops:.0f},{complexity_order},{gpu_time_fp32_str},{gpu_time_fp16_str},{gpu_time_simple_fp32_str},{gpu_time_simple_fp16_str},{speedup_fp32_str},{speedup_fp16_str}\n")
                            except Exception as exc :
                                warnings .warn (f"Failed to write LP timing measurement: {exc}")

                            import sys
                            print ("Measurement completed. Exiting...")
                            sys .exit (0 )
                        else :

                            act_tensor =agent .act (obs1 ,env =env ,noise =False )
                    else :
                        act_tensor =agent .act (obs1 ,env =env ,noise =True )

                    # Training uses the lightweight info payload for speed.
                    # Evaluation/timing requests detailed snapshots and reward
                    # decomposition for plotting and diagnostics.
                    _build_info_full =bool (getattr (agent ,'test_mode',False ))or MEASURE_LP_STEP_TIME
                    _ ,r_local_tmp ,r_global_tmp ,done_tmp ,info_tmp =env .apply_action (act_tensor ,build_info =_build_info_full )

                    if all (done_tmp ):
                        next_state_tmp =normalize_observation (env ._get_obs ())
                        _next_prefetch_tmp =None
                    else :
                        _next_raw =env .begin_step ()
                        next_state_tmp =normalize_observation (_next_raw )
                        _next_prefetch_tmp =_next_raw

                    if not (is_safe (obs1 )and is_safe (next_state_tmp )and is_safe (act_tensor )and is_safe (r_local_tmp )and is_safe (r_global_tmp )):
                        retry_count +=1
                        print (f"Retry: Step data NaN/Inf detected (attempt {retry_count}/{MAX_RETRIES})")
                        continue

                    info =info_tmp
                    next_state =next_state_tmp
                    r_local =r_local_tmp
                    r_global =r_global_tmp
                    done =done_tmp
                    _prefetch_obs =_next_prefetch_tmp
                    step_success =True

                if not step_success :
                    print (f"Error: Step data retry failed after {MAX_RETRIES} attempts, stopping training")
                    agent .episode_end ()
                    break

                agent .update_active_evs (env )

                if torch .is_tensor (r_local ):
                    _r_local_sum =float (r_local .sum ().item ())
                    _r_local_mean =float (r_local .mean ().item ())
                else :
                    _r_local_sum =float (sum (r_local ))
                    _r_local_mean =float (np .mean (r_local ))
                ep_r +=_r_local_sum +float (r_global )
                ep_local_r +=_r_local_mean
                ep_global_r +=r_global

                if 'reward_breakdown'in info and 'per_station'in info ['reward_breakdown']:
                    for st_idx ,station_data in enumerate (info ['reward_breakdown']['per_station']):

                        dep_r =station_data .get ('departure_reward',0.0 )
                        shaping_r =station_data .get ('progress_shaping',0.0 )
                        discharge_penalty_r =station_data .get ('discharge_penalty',0.0 )
                        switch_penalty_r =station_data .get ('switch_penalty',0.0 )
                        station_limit_penalty_r =station_data .get ('station_limit_penalty',0.0 )
                        total_r =station_data .get ('local_total',0.0 )

                        ep_local_departure_r +=dep_r
                        ep_local_progress_shaping_r +=shaping_r
                        ep_local_discharge_penalty_r +=discharge_penalty_r
                        ep_local_switch_penalty_r +=switch_penalty_r
                        ep_local_station_limit_penalty_r +=station_limit_penalty_r

                        if 0 <=st_idx <len (station_local_reward_sums ):
                            sums =station_local_reward_sums [st_idx ]
                            sums ["total"]+=total_r
                            sums ["departure"]+=dep_r
                            sums ["progress_shaping"]+=shaping_r
                            sums ["discharge_penalty"]+=discharge_penalty_r
                            sums ["switch_penalty"]+=switch_penalty_r
                            sums ["station_limit_penalty"]+=station_limit_penalty_r


                if hasattr (agent ,'cache_experience'):
                    # Replay transition includes both station-power totals and
                    # per-EV realized power so local/global critics can learn
                    # from physically clipped actions rather than raw actor
                    # outputs.
                    state_tensor_for_buffer =torch .as_tensor (obs1 ,dtype =torch .float32 ,device =device )
                    next_state_tensor =torch .as_tensor (next_state ,dtype =torch .float32 ,device =device )

                    actual_ev_power_kw_tensor =info .get ('actual_ev_power_kw')
                    _r_local_t =r_local if torch .is_tensor (r_local )else torch .as_tensor (r_local ,dtype =torch .float32 ,device =device )
                    _sp =info ['station_powers']
                    _sp_t =_sp if torch .is_tensor (_sp )else torch .as_tensor (_sp ,dtype =torch .float32 ,device =device )

                    agent .cache_experience (
                    state_tensor_for_buffer ,
                    next_state_tensor ,
                    act_tensor ,
                    _r_local_t ,
                    torch .tensor (r_global ,dtype =torch .float32 ,device =device ),
                    torch .as_tensor (done ,dtype =torch .float32 ,device =device ),
                    actual_station_powers =_sp_t ,
                    actual_ev_power_kw =actual_ev_power_kw_tensor ,
                    )


                if hasattr (agent ,'actors')and hasattr (agent ,'critics')and not all (done ):
                    agent .update ()
                else :

                    if hasattr (agent ,'update'):
                        agent .update ()


                # Agents expose their latest Q/gradient/loss/clipping values;
                # the visualizer averages them over the episode for TensorBoard.
                if hasattr (agent ,'last_central_q_value'):
                    visualizer .update_central_q_value (agent .last_central_q_value )
                elif hasattr (agent ,'last_local_q_values_per_agent')and hasattr (agent ,'last_global_q_value'):
                    visualizer .update_q_values (
                    agent .last_local_q_values_per_agent ,
                    np .mean (agent .last_local_q_values_per_agent )if agent .last_local_q_values_per_agent else 0.0 ,
                    agent .last_global_q_value
                    )
                else :
                    raise AttributeError (
                    "Agent does not expose expected Q diagnostics "
                    "(last_central_q_value or last_local_q_values_per_agent + last_global_q_value)."
                    )

                visualizer .update_gradients (agent )
                visualizer .update_losses (agent )
                visualizer .update_clipping (agent )

                episode_data ['ag_requests'].append (info ['net_demand'])

                if 'pre_total_ev_transport'not in episode_data :
                    episode_data ['pre_total_ev_transport']=[]
                    for i in range (1 ,env .num_stations +1 ):
                        episode_data [f'pre_ev{i}']=[]
                if 'actual_ev_power_kw'not in info :
                    raise KeyError ("info must include 'actual_ev_power_kw'")
                ev_changes_tensor =info ['actual_ev_power_kw']

                station_sums =ev_changes_tensor .sum (dim =1 ).detach ().cpu ().tolist ()
                pre_total =0.0
                for station_idx ,station_sum in enumerate (station_sums ):
                    episode_data [f'pre_ev{station_idx+1}'].append (float (station_sum ))
                    pre_total +=float (station_sum )
                episode_data ['pre_total_ev_transport'].append (pre_total )

                _tev =info ['total_ev_transport']
                transport =float (_tev .item ()if torch .is_tensor (_tev )else _tev )
                _sp =info ['station_powers']
                if torch .is_tensor (_sp ):
                    _sp_list =_sp .detach ().cpu ().tolist ()
                else :
                    _sp_list =_sp

                episode_data ['total_ev_transport'].append (transport )

                request =info ['net_demand']
                if request >0 :
                    total_surplus_available +=request
                    total_surplus_absorbed +=min (request ,transport )
                elif request <0 :
                    total_discharge_request +=abs (request )
                    total_discharge_fulfilled +=min (abs (request ),transport )

                for i in range (env .num_stations ):
                    if f'actual_ev{i+1}'not in episode_data :
                        episode_data [f'actual_ev{i+1}']=[]
                    episode_data [f'actual_ev{i+1}'].append (_sp_list [i ])


                if 'departed_evs'in info and info ['departed_evs']:
                    for departed_ev in info ['departed_evs']:
                        ev_id =str (departed_ev .get ('id'))
                        station_idx =departed_ev .get ('station')
                        if station_idx is None :
                            continue

                        if f'station{station_idx+1}'not in episode_data ['soc_data']:
                            episode_data ['soc_data'][f'station{station_idx+1}']={}
                        if ev_id in episode_data ['soc_data'][f'station{station_idx+1}']:

                            after_list =info .get ('snapshot_after',{}).get (station_idx ,[])
                            matched =next ((d for d in after_list if str (d .get ('id'))==ev_id ),None )
                            if matched is not None :
                                final_soc =matched .get ('new_soc',None )
                                target_soc =matched .get ('target_soc',None )
                                if target_soc is None :
                                    raise ValueError (f"EV {ev_id}: target_soc not found in snapshot data. Cannot proceed without target SoC information.")
                                if final_soc is not None :
                                    episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['final_soc']=float (final_soc )
                                if target_soc is not None :
                                    episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['target_soc']=float (target_soc )

                            episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['depart_step']=int (info .get ('step_count',env .step_count ))

                if all (done ):
                    agent .episode_end ()
                    break



            ep_end_time =time .time ()
            ep_duration =ep_end_time -ep_start_time


            steps_in_ep =env .step_count if hasattr (env ,'step_count')and env .step_count >0 else 1



            _is_warmup =False

            metrics =env .get_metrics ()
            soc_miss_rate =metrics ['soc_miss_rate']
            surplus_absorption_rate =metrics ['surplus_absorption_rate']
            supply_cooperation_rate =metrics ['supply_cooperation_rate']
            surplus_steps =metrics ['surplus_steps']
            surplus_success =metrics ['surplus_within_narrow']
            shortage_steps =metrics ['shortage_steps']
            shortage_success =metrics ['shortage_within_narrow']
            departing_evs_total =env .metrics .get ('departing_evs',0 )
            departing_evs_soc_met =env .metrics .get ('departing_evs_soc_met',0 )
            avg_soc_deficit =metrics .get ('avg_soc_deficit',0.0 )
            avg_switches =metrics .get ('avg_switches',0.0 )
            station_limit_hits =metrics .get ('station_limit_hits',0 )
            station_limit_steps =metrics .get ('station_limit_steps',0 )
            station_charge_limit_hits =metrics .get ('station_charge_limit_hits',0 )
            station_discharge_limit_hits =metrics .get ('station_discharge_limit_hits',0 )
            station_limit_penalty_total =metrics .get ('station_limit_penalty_total',0.0 )
            station_limit_penalty_per_step =(
            station_limit_penalty_total /max (steps_in_ep ,1 )
            )
            station_limit_penalty_per_hit =(
            station_limit_penalty_total /max (station_limit_hits ,1 )
            if station_limit_hits >0 else 0.0
            )

            if not _is_warmup :
                all_rewards .append (ep_r /steps_in_ep )
                all_local_rewards .append (ep_local_r /steps_in_ep )
                all_global_rewards .append (ep_global_r /steps_in_ep )

                if not hasattr (train ,'all_local_rewards'):
                    train .all_local_rewards =[]
                    train .all_global_rewards =[]
                    train .charge_rates =[]
                    train .discharge_rates =[]
                    train .soc_hit_rates =[]
                train .all_local_rewards .append (ep_local_r /steps_in_ep )
                train .all_global_rewards .append (ep_global_r /steps_in_ep )

                performance_metrics ['soc_miss_count'].append (soc_miss_rate )
                performance_metrics ['avg_soc_deficit'].append (avg_soc_deficit )
                performance_metrics ['surplus_absorption_rate'].append (surplus_absorption_rate )
                performance_metrics ['supply_cooperation_rate'].append (supply_cooperation_rate )
                if 'avg_switches'in performance_metrics :
                    performance_metrics ['avg_switches'].append (avg_switches )
                performance_metrics ['departing_evs'].append (departing_evs_total )
                performance_metrics ['departing_evs_soc_met'].append (departing_evs_soc_met )
                performance_metrics ['surplus_steps'].append (surplus_steps )
                performance_metrics ['surplus_within_narrow'].append (surplus_success )
                performance_metrics ['shortage_steps'].append (shortage_steps )
                performance_metrics ['shortage_within_narrow'].append (shortage_success )
                if 'station_limit_hits'in performance_metrics :
                    performance_metrics ['station_limit_hits'].append (station_limit_hits )
                    performance_metrics ['station_limit_steps'].append (station_limit_steps )
                    performance_metrics ['station_charge_limit_hits'].append (station_charge_limit_hits )
                    performance_metrics ['station_discharge_limit_hits'].append (station_discharge_limit_hits )
                    performance_metrics ['station_limit_penalty_total'].append (station_limit_penalty_total )
                    performance_metrics ['station_limit_penalty_per_step'].append (station_limit_penalty_per_step )
                    performance_metrics ['station_limit_penalty_per_hit'].append (station_limit_penalty_per_hit )

                train .charge_rates .append (surplus_absorption_rate )
                train .discharge_rates .append (supply_cooperation_rate )
                train .soc_hit_rates .append (100 -soc_miss_rate )

                all_episode_data [ep ]=episode_data

            train_parts =[
            f"train{training_ep} ",
            f"SoC hit: {100-soc_miss_rate:.1f}%",
            ]
            if enable_switch_metrics :
                train_parts .append (f"Switches: {avg_switches:.2f}")
            if enable_stlimit_metrics :
                train_parts .append (
                f"StLimit: steps={station_limit_steps}, hits={station_limit_hits}, pen={station_limit_penalty_total:.2f}"
                )
            train_parts .append (f"Surplus: {surplus_success}/{surplus_steps} ({surplus_absorption_rate:.1f}%)")
            train_parts .append (f"Supply: {shortage_success}/{shortage_steps} ({supply_cooperation_rate:.1f}%)")
            train_parts .append (f"Duration={ep_duration:.1f}s")
            print (" | ".join (train_parts ),flush =True )

            # ---------------------------------------------------------------
            # ---------------------------------------------------------------
            if tb_writer and not _is_warmup :





                write_train_episode_tb_scalars (
                tb_writer ,training_ep ,steps_in_ep ,
                ep_local_r =ep_local_r ,
                ep_global_r =ep_global_r ,
                soc_miss_rate =soc_miss_rate ,
                surplus_absorption_rate =surplus_absorption_rate ,
                supply_cooperation_rate =supply_cooperation_rate ,
                avg_switches =avg_switches ,
                station_limit_steps =station_limit_steps ,
                station_limit_penalty_total =station_limit_penalty_total ,
                ep_local_departure_r =ep_local_departure_r ,
                ep_local_progress_shaping_r =ep_local_progress_shaping_r ,
                ep_local_discharge_penalty_r =ep_local_discharge_penalty_r ,
                ep_local_switch_penalty_r =ep_local_switch_penalty_r ,
                ep_local_station_limit_penalty_r =ep_local_station_limit_penalty_r ,
                station_local_reward_sums =station_local_reward_sums ,
                enable_switch_metrics =enable_switch_metrics ,
                enable_stlimit_metrics =enable_stlimit_metrics ,
                )


            if _interrupt_handler .is_interrupted ():
                break

            if tb_writer and not _is_warmup :
                visualizer .record_to_tensorboard (training_ep )
                visualizer .record_agent_state (agent ,training_ep )

            visualizer .reset_episode_data ()



            # Periodic deterministic evaluation on held-out demand data. Actor
            # snapshots are saved next to the TEST* artifact folder so each row
            # in test history is traceable to a loadable checkpoint.

            should_run_interim_csv =(not _is_warmup )and (training_ep %INTERIM_TEST_INTERVAL ==0 )
            should_save_interim_graph =(not _is_warmup )and (training_ep %INTERIM_PLOT_INTERVAL ==0 )

            if should_run_interim_csv :
                print (f"------------test{training_ep}")
                test_start =time .time ()

                _ =test (agent ,random_window =False ,working_dir =working_dir ,
                test_results =None ,test_episode_num =training_ep ,enable_png =should_save_interim_graph ,
                save_test_detail_files =should_run_interim_csv )

                try :
                    save_dir =os .path .join (working_dir ,"results",f"TEST{training_ep}")
                    os .makedirs (save_dir ,exist_ok =True )
                    if hasattr (agent ,"save_actors"):
                        agent .save_actors (save_dir ,episode =training_ep )
                    elif hasattr (agent ,"save_models"):
                        agent .save_models (save_dir ,episode =training_ep )
                except Exception as exc :
                    warnings .warn (f"Failed to save checkpoint actors at episode {training_ep}: {exc}")
                test_duration =time .time ()-test_start
                print (f"test_done{training_ep} ({test_duration:.1f}s)")
                print ("======================================")

                try :
                    interim_results_dir =os .path .join (working_dir ,"results")
                    os .makedirs (interim_results_dir ,exist_ok =True )
                    # Root-level train summaries must stay aligned with the CSV
                    # written by the same helper. Detailed/interim plots are
                    # still gated by should_save_interim_graph in test().
                    skip_png_flag =False

                    if len (all_local_rewards )>0 and len (all_global_rewards )>0 :
                        plot_daily_rewards (all_local_rewards ,all_global_rewards ,
                        interim_results_dir ,episode_num =len (all_local_rewards ),
                        performance_metrics =performance_metrics ,title_prefix ="Train Results",
                        skip_png =skip_png_flag )

                    if performance_metrics and len (performance_metrics .get ('soc_miss_count',[]))>0 :
                        plot_performance_metrics (performance_metrics ,interim_results_dir ,title_prefix ="Train Results",
                        skip_png =skip_png_flag )
                except Exception as exc :
                    warnings .warn (f"Failed to write interim training plots at episode {training_ep}: {exc}")

    except Exception as e :
        print (f"Training interrupted at episode {len(all_rewards)}: {e}")
        traceback .print_exc ()

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    results_dir =os .path .join (working_dir ,"results")
    os .makedirs (results_dir ,exist_ok =True )

    skip_png_flag =False
    if len (all_local_rewards )>0 and len (all_global_rewards )>0 :
        plot_daily_rewards (all_local_rewards ,all_global_rewards ,
        results_dir ,episode_num =len (all_local_rewards ),
        performance_metrics =performance_metrics ,title_prefix ="Train Results",
        skip_png =skip_png_flag )

    if performance_metrics and len (performance_metrics .get ('soc_miss_count',[]))>0 :
        plot_performance_metrics (performance_metrics ,results_dir ,title_prefix ="Train Results",
        skip_png =skip_png_flag )

    return agent ,all_rewards ,performance_metrics ,all_episode_data ,working_dir


if __name__ =="__main__":


     ag ,all_rewards ,perf ,ep_data ,work_dir =train ()
