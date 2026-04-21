"""
テスト（評価）専用モジュール。

学習済みエージェントを複数エピソード実行してSoC達成率・需給追従率等を評価し、
結果をtest_history.jsonに累積保存するとともにグラフ・CSVを出力する。
"""

import os
import csv
import json
import numpy as np
import torch
import random
import time
import traceback
import warnings
import logging

from environment.readcsv import load_multiple_demand_files
from environment.EVEnv import EVEnv
from environment.normalize import normalize_observation
from tools.Utils import (
plot_daily_rewards ,
plot_station_cooperation_full ,
plot_ev_detailed_soc ,
plot_performance_metrics ,
plot_arrival_counts ,
plot_reward_breakdown ,
plot_power_mismatch_analysis ,
)
from Config import (
NUM_EPISODES ,EPISODE_STEPS ,NUM_EVS ,NUM_STATIONS ,
ENV_SEED ,USE_MILP ,
USE_SWITCHING_CONSTRAINTS ,LOCAL_SWITCH_PENALTY ,
USE_STATION_TOTAL_POWER_LIMIT ,LOCAL_STATION_LIMIT_PENALTY ,
)


def set_env_seed (seed =None ):
    """環境・各ライブラリの乱数シードを一括設定する。"""
    if seed is not None :
        random .seed (seed )
        np .random .seed (seed )
        torch .manual_seed (seed )
        if torch .cuda .is_available ():
            torch .cuda .manual_seed_all (seed )


def test (agent ,demand_adjustment =None ,random_window =True ,working_dir =None ,test_results =None ,test_episode_num =None ,dataset_dir =None ,num_episodes =None ,enable_png =True ,save_test_detail_files =True ):
    """
    学習済みエージェントを複数エピソード実行して性能を評価する。

    - agent.set_test_mode(True/False) で探索ノイズをOFF/ON切替
    - SoC達成率・需給追従率・スイッチ数等を集計して返す
    - test_history.json に結果を累積（training_ep をキーとした時系列）
    - グラフ（PNG）と CSV（x軸は training_ep 値）を出力
    """
    # フラグ類を事前に計算（finally節でも参照するため関数冒頭で定義）
    # スイッチング制約が有効か、またはペナルティが設定されている場合にTrueになる
    enable_switch_metrics =bool (USE_SWITCHING_CONSTRAINTS )or float (LOCAL_SWITCH_PENALTY )!=0.0
    # ステーション電力制限が有効か、またはペナルティが設定されている場合にTrueになる
    enable_stlimit_metrics =bool (USE_STATION_TOTAL_POWER_LIMIT )or float (LOCAL_STATION_LIMIT_PENALTY )!=0.0

    # 乱数シードを固定して再現性を担保
    set_env_seed (ENV_SEED )

    # 出力先ディレクトリのデフォルトはカレントディレクトリ
    if working_dir is None :
        working_dir =os .getcwd ()

    # テスト用需要データを読み込む（train_split=25 → 上位25%をテストに使用）
    demand_data_test =None
    try :
        if dataset_dir is not None :
            # 指定ディレクトリから需要CSVを読み込む
            all_demand_data =load_multiple_demand_files (directory =dataset_dir ,train_split =25 )
            demand_data_test =all_demand_data ['test']
        else :
            # デフォルトディレクトリから需要CSVを読み込む
            all_demand_data =load_multiple_demand_files (train_split =25 )
            demand_data_test =all_demand_data ['test']
        if demand_data_test :
            all_test_data =np .concatenate (demand_data_test )
    except FileNotFoundError as e :
        error_message ="Failed to load test demand CSV files."
        print (error_message )
        raise FileNotFoundError (error_message )


    # 評価用環境を生成（学習時と同じパラメータ）
    env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )

    # テストモードON: 探索ノイズを無効化
    agent .set_test_mode (True )

    # TensorBoardへの書き込みをテスト中は一時的に無効化
    original_use_tensorboard =None
    if hasattr (agent ,'use_tensorboard'):
        original_use_tensorboard =agent .use_tensorboard
        agent .use_tensorboard =False

    # テスト結果の蓄積コンテナを初期化（引数で渡された場合はそちらを使う）
    if test_results is None :
        test_results ={
        'all_rewards':[],
        'all_local_rewards':[],
        'all_global_rewards':[],
        'all_episode_data':{},
        'performance_metrics':{
        'soc_miss_count':[],
        'surplus_absorption_rate':[],
        'supply_cooperation_rate':[],
        'departing_evs':[],
        'departing_evs_soc_met':[],
        'surplus_steps':[],
        'surplus_within_narrow':[],
        'shortage_steps':[],
        'shortage_within_narrow':[],
        }
        }
        if enable_switch_metrics :
            test_results ['performance_metrics']['avg_switches']=[]
        if enable_stlimit_metrics :
            test_results ['performance_metrics']['station_limit_hits']=[]
            test_results ['performance_metrics']['station_limit_steps']=[]
            test_results ['performance_metrics']['station_charge_limit_hits']=[]
            test_results ['performance_metrics']['station_discharge_limit_hits']=[]
            test_results ['performance_metrics']['station_limit_penalty_total']=[]

    # エピソードごとのEV到着SoCや在室時間の統計を蓄積するリスト
    episode_soc_stats =[]


    def run_single_test_episode (ep_idx ):
        """1エピソード分のテストを実行してすべての記録を返す。"""

        # スレッドセーフのためエピソードごとに独立した環境インスタンスを作成
        local_env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )
        # スナップショット記録を有効化（SoC推移の追跡に使用）
        local_env .record_snapshots =True

        # エージェントは共有（テストモードのため並列更新はなし）
        local_agent =agent

        # 需要データのインデックスをエピソード番号からサイクリックに取得
        if not demand_data_test :
            raise RuntimeError ("Error: invalid runtime state.")
        data_idx =(ep_idx -1 )%len (demand_data_test )
        _data =np .asarray (demand_data_test [data_idx ],dtype =float ).reshape (-1 )
        if _data .size ==0 :
            raise ValueError ("Error: invalid runtime state.")
        # ステップ数に合わせてトリミングまたはゼロパディング
        if _data .size >=int (local_env .episode_steps ):
            episode_demand =_data [:int (local_env .episode_steps )]
        else :
            episode_demand =np .pad (_data ,(0 ,int (local_env .episode_steps )-int (_data .size )))
        local_env .reset (net_demand_series =episode_demand )

        local_agent .episode_start ()
        local_agent .update_active_evs (local_env )

        # エピソードデータの記録用辞書を初期化
        ep_data ={
        'ag_requests':[],'total_ev_transport':[],'soc_data':{},'power_mismatch':[],
        'arrivals_per_step':[],'rewards_global_balance':[],'rewards_local_shaping':[],
        'rewards_local_departure':[],'rewards_local_discharge_penalty':[],
        'rewards_local_switch_penalty':[],
        'rewards_local_station_limit_penalty':[],
        }

        # エピソード開始時点でのEV初期SoCを記録
        for station_idx in range (local_env .num_stations ):
            ep_data ['soc_data'][f'station{station_idx+1}']={}
            for ev_idx ,ev in enumerate (local_env .stations_evs [station_idx ]):
                ev_id =str (ev ['id'])
                ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]={
                'id':ev ['id'],
                'station':station_idx ,
                'depart':ev ['depart'],
                'target':ev ['target'],
                'times':[int (local_env .step_count )],
                'soc':[float (ev ['soc'])]
                }

        # ステーションごとの実績電力・計画電力を格納するリストを準備
        for i in range (1 ,local_env .num_stations +1 ):ep_data [f'actual_ev{i}']=[]
        for i in range (1 ,local_env .num_stations +1 ):ep_data [f'pre_ev{i}']=[]
        ep_data ['pre_total_ev_transport']=[]

        # エピソード累積報酬（合計・ローカル平均・グローバル）
        ep_r ,ep_local_r ,ep_global_r =0.0 ,0.0 ,0.0

        while True :
            obs =local_env .begin_step ()
            local_agent .update_active_evs (local_env )
            obs =normalize_observation (obs )
            act =local_agent .act (obs ,env =local_env ,noise =False )
            _ ,r_l ,r_g ,done ,info =local_env .apply_action (act )

            
            ep_r +=sum (r_l )+r_g 
            ep_local_r +=np .mean (r_l )
            ep_global_r +=r_g 

            ep_data ['ag_requests'].append (info ['net_demand'])
            ep_data ['total_ev_transport'].append (info ['total_ev_transport'])
            ep_data ['power_mismatch'].append (info ['net_demand']-info ['total_ev_transport'])
            
            
            ep_data ['arrivals_per_step'].append (
            info .get ('arrivals_by_station',info .get ('arrivals_this_step',0 ))
            )

            
            current_step =int (info .get ('step_count',local_env .step_count ))
            for station_idx in range (local_env .num_stations ):
                if f'station{station_idx+1}'not in ep_data ['soc_data']:
                    ep_data ['soc_data'][f'station{station_idx+1}']={}
                for ev_idx ,ev in enumerate (local_env .stations_evs [station_idx ]):
                    ev_id =str (ev ['id'])
                    if ev_id not in ep_data ['soc_data'][f'station{station_idx+1}']:
                        ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]={
                        'id':ev ['id'],
                        'station':station_idx ,
                        'depart':ev ['depart'],
                        'target':ev ['target'],
                        'times':[current_step ],
                        'soc':[float (ev ['soc'])]
                        }
                    else :
                        ev_times =ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['times']
                        if len (ev_times )>0 and ev_times [-1 ]==current_step :
                            ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['soc'][-1 ]=float (ev ['soc'])
                        else :
                            ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['times'].append (current_step )
                            ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['soc'].append (float (ev ['soc']))

                            
            if 'departed_evs'in info and info ['departed_evs']:
                for departed_ev in info ['departed_evs']:
                    ev_id =str (departed_ev .get ('id'))
                    station_idx =departed_ev .get ('station')
                    if station_idx is not None and ev_id in ep_data ['soc_data'][f'station{station_idx+1}']:
                        after_list =info .get ('snapshot_after',{}).get (station_idx ,[])
                        matched =next ((d for d in after_list if str (d .get ('id'))==ev_id ),None )
                        if matched is not None :
                            ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['final_soc']=float (matched .get ('new_soc',0.0 ))
                            ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['target_soc']=float (matched .get ('target_soc',0.0 ))
                        ep_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['depart_step']=current_step 

            if 'reward_breakdown'in info :
                rb =info ['reward_breakdown']
                ep_data ['rewards_global_balance'].append (rb ['global'].get ('balance_reward',0.0 ))
                ep_data ['rewards_local_shaping'].append (sum (s .get ('progress_shaping',0.0 )for s in rb ['per_station']))
                ep_data ['rewards_local_departure'].append (sum (s .get ('departure_reward',0.0 )for s in rb ['per_station']))
                ep_data ['rewards_local_discharge_penalty'].append (sum (s .get ('discharge_penalty',0.0 )for s in rb ['per_station']))
                ep_data ['rewards_local_switch_penalty'].append (sum (s .get ('switch_penalty',0.0 )for s in rb ['per_station']))
                ep_data ['rewards_local_station_limit_penalty'].append (sum (s .get ('station_limit_penalty',0.0 )for s in rb ['per_station']))

            ev_changes_tensor =info ['actual_ev_power_kw']
            
            station_sums =ev_changes_tensor .sum (dim =1 ).detach ().cpu ().tolist ()
            pre_total =0.0 
            for st_idx in range (local_env .num_stations ):
                s_sum =float (station_sums [st_idx ])
                ep_data [f'pre_ev{st_idx+1}'].append (s_sum )
                ep_data [f'actual_ev{st_idx+1}'].append (info ['station_powers'][st_idx ])
                pre_total +=s_sum 
            ep_data ['pre_total_ev_transport'].append (pre_total )

            if all (done ):
                break 

        local_agent .episode_end ()
        metrics =local_env .get_metrics ()

        
        arrival_log =np .asarray (getattr (local_env ,'arrival_soc_log',[]),dtype =np .float64 )
        needed_log =np .asarray (getattr (local_env ,'arrival_needed_log',[]),dtype =np .float64 )
        dwell_log =np .asarray (getattr (local_env ,'arrival_dwell_log',[]),dtype =np .float64 )

        def _stats (arr :np .ndarray ):
            if arr .size ==0 :return 0.0 ,0.0 ,0.0 ,0.0 
            return float (arr .mean ()),float (arr .max ()),float (arr .min ()),float (arr .std (ddof =0 ))

        arrival_mean ,arrival_max ,arrival_min ,arrival_std =_stats (arrival_log )
        needed_mean ,needed_max ,needed_min ,needed_std =_stats (needed_log )
        dwell_mean ,dwell_max ,dwell_min ,dwell_std =_stats (dwell_log )

        soc_stat ={
        'episode':ep_idx ,
        'arrival_mean':arrival_mean ,'arrival_max':arrival_max ,'arrival_min':arrival_min ,'arrival_std':arrival_std ,
        'needed_mean':needed_mean ,'needed_max':needed_max ,'needed_min':needed_min ,'needed_std':needed_std ,
        'dwell_mean':dwell_mean ,'dwell_max':dwell_max ,'dwell_min':dwell_min ,'dwell_std':dwell_std ,
        'count':int (arrival_log .size ),
        }

        return ep_idx ,ep_r ,ep_local_r ,ep_global_r ,ep_data ,metrics ,local_env .step_count ,soc_stat 

    try :
    
    
        if num_episodes is not None :
            num_test_episodes =int (num_episodes )
        else :
            num_test_episodes =min (5 ,len (demand_data_test ))if demand_data_test else 0 
        if num_test_episodes <=0 :
            raise RuntimeError ("Error: invalid runtime state.")

        all_results =[]

        
        for ep in range (1 ,num_test_episodes +1 ):
            t_start =time .time ()
            res =run_single_test_episode (ep )
            duration =time .time ()-t_start 
            all_results .append (res )

            
            ep_idx ,ep_r ,ep_local_r ,ep_global_r ,_ ,metrics ,steps_in_ep ,_ =res 
            parts =[
            f"test{ep_idx}",
            f"SoC hit: {100-metrics['soc_miss_rate']:.1f}%",
            f"Avg SoC deficit: {metrics.get('avg_soc_deficit', 0.0):.2f} kWh",
            ]
            if enable_switch_metrics :
                parts .append (f"Avg Switches: {metrics.get('avg_switches', 0.0):.2f}")
            if enable_stlimit_metrics :
                parts .append (
                f"StLimit: steps={metrics.get('station_limit_steps', 0)}, hits={metrics.get('station_limit_hits', 0)}, pen={metrics.get('station_limit_penalty_total', 0.0):.2f}"
                )
            parts .append (
            f"Surplus: {metrics['surplus_within_narrow']}/{metrics['surplus_steps']} ({metrics['surplus_absorption_rate']:.1f}%)"
            )
            parts .append (
            f"Supply: {metrics['shortage_within_narrow']}/{metrics['shortage_steps']} ({metrics['supply_cooperation_rate']:.1f}%)"
            )
            parts .append (f"Duration={duration:.1f}s")
            print (" | ".join (parts ))

            
        for res in all_results :
            ep_idx ,ep_r ,ep_local_r ,ep_global_r ,ep_data ,metrics ,steps_in_ep ,soc_stat =res 

            test_results ['all_rewards'].append (ep_r /steps_in_ep )
            test_results ['all_local_rewards'].append (ep_local_r /steps_in_ep )
            test_results ['all_global_rewards'].append (ep_global_r /steps_in_ep )
            test_results ['all_episode_data'][ep_idx ]=ep_data 

            pm =test_results ['performance_metrics']
            pm ['soc_miss_count'].append (metrics ['soc_miss_rate'])
            if enable_switch_metrics :
                pm ['avg_switches']=pm .get ('avg_switches',[])
                pm ['avg_switches'].append (metrics .get ('avg_switches',0.0 ))
            pm ['avg_soc_deficit']=pm .get ('avg_soc_deficit',[])
            pm ['avg_soc_deficit'].append (metrics .get ('avg_soc_deficit',0.0 ))
            pm ['surplus_absorption_rate'].append (metrics ['surplus_absorption_rate'])
            pm ['supply_cooperation_rate'].append (metrics ['supply_cooperation_rate'])
            pm ['departing_evs'].append (metrics .get ('departing_evs',0 ))
            pm ['departing_evs_soc_met'].append (metrics .get ('departing_evs_soc_met',0 ))
            pm ['surplus_steps'].append (metrics ['surplus_steps'])
            pm ['surplus_within_narrow'].append (metrics ['surplus_within_narrow'])
            pm ['shortage_steps'].append (metrics ['shortage_steps'])
            pm ['shortage_within_narrow'].append (metrics ['shortage_within_narrow'])
            if enable_stlimit_metrics :
                pm .setdefault ('station_limit_hits',[]).append (metrics .get ('station_limit_hits',0 ))
                pm .setdefault ('station_limit_steps',[]).append (metrics .get ('station_limit_steps',0 ))
                pm .setdefault ('station_charge_limit_hits',[]).append (metrics .get ('station_charge_limit_hits',0 ))
                pm .setdefault ('station_discharge_limit_hits',[]).append (metrics .get ('station_discharge_limit_hits',0 ))
                pm .setdefault ('station_limit_penalty_total',[]).append (metrics .get ('station_limit_penalty_total',0.0 ))

            episode_soc_stats .append (soc_stat )

    except Exception as e :
        traceback .print_exc ()

    finally :
    
        if save_test_detail_files :
            if test_episode_num is None :
                raise ValueError ("test_episode_num is required for deterministic result directory naming.")
            results_dir =os .path .join (working_dir ,'results',f'TEST{test_episode_num}')
        else :
            results_dir =os .path .join (working_dir ,'results')
        os .makedirs (results_dir ,exist_ok =True )

        if save_test_detail_files and episode_soc_stats :
            soc_summary_path =os .path .join (results_dir ,'episode_soc_summary.csv')
            try :
                with open (soc_summary_path ,'w',newline ='',encoding ='utf-8')as csvfile :
                    writer =csv .writer (csvfile )
                    writer .writerow ([
                    'episode',
                    'arrival_mean',
                    'arrival_max',
                    'arrival_min',
                    'arrival_std',
                    'needed_mean',
                    'needed_max',
                    'needed_min',
                    'needed_std',
                    'dwell_hours_mean',
                    'dwell_hours_max',
                    'dwell_hours_min',
                    'dwell_hours_std',
                    'num_records',
                    ])
                    for stat in episode_soc_stats :
                        writer .writerow ([
                        stat ['episode'],
                        f"{stat['arrival_mean']:.4f}",
                        f"{stat['arrival_max']:.4f}",
                        f"{stat['arrival_min']:.4f}",
                        f"{stat['arrival_std']:.4f}",
                        f"{stat['needed_mean']:.4f}",
                        f"{stat['needed_max']:.4f}",
                        f"{stat['needed_min']:.4f}",
                        f"{stat['needed_std']:.4f}",
                        f"{stat['dwell_mean']:.4f}",
                        f"{stat['dwell_max']:.4f}",
                        f"{stat['dwell_min']:.4f}",
                        f"{stat['dwell_std']:.4f}",
                        stat ['count'],
                        ])
                        
            except Exception as csv_error :
                pass

                
                
        try :
            import numpy as _np 
            if test_results ['all_local_rewards']:
                avg_local =float (_np .mean (test_results ['all_local_rewards']))
            else :
                avg_local =0.0 
            if test_results ['all_global_rewards']:
                avg_global =float (_np .mean (test_results ['all_global_rewards']))
            else :
                avg_global =0.0 

            pm =test_results ['performance_metrics']
            avg_pm ={
            'soc_miss_count':[float (_np .mean (pm ['soc_miss_count']))]if pm .get ('soc_miss_count')else [],
            'surplus_absorption_rate':[float (_np .mean (pm ['surplus_absorption_rate']))]if pm .get ('surplus_absorption_rate')else [],
            'supply_cooperation_rate':[float (_np .mean (pm ['supply_cooperation_rate']))]if pm .get ('supply_cooperation_rate')else [],
            'departing_evs':[int (_np .sum (pm ['departing_evs']))]if pm .get ('departing_evs')else [],
            'departing_evs_soc_met':[int (_np .sum (pm ['departing_evs_soc_met']))]if pm .get ('departing_evs_soc_met')else [],
            'surplus_steps':[int (_np .sum (pm ['surplus_steps']))]if pm .get ('surplus_steps')else [],
            'surplus_within_narrow':[int (_np .sum (pm ['surplus_within_narrow']))]if pm .get ('surplus_within_narrow')else [],
            'shortage_steps':[int (_np .sum (pm ['shortage_steps']))]if pm .get ('shortage_steps')else [],
            'shortage_within_narrow':[int (_np .sum (pm ['shortage_within_narrow']))]if pm .get ('shortage_within_narrow')else [],
            }
            if enable_switch_metrics :
                avg_pm ['avg_switches']=[float (_np .mean (pm ['avg_switches']))]if pm .get ('avg_switches')else []
            if enable_stlimit_metrics :
                avg_pm ['station_limit_hits']=[int (_np .sum (pm ['station_limit_hits']))]if pm .get ('station_limit_hits')else []
                avg_pm ['station_limit_steps']=[int (_np .sum (pm ['station_limit_steps']))]if pm .get ('station_limit_steps')else []
                avg_pm ['station_charge_limit_hits']=[int (_np .sum (pm ['station_charge_limit_hits']))]if pm .get ('station_charge_limit_hits')else []
                avg_pm ['station_discharge_limit_hits']=[int (_np .sum (pm ['station_discharge_limit_hits']))]if pm .get ('station_discharge_limit_hits')else []
                avg_pm ['station_limit_penalty_total']=[float (_np .sum (pm ['station_limit_penalty_total']))]if pm .get ('station_limit_penalty_total')else []
        except Exception :
            avg_local =avg_global =0.0 
            avg_pm ={'soc_miss_count':[],'surplus_absorption_rate':[],'supply_cooperation_rate':[]}

            
        skip_png_flag =bool (not enable_png )
        plot_daily_rewards ([avg_local ],[avg_global ],
        results_dir ,episode_num =1 ,
        performance_metrics =avg_pm ,title_prefix ="Test Results",
        skip_png =skip_png_flag )

        
        plot_performance_metrics (avg_pm ,results_dir ,title_prefix ="Test Results",
        skip_png =skip_png_flag )

        if (not skip_png_flag )and save_test_detail_files :
            latest_episodes =test_results .get ('all_episode_data',{})or {}
            if latest_episodes :
                try :
                    plot_station_cooperation_full (latest_episodes ,results_dir ,random_window =random_window ,title_prefix ="Test Results")
                except Exception as _e :
                    pass
                try :
                    plot_ev_detailed_soc (latest_episodes ,results_dir ,display_steps =10 ,random_window =random_window ,title_prefix ="Test Results")
                except Exception as _e :
                    pass

                    
            plot_power_mismatch_analysis (latest_episodes ,results_dir ,title_prefix ="Test Results")
            
            plot_reward_breakdown (latest_episodes ,results_dir ,title_prefix ="Test Results")
            
            plot_arrival_counts (latest_episodes ,results_dir ,title_prefix ="Test Results")
        elif skip_png_flag :
            print (f"[Info] Detailed test plots are skipped. Basic metrics are saved in CSV.")


            

            
        try :
            base_results_dir =os .path .dirname (results_dir )if "TEST"in os .path .basename (results_dir )else results_dir 
            os .makedirs (base_results_dir ,exist_ok =True )

            history_path =os .path .join (base_results_dir ,'test_history.json')
            
            if os .path .exists (history_path ):
                with open (history_path ,'r',encoding ='utf-8')as f :
                    hist =json .load (f )
            else :
                hist ={
                'episodes':[],
                'local_rewards':[],
                'global_rewards':[],
                'soc_miss_count':[],
                'surplus_absorption_rate':[],
                'supply_cooperation_rate':[],
                'surplus_steps':[],
                'surplus_within_narrow':[],
                'shortage_steps':[],
                'shortage_within_narrow':[],
                }
                if enable_switch_metrics :
                    hist ['avg_switches']=[]
                if enable_stlimit_metrics :
                    hist ['station_limit_hits']=[]
                    hist ['station_limit_steps']=[]
                    hist ['station_charge_limit_hits']=[]
                    hist ['station_discharge_limit_hits']=[]
                    hist ['station_limit_penalty_total']=[]

                
            try :
                import numpy as _np 
                cur_local =float (_np .mean (test_results ['all_local_rewards']))if test_results .get ('all_local_rewards')else 0.0 
                cur_global =float (_np .mean (test_results ['all_global_rewards']))if test_results .get ('all_global_rewards')else 0.0 
                pm =test_results .get ('performance_metrics',{})
                cur_soc_miss =float (_np .mean (pm .get ('soc_miss_count',[])))if pm .get ('soc_miss_count')else 0.0 
                cur_surplus_rate =float (_np .mean (pm .get ('surplus_absorption_rate',[])))if pm .get ('surplus_absorption_rate')else 0.0 
                cur_supply_rate =float (_np .mean (pm .get ('supply_cooperation_rate',[])))if pm .get ('supply_cooperation_rate')else 0.0 

                
                cur_surplus_steps =int (_np .sum (pm .get ('surplus_steps',[])))
                cur_surplus_within =int (_np .sum (pm .get ('surplus_within_narrow',[])))
                cur_shortage_steps =int (_np .sum (pm .get ('shortage_steps',[])))
                cur_shortage_within =int (_np .sum (pm .get ('shortage_within_narrow',[])))
                cur_station_limit_hits =int (_np .sum (pm .get ('station_limit_hits',[])))
                cur_station_limit_steps =int (_np .sum (pm .get ('station_limit_steps',[])))
                cur_station_charge_limit_hits =int (_np .sum (pm .get ('station_charge_limit_hits',[])))
                cur_station_discharge_limit_hits =int (_np .sum (pm .get ('station_discharge_limit_hits',[])))
                cur_station_limit_penalty_total =float (_np .sum (pm .get ('station_limit_penalty_total',[])))
            except Exception :
                cur_local =cur_global =cur_soc_miss =cur_surplus_rate =cur_supply_rate =0.0 
                cur_surplus_steps =cur_surplus_within =cur_shortage_steps =cur_shortage_within =0 
                cur_station_limit_hits =cur_station_limit_steps =0 
                cur_station_charge_limit_hits =cur_station_discharge_limit_hits =0 
                cur_station_limit_penalty_total =0.0 

                
            if test_episode_num in hist ['episodes']:
                idx =hist ['episodes'].index (test_episode_num )
                hist ['local_rewards'][idx ]=cur_local 
                hist ['global_rewards'][idx ]=cur_global 
                hist ['soc_miss_count'][idx ]=cur_soc_miss 
                hist ['surplus_absorption_rate'][idx ]=cur_surplus_rate 
                hist ['supply_cooperation_rate'][idx ]=cur_supply_rate 
                hist ['surplus_steps'][idx ]=cur_surplus_steps 
                hist ['surplus_within_narrow'][idx ]=cur_surplus_within 
                hist ['shortage_steps'][idx ]=cur_shortage_steps 
                hist ['shortage_within_narrow'][idx ]=cur_shortage_within 
                if enable_switch_metrics :
                    hist .setdefault ('avg_switches',[0.0 ]*len (hist ['episodes']))[idx ]=float (_np .mean (pm ['avg_switches']))if pm .get ('avg_switches')else 0.0 
                if enable_stlimit_metrics :
                    hist .setdefault ('station_limit_hits',[0 ]*len (hist ['episodes']))[idx ]=cur_station_limit_hits 
                    hist .setdefault ('station_limit_steps',[0 ]*len (hist ['episodes']))[idx ]=cur_station_limit_steps 
                    hist .setdefault ('station_charge_limit_hits',[0 ]*len (hist ['episodes']))[idx ]=cur_station_charge_limit_hits 
                    hist .setdefault ('station_discharge_limit_hits',[0 ]*len (hist ['episodes']))[idx ]=cur_station_discharge_limit_hits 
                    hist .setdefault ('station_limit_penalty_total',[0.0 ]*len (hist ['episodes']))[idx ]=cur_station_limit_penalty_total 
            else :
                hist ['episodes'].append (test_episode_num )
                hist ['local_rewards'].append (cur_local )
                hist ['global_rewards'].append (cur_global )
                hist ['soc_miss_count'].append (cur_soc_miss )
                hist ['surplus_absorption_rate'].append (cur_surplus_rate )
                hist ['supply_cooperation_rate'].append (cur_supply_rate )
                hist ['surplus_steps'].append (cur_surplus_steps )
                hist ['surplus_within_narrow'].append (cur_surplus_within )
                hist ['shortage_steps'].append (cur_shortage_steps )
                hist ['shortage_within_narrow'].append (cur_shortage_within )
                if enable_switch_metrics :
                    hist .setdefault ('avg_switches',[]).append (float (_np .mean (pm ['avg_switches']))if pm .get ('avg_switches')else 0.0 )
                if enable_stlimit_metrics :
                    hist .setdefault ('station_limit_hits',[]).append (cur_station_limit_hits )
                    hist .setdefault ('station_limit_steps',[]).append (cur_station_limit_steps )
                    hist .setdefault ('station_charge_limit_hits',[]).append (cur_station_charge_limit_hits )
                    hist .setdefault ('station_discharge_limit_hits',[]).append (cur_station_discharge_limit_hits )
                    hist .setdefault ('station_limit_penalty_total',[]).append (cur_station_limit_penalty_total )

                
            if hist ['episodes']:
                order =sorted (range (len (hist ['episodes'])),key =lambda i :hist ['episodes'][i ])
                hist ={
                'episodes':[hist ['episodes'][i ]for i in order ],
                'local_rewards':[hist ['local_rewards'][i ]for i in order ],
                'global_rewards':[hist ['global_rewards'][i ]for i in order ],
                'soc_miss_count':[hist ['soc_miss_count'][i ]for i in order ],
                'surplus_absorption_rate':[hist ['surplus_absorption_rate'][i ]for i in order ],
                'supply_cooperation_rate':[hist ['supply_cooperation_rate'][i ]for i in order ],
                'surplus_steps':[hist .get ('surplus_steps',[0 ]*len (order ))[i ]for i in order ],
                'surplus_within_narrow':[hist .get ('surplus_within_narrow',[0 ]*len (order ))[i ]for i in order ],
                'shortage_steps':[hist .get ('shortage_steps',[0 ]*len (order ))[i ]for i in order ],
                'shortage_within_narrow':[hist .get ('shortage_within_narrow',[0 ]*len (order ))[i ]for i in order ],
                }
                if enable_switch_metrics :
                    hist ['avg_switches']=[hist .get ('avg_switches',[0.0 ]*len (order ))[i ]for i in order ]

                
            with open (history_path ,'w',encoding ='utf-8')as f :
                json .dump (hist ,f ,ensure_ascii =False ,indent =2 )

                
            skip_png_flag =bool (USE_MILP or (not enable_png ))
            
            plot_daily_rewards (hist ['local_rewards'],hist ['global_rewards'],base_results_dir ,
            title_prefix ="Test Results",skip_png =skip_png_flag ,
            x_values =hist ['episodes'])
            
            perf_hist ={
            'soc_miss_count':hist ['soc_miss_count'],
            'surplus_absorption_rate':hist ['surplus_absorption_rate'],
            'supply_cooperation_rate':hist ['supply_cooperation_rate'],
            'surplus_steps':hist .get ('surplus_steps',[]),
            'surplus_within_narrow':hist .get ('surplus_within_narrow',[]),
            'shortage_steps':hist .get ('shortage_steps',[]),
            'shortage_within_narrow':hist .get ('shortage_within_narrow',[]),
            }
            if enable_switch_metrics :
                perf_hist ['avg_switches']=hist .get ('avg_switches',[])
            plot_performance_metrics (perf_hist ,base_results_dir ,title_prefix ="Test Results",
            x_values =hist ['episodes'],skip_png =skip_png_flag )
        except Exception as e :
            pass

            
        agent .set_test_mode (False )

        
        if original_use_tensorboard is not None :
            agent .use_tensorboard =original_use_tensorboard 

        return test_results 
    enable_switch_metrics =bool (USE_SWITCHING_CONSTRAINTS )and (float (LOCAL_SWITCH_PENALTY )!=0.0 )
    enable_stlimit_metrics =bool (USE_STATION_TOTAL_POWER_LIMIT )and (float (LOCAL_STATION_LIMIT_PENALTY )!=0.0 )
