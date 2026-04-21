"""
training/train.py

EV充放電マルチエージェント学習システムのメイン学習ループモジュール。

Config.py のフラグに基づいてエージェント（MILP / IndependentDDPG / SharedObsDDPG /
SharedObsSAC / MADDPG）を選択・インスタンス化し、EVEnv 環境上で強化学習を実行する。
ウォームアップフェーズでリプレイバッファを蓄積した後、学習フェーズへ移行する。
TensorBoard への各種メトリクス書き込み・定期評価（test()）・モデルスナップショット
保存も本モジュールが担う。
"""
import os
import csv
import json
import numpy as np
import torch
import random
from datetime import datetime
import traceback
import subprocess
import webbrowser
import time
import warnings
import logging


warnings .filterwarnings ("ignore",category =UserWarning ,module ="matplotlib")
warnings .filterwarnings ("ignore",message ="Glyph .* missing from font")
logging .getLogger ('matplotlib.font_manager').setLevel (logging .ERROR )
logging .getLogger ('matplotlib.ticker').setLevel (logging .ERROR )

from torch .utils .tensorboard import SummaryWriter
from environment.normalize import normalize_observation
from tools.Utils import create_tensorboard_writer ,GradientLossVisualizer ,snapshot_code_to_archive
import matplotlib
matplotlib .use ('Agg')
matplotlib .rcParams ['font.family']='sans-serif'
matplotlib .rcParams ['font.sans-serif']=['Arial','Helvetica','Liberation Sans','FreeSans','sans-serif']
from tools.Utils import plot_daily_rewards ,plot_station_cooperation_full ,plot_ev_detailed_soc ,plot_performance_metrics ,plot_arrival_counts ,plot_reward_breakdown
from tools.Utils import plot_power_mismatch_analysis
from environment.readcsv import load_multiple_demand_files
from tools.evaluator import set_env_seed ,test
from Config import (
NUM_EPISODES ,EPISODE_STEPS ,NUM_EVS ,NUM_STATIONS ,
LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC ,
USE_MILP ,USE_INDEPENDENT_DDPG ,ENV_SEED ,
REGULAR_MADDPG ,
GAMMA ,TAU ,BATCH_SIZE ,SMOOTHL1_BETA ,
MEMORY_SIZE ,
TRAIN_INTERIM_CSV_INTERVAL_EPISODES ,TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES ,
USE_SWITCHING_CONSTRAINTS ,LOCAL_SWITCH_PENALTY ,
USE_STATION_TOTAL_POWER_LIMIT ,LOCAL_STATION_LIMIT_PENALTY ,
)
from Config import (
USE_SHARED_OBS_DDPG ,USE_SHARED_OBS_SAC ,
MEASURE_LP_STEP_TIME ,MEASURE_STEP_INDEX ,
AUTO_ADJUST_MILP_WEIGHTS ,AUTO_ADJUST_TARGET_SOC_HIT ,
AUTO_ADJUST_TARGET_DISPATCH ,AUTO_ADJUST_MAX_SWITCHES ,AUTO_ADJUST_MAX_EPISODES ,
)
from environment.EVEnv import EVEnv
from training.Agent import MADDPG
from training.benchmark_agents.independent_ddpg import IndependentDDPG
from training.benchmark_agents.shared_obs_ddpg import SharedObsDDPG
from training.benchmark_agents.shared_obs_sac import SharedObsSAC
from training.Agent .maddpg import device


# -----------------------------------------------------------------------
# 定期評価・グラフ保存のインターバルを Config から読み込む
# max(1, ...) により 0 以下の設定値を防ぐ
# -----------------------------------------------------------------------
INTERIM_TEST_INTERVAL =max (1 ,int (TRAIN_INTERIM_CSV_INTERVAL_EPISODES ))
INTERIM_PLOT_INTERVAL =max (1 ,int (TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES ))


# -----------------------------------------------------------------------
# グローバルフラグ: Ctrl+C による学習中断を signal_handler が立てる
# -----------------------------------------------------------------------
interrupted =False

def sample_episode_demand_strict (demand_data ,episode_steps ):
    """
    エピソード用の需給調整量（net_demand）時系列データをランダムにサンプリングする。

    demand_data のいずれか1本をランダムに選び、episode_steps 分のデータを返す。
    選んだ系列が episode_steps より短い場合は末尾をゼロでパディングする。
    逆に長い場合は先頭 episode_steps ステップに切り詰める。

    Parameters
    ----------
    demand_data : list of array-like
        複数日の需給調整量系列のリスト（各要素が1日分の1次元配列）。
    episode_steps : int
        1エピソードのステップ数（ENV の episode_steps と一致させること）。

    Returns
    -------
    np.ndarray
        長さ episode_steps の1次元 float64 配列。
    """
    if int (episode_steps )<=0 :
        raise ValueError (f"episode_steps must be > 0, got {episode_steps}")
    if len (demand_data )==0 :
        raise ValueError ("demand_data is empty")
    # スレッドセーフなローカル RNG を使用してグローバル乱数状態を汚染しない
    _rng =random .Random ()
    _idx =_rng .randrange (len (demand_data ))
    _data =np .asarray (demand_data [_idx ],dtype =float ).reshape (-1 )
    if _data .size ==0 :
        raise ValueError ("Error: invalid runtime state.")

    # データが十分長ければ先頭から切り出す
    if _data .size >=int (episode_steps ):
        return _data [:int (episode_steps )]
    # 短い場合はゼロパディングで補完する
    return np .pad (_data ,(0 ,int (episode_steps )-int (_data .size )))


def launch_tensorboard (log_dir ,max_retries =3 ,port =6006 ):
    """
    TensorBoard サーバーをバックグラウンドプロセスとして起動し、ブラウザで開く。

    指定ポートが使用中の場合は既存の TensorBoard プロセスを終了させるか、
    空きポートを探して代替ポートで起動する。起動確認後にブラウザ（Chrome 優先）
    で URL を開く。起動に失敗した場合は None を返す。

    Parameters
    ----------
    log_dir : str
        TensorBoard が読み込むログディレクトリのパス。
    max_retries : int, optional
        起動試行回数の上限（デフォルト: 3）。
    port : int, optional
        最初に試みるポート番号（デフォルト: 6006）。

    Returns
    -------
    subprocess.Popen or None
        起動した TensorBoard プロセス。失敗時は None。
    """
    import sys
    import socket

    def check_port_available (port ):
        """指定ポートが未使用かどうかを確認する（True: 利用可能）。"""
        sock =socket .socket (socket .AF_INET ,socket .SOCK_STREAM )
        try :
            sock .bind (('localhost',port ))
            sock .close ()
            return True
        except OSError :
            return False

    def find_available_port (start_port =6006 ,max_attempts =10 ):
        """start_port から順に空きポートを探して返す。見つからなければ None。"""
        for p in range (start_port ,start_port +max_attempts ):
            if check_port_available (p ):
                return p
        return None

    def check_tensorboard_running (port ,max_wait =30 ):
        """TensorBoard が指定ポートで応答するまで最大 max_wait 秒待機する。"""
        import urllib .request
        import urllib .error
        url =f"http://localhost:{port}"
        for i in range (max_wait ):
            try :
                response =urllib .request .urlopen (url ,timeout =1 )
                if response .getcode ()==200 :
                    return True
            except (urllib .error .URLError ,socket .timeout ,ConnectionRefusedError ):
                time .sleep (1 )
        return False

    for attempt in range (max_retries ):
        try :

            if not check_port_available (port ):
                try :

                    import psutil
                    killed =False
                    for proc in psutil .process_iter (['pid','name','cmdline']):
                        try :
                            cmdline =proc .info .get ('cmdline',[])
                            if cmdline and any ('tensorboard'in str (arg ).lower ()for arg in cmdline ):
                                if str (port )in ' '.join (str (arg )for arg in cmdline ):
                                    proc .terminate ()
                                    try :
                                        proc .wait (timeout =3 )
                                    except psutil .TimeoutExpired :
                                        proc .kill ()
                                    killed =True
                                    time .sleep (1 )
                                    break
                        except (psutil .NoSuchProcess ,psutil .AccessDenied ,psutil .ZombieProcess ):
                            continue
                    if killed :

                        if not check_port_available (port ):
                            new_port =find_available_port (port +1 )
                            if new_port :
                                port =new_port
                            else :
                                pass
                    else :
                        new_port =find_available_port (port +1 )
                        if new_port :
                            port =new_port
                        else :
                            pass
                except ImportError :

                    # psutil が未インストールの場合は空きポートへフォールバック
                    new_port =find_available_port (port +1 )
                    if new_port :
                        port =new_port
                    else :
                        pass
                except Exception as e :
                    new_port =find_available_port (port +1 )
                    if new_port :
                        port =new_port
                    else :
                        pass


            if not check_port_available (port ):
                if attempt <max_retries -1 :
                    new_port =find_available_port (port +1 )
                    if new_port :
                        port =new_port
                    else :
                        pass
                else :
                    pass


            cmd =[sys .executable ,"-m","tensorboard.main","--logdir",log_dir ,"--port",str (port )]
            process =subprocess .Popen (
            cmd ,
            stdout =subprocess .PIPE ,
            stderr =subprocess .PIPE ,
            creationflags =subprocess .CREATE_NO_WINDOW if sys .platform =='win32'else 0
            )


            if check_tensorboard_running (port ,max_wait =30 ):

                try :

                    if sys .platform =='win32':
                        import os

                        chrome_paths =[
                        'C:/Program Files/Google/Chrome/Application/chrome.exe',
                        'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
                        os .path .expanduser ('~/AppData/Local/Google/Chrome/Application/chrome.exe')
                        ]
                        chrome_opened =False
                        for chrome_path in chrome_paths :
                            if os .path .exists (chrome_path ):

                                subprocess .Popen ([chrome_path ,f"http://localhost:{port}"])
                                chrome_opened =True
                                break
                        if not chrome_opened :

                            webbrowser .open (f"http://localhost:{port}")
                    else :

                        webbrowser .get ('google-chrome').open (f"http://localhost:{port}")
                except Exception as e :

                    webbrowser .open (f"http://localhost:{port}")
                return process
            else :
                try :
                    process .terminate ()
                    process .wait (timeout =2 )
                except :
                    try :
                        process .kill ()
                    except :
                        pass
                if attempt <max_retries -1 :

                    new_port =find_available_port (port +1 )
                    if new_port :
                        port =new_port
                    time .sleep (2 )

        except FileNotFoundError :
            return None
        except Exception as e :
            if attempt <max_retries -1 :

                new_port =find_available_port (port +1 )
                if new_port :
                    port =new_port
                time .sleep (2 )

    print (f"  python -m tensorboard.main --logdir {log_dir} --port {port}")
    return None

def signal_handler (sig ,frame ):
    """
    SIGINT（Ctrl+C）シグナルを受け取ったときのハンドラ。

    グローバルフラグ interrupted を True に設定することで、
    学習ループを現在のエピソード終了後に安全に中断させる。
    """
    global interrupted
    interrupted =True



def create_model_directory (model_name ):
    """
    学習結果を保存するディレクトリ群を作成して返す。

    archive/<model_name>_YYYYMMDD_HHMMSS/ の形式でタイムスタンプ付きの
    ルートディレクトリを作成し、その下に results / runs / performance / input
    の各サブディレクトリを生成する。さらにコードスナップショットを保存する。

    Parameters
    ----------
    model_name : str
        モデル名プレフィックス（例: "model"）。

    Returns
    -------
    str
        作成したルートディレクトリの絶対パス。
    """
    # カレントディレクトリ直下の archive/ を基底とする
    base_dir =os .getcwd ()

    archive_dir =os .path .join (base_dir ,"archive")
    os .makedirs (archive_dir ,exist_ok =True )

    # タイムスタンプ付きディレクトリ名を生成する
    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    model_dir_name =f"{model_name}_{timestamp}"
    model_dir =os .path .join (archive_dir ,model_dir_name )



    os .makedirs (model_dir ,exist_ok =True )

    # 固定サブディレクトリを作成する
    subdirs =["results","runs","performance","input"]
    for subdir in subdirs :
        os .makedirs (os .path .join (model_dir ,subdir ),exist_ok =True )


    # テスト結果専用のサブディレクトリを追加で作成する
    os .makedirs (os .path .join (model_dir ,"results","test"),exist_ok =True )


    # コードスナップショットを archive/ 内に保存する（再現性確保のため）
    try :
        snapshot_dir =snapshot_code_to_archive (model_dir )

    except Exception as e :
        pass

    return model_dir

def train (num_episodes =NUM_EPISODES ,random_window =True ,agent =None ,start_episode =0 ,model_name ="model",working_dir =None ,all_rewards =None ,performance_metrics =None ,all_episode_data =None ,profile_episode =None ):
    """
    メイン学習関数。エージェントを選択し、EVEnv 環境上で強化学習を実行する。

    処理の流れ:
        pass
    1. working_dir が未指定の場合は create_model_directory() でディレクトリを作成する。
    2. 環境乱数シードを固定し、EVEnv と需給調整量データをロードする。
    3. agent が None の場合は Config.py のフラグに基づいてエージェントをインスタンス化する。
    4. ウォームアップフェーズ: バッファが WARMUP_STEPS に達するまでデータ収集のみ行う。
    5. 学習フェーズ: 各エピソードで行動・環境ステップ・バッファ追加・ネットワーク更新を繰り返す。
    6. INTERIM_TEST_INTERVAL エピソードごとに評価（test()）とモデル保存を実施する。
    7. 学習終了後に最終グラフを保存し、エージェントや報酬履歴を返す。

    Parameters
    ----------
    num_episodes : int
        学習エピソード総数（ウォームアップ分を除く）。
    random_window : bool
        需給調整量のサンプリングにランダムウィンドウを使用するか否か。
    agent : object or None
        学習済みエージェントを渡す場合に指定する。None の場合は新規生成。
    start_episode : int
        エピソードカウントの開始番号（再開学習用）。
    model_name : str
        モデルディレクトリ名のプレフィックス。
    working_dir : str or None
        結果保存先のルートディレクトリパス。None の場合は自動生成。
    all_rewards : list or None
        エピソード報酬の累積リスト（再開学習用）。
    performance_metrics : dict or None
        パフォーマンス指標の累積辞書（再開学習用）。
    all_episode_data : dict or None
        エピソードごとの詳細データの累積辞書（再開学習用）。
    profile_episode : int or None
        プロファイリング対象エピソード番号（未使用予約引数）。

    Returns
    -------
    tuple
        (agent, all_rewards, performance_metrics, all_episode_data, working_dir)
    """

    # working_dir が未指定の場合はタイムスタンプ付きディレクトリを新規作成する
    if working_dir is None :
        working_dir =create_model_directory (model_name )

    # 環境乱数シードを固定して再現性を確保する
    set_env_seed (ENV_SEED )


    # TensorBoard ライター初期化（MILP モードでは無効化する）
    performance_dir =os .path .join (working_dir ,"performance")
    os .makedirs (performance_dir ,exist_ok =True )
    if USE_MILP :
        tb_writer =None
        print ("[Info] MILP mode: TensorBoard output is disabled.")
    else :
        tb_writer =create_tensorboard_writer (log_dir =performance_dir )

    # 中断フラグをリセットする（再実行時に前回の状態を引き継がないため）
    global interrupted
    interrupted =False


    # 環境インスタンスを生成する
    env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )


    demand_data_train =None
    try :

        # 学習用需給調整量 CSV を複数ファイルからロードし、train セットを取得する
        all_demand_data =load_multiple_demand_files (train_split =25 )
        demand_data_train =all_demand_data ['train']
    except FileNotFoundError as e :
        error_message ="Failed to load training demand CSV files."
        print (error_message )
        raise FileNotFoundError (error_message )

    # 需給調整量データが空の場合は続行不可
    if not demand_data_train :
        raise RuntimeError ("Error: invalid runtime state.")
    # 環境を最初のエピソード用データでリセットして観測次元を確認する
    episode_demand =sample_episode_demand_strict (demand_data_train ,env .episode_steps )
    env .reset (net_demand_series =episode_demand )

    state_dim =env ._get_obs ().shape [1 ]

    # エージェントが未指定の場合は Config フラグに従って生成する
    if agent is None :




        # 複数フラグが同時に True になっていないか検証する（排他チェック）
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

        if USE_MILP :

            # MILP 中央制御エージェントを生成する
            from training.benchmark_agents.milp_agent import MILPAgent
            from Config import MILP_HORIZON
            agent =MILPAgent (max_evs_per_station =env .max_ev_per_station ,horizon =MILP_HORIZON )
        elif USE_INDEPENDENT_DDPG :

            # 各ステーションが独立した Actor-Critic を持つ Independent DDPG エージェントを生成する
            agent =IndependentDDPG (state_dim ,env .max_ev_per_station ,n_agent =env .num_stations ,
            num_episodes =num_episodes ,
            batch =BATCH_SIZE ,gamma =GAMMA ,tau =TAU ,
            lr_a =LR_ACTOR ,lr_c =LR_CRITIC_LOCAL ,
            smoothl1_beta =SMOOTHL1_BETA )

            # バッファサイズを Config の MEMORY_SIZE に設定する
            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )
        elif USE_SHARED_OBS_DDPG :

            # 全ステーションの観測を共有する Shared-Obs DDPG エージェントを生成する
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

            # 共有観測ベースの Soft Actor-Critic（SAC）エージェントを生成する
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


            # 標準 MADDPG（CTDE: 集中学習・分散実行）エージェントを生成する
            from Config import TAU_GLOBAL ,TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,TD3_SIGMA_LOCAL ,TD3_CLIP_LOCAL
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
            td3_sigma_local =TD3_SIGMA_LOCAL ,
            td3_clip_local =TD3_CLIP_LOCAL ,
            smoothl1_beta =0.01
            )
            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )
        else :

            # デフォルト: 拡張 MADDPG（グローバルクリティック付き TD3 構成）を生成する
            from Config import TAU_GLOBAL ,TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,TD3_SIGMA_LOCAL ,TD3_CLIP_LOCAL
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

            td3_sigma_local =TD3_SIGMA_LOCAL ,
            td3_clip_local =TD3_CLIP_LOCAL ,
            smoothl1_beta =0.01
            )

            if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
                agent .buf .buf_size =int (MEMORY_SIZE )


    try :
        if USE_MILP :
            # MILP はニューラルネットワーク不要なので TensorBoard ライターを無効にする
            agent .writer =None
            agent .use_tensorboard =False
        else :
            # エージェント内部の SummaryWriter を runs/ 以下に向ける
            runs_dir =os .path .join (working_dir ,"runs")
            os .makedirs (runs_dir ,exist_ok =True )
            if hasattr (agent ,'writer'):
                if agent .writer is not None :
                    try :
                        agent .writer .close ()
                    except :
                        pass
            agent .writer =SummaryWriter (log_dir =runs_dir )
            agent .use_tensorboard =True


    except Exception :
        pass

    # -----------------------------------------------------------------------
    # MILP 重みの自動調整フェーズ（AUTO_ADJUST_MILP_WEIGHTS が True の場合のみ）
    # 二分探索で w_ag（需給追従重み）を調整し、SoC 達成率と需給追従率の目標を満たす
    # -----------------------------------------------------------------------
    if USE_MILP and AUTO_ADJUST_MILP_WEIGHTS :
        print ("\n===== MILP/LP Weights Auto-Adjustment Phase Start =====")
        print (f"Target: SoC Hit >= {AUTO_ADJUST_TARGET_SOC_HIT}%, Dispatch Tracking >= {AUTO_ADJUST_TARGET_DISPATCH}%, Max Switches <= {AUTO_ADJUST_MAX_SWITCHES}")

        # 二分探索の初期範囲を設定する
        low_w_ag =0.0
        high_w_ag =20.0
        agent .set_weights (w_ag =(low_w_ag +high_w_ag )/2.0 ,w_soc =100.0 )

        for adj_ep in range (1 ,AUTO_ADJUST_MAX_EPISODES +1 ):

            # 1エピソード分の評価を行い、パフォーマンス指標を取得する
            adj_results =test (agent ,demand_adjustment =None ,random_window =False ,
            working_dir =working_dir ,test_episode_num =0 ,num_episodes =1 ,
            enable_png =False ,save_test_detail_files =False )

            pm =adj_results ['performance_metrics']
            soc_hit =100.0 -pm ['soc_miss_count'][-1 ]
            s_steps =pm ['surplus_steps'][-1 ]
            s_within =pm ['surplus_within_narrow'][-1 ]
            sh_steps =pm ['shortage_steps'][-1 ]
            sh_within =pm ['shortage_within_narrow'][-1 ]
            denom =s_steps +sh_steps
            dispatch_rate =(s_within +sh_within )/denom *100.0 if denom >0 else 100.0
            avg_sw =pm .get ('avg_switches',[0.0 ])[-1 ]
            adj_parts =[
            f"[Adj Ep {adj_ep}] SoC Hit: {soc_hit:.1f}%",
            f"Dispatch: {dispatch_rate:.1f}%",
            ]
            if bool (USE_SWITCHING_CONSTRAINTS )and (float (LOCAL_SWITCH_PENALTY )!=0.0 ):
                adj_parts .append (f"Switches: {avg_sw:.2f}")
            adj_parts .append (f"Current w_ag: {agent.w_ag:.4f} (Range: [{low_w_ag:.2f}, {high_w_ag:.2f}])")
            print (" | ".join (adj_parts ))

            # 目標達成かつ探索範囲が収束したら調整を終了する
            if (soc_hit >=AUTO_ADJUST_TARGET_SOC_HIT and
            dispatch_rate >=AUTO_ADJUST_TARGET_DISPATCH and
            (high_w_ag -low_w_ag )<0.1 ):
                print (">> Targets achieved with optimal balance! Ending adjustment phase.")
                break

            # 次の重み候補を算出する（スイッチング制約の調整も含む）
            new_w_ag =agent .w_ag
            new_w_soc =100.0
            new_w_switch =agent .w_switch


            if soc_hit <AUTO_ADJUST_TARGET_SOC_HIT :


                # SoC 達成率が不足 → w_ag を下げて SoC 優先度を高める（二分探索の上限を縮める）
                high_w_ag =agent .w_ag

                if soc_hit <10.0 :
                    new_w_ag =low_w_ag +(high_w_ag -low_w_ag )*0.3
                else :
                    new_w_ag =(low_w_ag +high_w_ag )/2.0
            else :


                # SoC 達成率は十分 → w_ag を上げて需給追従率を改善する（二分探索の下限を広げる）
                low_w_ag =agent .w_ag


                if high_w_ag -low_w_ag <0.1 :


                    # 探索範囲が狭くなりすぎた場合は強制的に上限を拡張する
                    expansion =max (1.0 ,(high_w_ag -low_w_ag )*5.0 )
                    high_w_ag =low_w_ag +expansion

                new_w_ag =(low_w_ag +high_w_ag )/2.0


            if USE_SWITCHING_CONSTRAINTS :
                if avg_sw >AUTO_ADJUST_MAX_SWITCHES :
                    new_w_switch +=0.2
                elif avg_sw <AUTO_ADJUST_MAX_SWITCHES *0.5 :
                    new_w_switch =max (0.01 ,new_w_switch *0.9 )


            agent .set_weights (w_ag =new_w_ag ,w_soc =new_w_soc ,w_switch =new_w_switch )

        print (f"===== Weights Auto-Adjustment Phase Finished (Final w_ag: {agent.w_ag:.4f}) =====\n")


        print ("===== Weights Auto-Adjustment Phase Finished =====\n")



    # -----------------------------------------------------------------------
    # 累積データ構造の初期化（再開学習時は外部から渡されたものを引き継ぐ）
    # -----------------------------------------------------------------------
    if all_rewards is None :
        all_rewards =[]

    # ローカル報酬・グローバル報酬の系列を別途保持する
    all_local_rewards =[]
    all_global_rewards =[]


    if all_episode_data is None :
        all_episode_data ={}

    # スイッチング制約・ステーション電力上限のメトリクス有効フラグ
    enable_switch_metrics =bool (USE_SWITCHING_CONSTRAINTS )and (float (LOCAL_SWITCH_PENALTY )!=0.0 )
    enable_stlimit_metrics =bool (USE_STATION_TOTAL_POWER_LIMIT )and (float (LOCAL_STATION_LIMIT_PENALTY )!=0.0 )


    if performance_metrics is None :
        # パフォーマンス指標辞書を初期化する（各キーがエピソードごとの値リスト）
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

        # ===================================================================
        # メインエピソードループ
        # ===================================================================
        for ep in range (start_episode +1 ,start_episode +num_episodes +1 ):






            # ---------------------------------------------------------------
            # ウォームアップ判定: バッファサイズが WARMUP_STEPS 未満ならウォームアップ
            # ---------------------------------------------------------------
            try :
                from Config import WARMUP_STEPS as _WARMUP_STEPS
                _is_warmup =hasattr (agent ,'buf')and getattr (agent .buf ,'size',0 )<_WARMUP_STEPS
            except Exception :
                _is_warmup =False
                _WARMUP_STEPS =0

            if _is_warmup :
                # ウォームアップフェーズ: スナップショット不要のため無効化する
                env .record_snapshots =False
                _wu_demand =sample_episode_demand_strict (demand_data_train ,env .episode_steps )
                env .reset (net_demand_series =_wu_demand )
                agent .episode_start ()
                agent .update_active_evs (env )
                _wu_prefetch =None
                while True :
                    # 前ステップのプリフェッチ観測があればそれを使い、なければ新規取得する
                    if _wu_prefetch is not None :
                        _wu_obs =normalize_observation (_wu_prefetch )
                        _wu_prefetch =None
                    else :
                        _wu_obs =normalize_observation (env .begin_step ())
                    try :
                        agent .update_active_evs (env )
                    except Exception :
                        pass
                    _wu_act =agent .act (_wu_obs ,env =env ,noise =True )
                    _ ,_wu_rl ,_wu_rg ,_wu_done ,_wu_info =env .apply_action (_wu_act )
                    if all (_wu_done ):
                        _wu_next =normalize_observation (env ._get_obs ())
                    else :
                        _wu_next_raw =env .begin_step ()
                        _wu_next =normalize_observation (_wu_next_raw )
                        _wu_prefetch =_wu_next_raw
                    # 終端遷移も含めてバッファに追加する（done=True のまま格納し y=r+0 で正しく学習させる）
                    if hasattr (agent ,'cache_experience'):
                        agent .cache_experience (
                        torch .as_tensor (_wu_obs ,dtype =torch .float32 ,device =device ),
                        torch .as_tensor (_wu_next ,dtype =torch .float32 ,device =device ),
                        _wu_act ,
                        torch .as_tensor (_wu_rl ,dtype =torch .float32 ,device =device ),
                        torch .tensor (_wu_rg ,dtype =torch .float32 ,device =device ),
                        torch .as_tensor (_wu_done ,dtype =torch .float32 ,device =device ),
                        actual_station_powers =torch .as_tensor (
                        _wu_info ['station_powers'],dtype =torch .float32 ,device =device ),
                        actual_ev_power_kw =_wu_info .get ('actual_ev_power_kw'),
                        )
                        if not all (_wu_done ):
                            agent .update ()
                    if all (_wu_done ):
                        agent .episode_end ()
                        break
                # 5エピソードごと、または WARMUP_STEPS 到達時に進捗を表示する
                _wu_buf =getattr (agent .buf ,'size',0 )
                if ep %5 ==1 or _wu_buf >=_WARMUP_STEPS :
                    print (f"[WARMUP] ep={ep:4d}  buffer={_wu_buf}/{_WARMUP_STEPS}",flush =True )
                continue


            # ---------------------------------------------------------------
            # ウォームアップ完了を検出して学習開始エピソード番号を記録する
            # ---------------------------------------------------------------
            if not getattr (agent ,'_learning_started_episode',None ):
                agent ._learning_started_episode =ep
                print (
                f"[Info] Warmup complete at ep={ep}. "
                f"Learning starts (training_ep=1/{num_episodes}).",
                flush =True ,
                )
            # TensorBoard の x 軸に使うウォームアップ後通し番号を算出する
            training_ep =ep -agent ._learning_started_episode +1




            # テストモード（評価時）かつ LP 計測モードでなければスナップショットを記録する
            env .record_snapshots =bool (getattr (agent ,'test_mode',False ))and not MEASURE_LP_STEP_TIME


            if ep ==2 :
                try :

                    # 2エピソード目に TensorBoard を起動してブラウザで開く
                    launch_tensorboard (working_dir )
                except Exception as e :
                    pass

            # Q 値・勾配・損失を TensorBoard に記録するビジュアライザーを初期化する
            visualizer =GradientLossVisualizer (env .num_stations ,tb_writer )




            # 各エピソード用の需給調整量データをサンプリングして環境をリセットする
            episode_demand =sample_episode_demand_strict (demand_data_train ,env .episode_steps )
            state =env .reset (net_demand_series =episode_demand )

            agent .episode_start ()
            ep_r =0.0

            ep_start_time =time .time ()

            # 現在在籍している EV のリストをエージェントに通知する
            agent .update_active_evs (env )

            # エピソード内データ収集用の辞書を初期化する
            episode_data ={
            'ag_requests':[],
            'total_ev_transport':[],
            'soc_data':{},
            'power_mismatch':[]
            }

            # 各ステーションの実際の EV 電力記録リストを用意する
            for i in range (1 ,env .num_stations +1 ):
                episode_data [f'actual_ev{i}']=[]

            # 各ステーション・各 EV の SoC 時系列データを初期化する
            for station_idx in range (env .num_stations ):

                episode_data ['soc_data'][f'station{station_idx+1}']={}


                for ev_idx ,ev in enumerate (env .stations_evs [station_idx ]):

                    ev_id =str (ev ['id'])

                    # 各 EV の出発ステップ・目標 SoC・SoC 時系列を初期化する
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

            # 報酬内訳ごとのエピソード累積値を初期化する
            ep_local_departure_r =0.0
            ep_local_progress_shaping_r =0.0
            ep_local_discharge_penalty_r =0.0
            ep_local_switch_penalty_r =0.0
            ep_local_station_limit_penalty_r =0.0

            # 各ステーションの報酬内訳合計（エピソード単位の集計用）
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

            # 余剰電力の吸収量（需給追従）の累計
            total_surplus_available =0
            total_surplus_absorbed =0

            # 不足電力の放電供給量の累計
            total_discharge_request =0
            total_discharge_fulfilled =0

            # 前ステップの観測をプリフェッチして begin_step() 呼び出し回数を削減する
            _prefetch_obs =None

            # ---------------------------------------------------------------
            # ステップループ（1 エピソード = EPISODE_STEPS ステップ）
            # ---------------------------------------------------------------
            while True :

                MAX_RETRIES =3
                retry_count =0
                step_success =False

                def is_safe (x ):
                    if isinstance (x ,torch .Tensor ):
                        return torch .isfinite (x ).all ().item ()
                    else :
                        return np .isfinite (x ).all ()

                # ステップデータを格納する変数の初期化
                info =None
                act_tensor =None
                obs1 =None
                next_state =None
                r_local =None
                r_global =None
                done =None

                # プリフェッチ済み観測があればそれを利用し、なければ None のままにする
                _step_obs_raw =_prefetch_obs
                _prefetch_obs =None

                while retry_count <MAX_RETRIES and not step_success :

                    # プリフェッチ観測を正規化して使用する（2 回目以降は環境から取得）
                    if _step_obs_raw is not None and retry_count ==0 :
                        obs1 =normalize_observation (_step_obs_raw )
                        _step_obs_raw =None
                    else :
                        obs1 =normalize_observation (env .begin_step ())

                    try :
                        agent .update_active_evs (env )
                    except Exception :
                        pass


                    # LP ステップ計測モードの処理
                    if MEASURE_LP_STEP_TIME :
                        current_step =int (env .step_count )
                        if current_step <MEASURE_STEP_INDEX :


                            if current_step %10 ==1 or current_step ==1 :
                                print (f"[Measurement] Progress: Step {current_step}/{MEASURE_STEP_INDEX} (Random actions, {env.num_stations} stations)...")
                            act_tensor =(torch .rand ((env .num_stations ,env .max_ev_per_station ),device =device )*2.0 -1.0 )
                        elif current_step ==MEASURE_STEP_INDEX :

                            print (f"\n[Measurement] Measuring calculation time for Step {current_step}...")


                            def estimate_problem_size (agent ,env ):
                                """
                                LP/MILP 問題の規模（変数数・制約数）を事前に推定する。

                                MILP エージェントの場合は各ステーションのアクティブ EV と
                                ホライズン H をもとに変数・制約の概算値を計算する。
                                それ以外のエージェント（DRL 系）は全て 0 を返す。
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

                                                # 各 EV の残りホライズン分の変数・制約を加算する
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

                            # 計算量の理論推定値を算出する（反復内 FLOP × 反復回数）
                            n ,m =num_vars ,num_constraints
                            flops_per_iteration =m *m *m
                            estimated_iterations =max (m ,int (n *0.1 ))
                            estimated_flops =flops_per_iteration *estimated_iterations
                            simple_flops_per_iteration =n *m
                            simple_estimated_flops =simple_flops_per_iteration *estimated_iterations

                            # 変数数・制約数の大小関係に応じて計算量オーダーを文字列で表す
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

                            # RTX 4080 の理論性能（GPU との比較用）
                            RTX4080_TFLOPS_FP32 =83.0
                            RTX4080_TFLOPS_FP16 =1248.0

                            # LP が GPU で解けた場合の理論時間を計算する（参考値）
                            gpu_time_fp32 =estimated_flops /(RTX4080_TFLOPS_FP32 *1e12 )if estimated_flops >0 else 0
                            gpu_time_fp16 =estimated_flops /(RTX4080_TFLOPS_FP16 *1e12 )if estimated_flops >0 else 0
                            gpu_time_simple_fp32 =simple_estimated_flops /(RTX4080_TFLOPS_FP32 *1e12 )if simple_estimated_flops >0 else 0
                            gpu_time_simple_fp16 =simple_estimated_flops /(RTX4080_TFLOPS_FP16 *1e12 )if simple_estimated_flops >0 else 0


                            print (f"============================================================")
                            print (f"[Problem Size Estimation (Before Solving)]")
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
                            print (f"------------------------------------------------------------")
                            print (f"[Computational Complexity Estimation]")
                            print (f"  - Complexity order: {complexity_order}")
                            print (f"  - Estimated iterations: {estimated_iterations}")
                            print (f"  - Estimated FLOPs (O(m^3) per iteration): {format_flops(estimated_flops)}")
                            print (f"  - Estimated FLOPs (O(n*m) per iteration): {format_flops(simple_estimated_flops)}")
                            print (f"------------------------------------------------------------")
                            print (f"[GPU Performance Estimation (RTX 4080)]")
                            print (f"  Note: Current solver (CBC) runs on CPU. Below is theoretical estimate")
                            print (f"        if the problem could be solved on GPU.")
                            print (f"  - RTX 4080 FP32 performance: {RTX4080_TFLOPS_FP32:.1f} TFLOPS")
                            print (f"  - RTX 4080 FP16 performance: {RTX4080_TFLOPS_FP16:.1f} TFLOPS")
                            if gpu_time_fp32 >0 :
                                pass
                            if gpu_time_simple_fp32 >0 :
                                pass
                            print (f"============================================================")


                            print (f"\n[Executing LP solver...]")
                            t_start =time .time ()
                            act_tensor =agent .act (obs1 ,env =env ,noise =False )
                            t_end =time .time ()
                            duration =t_end -t_start

                            # エージェントが返す実際の問題サイズ情報を取得して事前推定と比較する
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


                            print (f"\n============================================================")
                            print (f"[Measurement Result] Step {current_step} LP solve time: {duration:.6f} seconds")
                            if actual_num_vars !=num_vars or actual_num_constraints !=num_constraints :
                                print (f"  Note: Actual problem size differs from estimation:")
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
                            print (f"============================================================")

                            # 計測結果を CSV ファイルに追記する
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
                            except Exception :
                                pass

                            # 計測完了後はプロセスを終了する
                            import sys
                            print ("Measurement completed. Exiting...")
                            sys .exit (0 )
                        else :

                            # 計測ステップを超えた後は通常の行動選択を行う
                            act_tensor =agent .act (obs1 ,env =env ,noise =False )
                    else :
                        # 通常の学習フェーズ: 探索ノイズを加えて行動を選択する
                        act_tensor =agent .act (obs1 ,env =env ,noise =True )

                    _ ,r_local_tmp ,r_global_tmp ,done_tmp ,info_tmp =env .apply_action (act_tensor )

                    if all (done_tmp ):
                        next_state_tmp =normalize_observation (env ._get_obs ())
                        _next_prefetch_tmp =None
                    else :
                        # 次ステップ観測をプリフェッチして次ループで再利用する
                        _next_raw =env .begin_step ()
                        next_state_tmp =normalize_observation (_next_raw )
                        _next_prefetch_tmp =_next_raw

                    # NaN/Inf が検出された場合はリトライする
                    if not (is_safe (obs1 )and is_safe (next_state_tmp )and is_safe (act_tensor )and is_safe (r_local_tmp )and is_safe (r_global_tmp )):
                        retry_count +=1
                        print (f"Retry: Step data NaN/Inf detected (attempt {retry_count}/{MAX_RETRIES})")
                        continue

                    # 安全なデータを本変数に確定する
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

                # ステップ終了後にアクティブ EV 情報を更新する
                agent .update_active_evs (env )

                # エピソード累積報酬を更新する
                ep_r +=float (sum (r_local )+r_global )
                ep_local_r +=np .mean (r_local )
                ep_global_r +=r_global

                # 報酬内訳（出発報酬・整形報酬・ペナルティ等）をステーションごとに集計する
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

                        # ステーション別の報酬内訳合計を更新する
                        if 0 <=st_idx <len (station_local_reward_sums ):
                            sums =station_local_reward_sums [st_idx ]
                            sums ["total"]+=total_r
                            sums ["departure"]+=dep_r
                            sums ["progress_shaping"]+=shaping_r
                            sums ["discharge_penalty"]+=discharge_penalty_r
                            sums ["switch_penalty"]+=switch_penalty_r
                            sums ["station_limit_penalty"]+=station_limit_penalty_r


                # 終端遷移も含めてリプレイバッファに経験を追加する（done=True のまま格納し y=r+0 で正しく学習させる）
                if hasattr (agent ,'cache_experience'):
                    state_tensor_for_buffer =torch .as_tensor (obs1 ,dtype =torch .float32 ,device =device )
                    next_state_tensor =torch .as_tensor (next_state ,dtype =torch .float32 ,device =device )

                    # EV 単位の実電力テンソルを info から取得する
                    actual_ev_power_kw_tensor =info .get ('actual_ev_power_kw')

                    agent .cache_experience (
                    state_tensor_for_buffer ,
                    next_state_tensor ,
                    act_tensor ,
                    torch .as_tensor (r_local ,dtype =torch .float32 ,device =device ),
                    torch .tensor (r_global ,dtype =torch .float32 ,device =device ),
                    torch .as_tensor (done ,dtype =torch .float32 ,device =device ),
                    actual_station_powers =torch .as_tensor (info ['station_powers'],dtype =torch .float32 ,device =device ),
                    actual_ev_power_kw =actual_ev_power_kw_tensor ,
                    )


                # MADDPG 系（actors/critics を持つ）はエピソード終了前のみ更新する
                if hasattr (agent ,'actors')and hasattr (agent ,'critics')and not all (done ):
                    agent .update ()
                else :

                    # MILP 等の他エージェントは update() を無条件で呼ぶ（内部でスキップ可能）
                    if hasattr (agent ,'update'):
                        agent .update ()


                # Q 値をビジュアライザーに渡す（エージェントの種類によってアクセス先が異なる）
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

                # 勾配・損失・クリッピング情報をビジュアライザーに渡す
                visualizer .update_gradients (agent )
                visualizer .update_losses (agent )
                visualizer .update_clipping (agent )

                # 需給調整量リクエストを記録する
                episode_data ['ag_requests'].append (info ['net_demand'])

                # 全 EV の合計電力輸送量を記録する
                episode_data ['total_ev_transport'].append (info ['total_ev_transport'])
                if 'pre_total_ev_transport'not in episode_data :
                    episode_data ['pre_total_ev_transport']=[]
                    for i in range (1 ,env .num_stations +1 ):
                        episode_data [f'pre_ev{i}']=[]
                if 'actual_ev_power_kw'not in info :
                    raise KeyError ("info must include 'actual_ev_power_kw'")
                ev_changes_tensor =info ['actual_ev_power_kw']

                # ステーション単位の EV 合計電力をリストに追加する
                station_sums =ev_changes_tensor .sum (dim =1 ).detach ().cpu ().tolist ()
                pre_total =0.0
                for station_idx ,station_sum in enumerate (station_sums ):
                    episode_data [f'pre_ev{station_idx+1}'].append (float (station_sum ))
                    pre_total +=float (station_sum )
                episode_data ['pre_total_ev_transport'].append (pre_total )

                # 余剰・不足の追従統計を更新する
                request =info ['net_demand']

                transport =info ['total_ev_transport']
                if request >0 :
                    # 余剰電力の吸収量を集計する
                    total_surplus_available +=request
                    total_surplus_absorbed +=min (request ,transport )
                elif request <0 :
                    # 不足電力の放電量を集計する
                    total_discharge_request +=abs (request )
                    total_discharge_fulfilled +=min (abs (request ),transport )

                # 各ステーションの実際の電力をエピソードデータに追記する
                for i in range (env .num_stations ):
                    if f'actual_ev{i+1}'not in episode_data :
                        episode_data [f'actual_ev{i+1}']=[]


                    episode_data [f'actual_ev{i+1}'].append (info ['station_powers'][i ])


                if 'departed_evs'in info and info ['departed_evs']:
                    for departed_ev in info ['departed_evs']:
                        ev_id =str (departed_ev .get ('id'))
                        station_idx =departed_ev .get ('station')
                        if station_idx is None :
                            continue

                        # 出発 EV の最終 SoC データを soc_data に記録する
                        if f'station{station_idx+1}'not in episode_data ['soc_data']:
                            episode_data ['soc_data'][f'station{station_idx+1}']={}
                        if ev_id in episode_data ['soc_data'][f'station{station_idx+1}']:

                            after_list =info .get ('snapshot_after',{}).get (station_idx ,[])
                            matched =next ((d for d in after_list if str (d .get ('id'))==ev_id ),None )
                            if matched is not None :
                                final_soc =matched .get ('new_soc',None )
                                prev_soc =matched .get ('prev_soc',None )
                                needed =matched .get ('needed_soc',None )
                                target_soc =matched .get ('target_soc',None )
                                if target_soc is None :
                                    raise ValueError (f"EV {ev_id}: target_soc not found in snapshot data. Cannot proceed without target SoC information.")
                                if final_soc is not None :
                                    episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['final_soc']=float (final_soc )
                                if target_soc is not None :
                                    episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['target_soc']=float (target_soc )

                            episode_data ['soc_data'][f'station{station_idx+1}'][ev_id ]['depart_step']=int (info .get ('step_count',env .step_count ))



                # 観測を次ステップ用に更新する
                state =next_state

                # done フラグが全ステーションで True になったらエピソード終了
                if all (done ):
                    agent .episode_end ()
                    break



            ep_end_time =time .time ()
            ep_duration =ep_end_time -ep_start_time


            # エピソードの実ステップ数を取得する（0 除算防止で最低 1 にする）
            steps_in_ep =env .step_count if hasattr (env ,'step_count')and env .step_count >0 else 1



            _is_warmup =False

            # 環境からエピソード終了時のパフォーマンス指標を取得する
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
            avg_switches =metrics .get ('avg_switches',0.0 )
            station_limit_hits =metrics .get ('station_limit_hits',0 )
            station_limit_steps =metrics .get ('station_limit_steps',0 )
            station_charge_limit_hits =metrics .get ('station_charge_limit_hits',0 )
            station_discharge_limit_hits =metrics .get ('station_discharge_limit_hits',0 )
            station_limit_penalty_total =metrics .get ('station_limit_penalty_total',0.0 )
            # ステップあたりのステーション電力上限ペナルティを計算する
            station_limit_penalty_per_step =(
            station_limit_penalty_total /max (steps_in_ep ,1 )
            )
            # ヒットあたりのペナルティ（ヒット 0 のときは 0 にする）
            station_limit_penalty_per_hit =(
            station_limit_penalty_total /max (station_limit_hits ,1 )
            if station_limit_hits >0 else 0.0
            )

            # ウォームアップでなければ累積リストと performance_metrics に追記する
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

            # コンソールにエピソード進捗を表示する
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
            # TensorBoard への報酬スカラーを書き込む
            # x 軸は training_ep（ウォームアップ後の通し番号）を使用する
            # ---------------------------------------------------------------
            if tb_writer and not _is_warmup :





                tb_writer .add_scalar ("Reward/local",ep_local_r /steps_in_ep ,training_ep )
                tb_writer .add_scalar ("Reward/global",ep_global_r /steps_in_ep ,training_ep )
                tb_writer .add_scalar ("Reward/local_departure",ep_local_departure_r /steps_in_ep ,training_ep )

                tb_writer .add_scalar ("Reward/local_shaping",ep_local_progress_shaping_r /steps_in_ep ,training_ep )

                tb_writer .add_scalar ("Reward/local_discharge_penalty",ep_local_discharge_penalty_r /steps_in_ep ,training_ep )
                if enable_switch_metrics :
                    tb_writer .add_scalar ("Reward/local_switch_penalty",ep_local_switch_penalty_r /steps_in_ep ,training_ep )
                if enable_stlimit_metrics :
                    tb_writer .add_scalar ("Reward/local_station_limit_penalty",ep_local_station_limit_penalty_r /steps_in_ep ,training_ep )

            # ステーション別の報酬内訳を TensorBoard に書き込む
            if tb_writer and not _is_warmup :
                for st_idx ,sums in enumerate (station_local_reward_sums ):
                    st_id =st_idx +1
                    tb_writer .add_scalar (f"Reward/local_station{st_id}_total",sums ["total"]/steps_in_ep ,training_ep )
                    tb_writer .add_scalar (f"Reward/local_station{st_id}_departure",sums ["departure"]/steps_in_ep ,training_ep )
                    tb_writer .add_scalar (f"Reward/local_station{st_id}_shaping",sums ["progress_shaping"]/steps_in_ep ,training_ep )
                    tb_writer .add_scalar (f"Reward/local_station{st_id}_discharge_penalty",sums ["discharge_penalty"]/steps_in_ep ,training_ep )
                    if enable_switch_metrics :
                        tb_writer .add_scalar (f"Reward/local_station{st_id}_switch_penalty",sums ["switch_penalty"]/steps_in_ep ,training_ep )
                    if enable_stlimit_metrics :
                        tb_writer .add_scalar (f"Reward/local_station{st_id}_station_limit_penalty",sums ["station_limit_penalty"]/steps_in_ep ,training_ep )
                # パフォーマンス指標を TensorBoard に書き込む
                tb_writer .add_scalar ("Metrics/soc_hit_rate",100 -soc_miss_rate ,training_ep )
                if enable_switch_metrics :
                    tb_writer .add_scalar ("Metrics/avg_switches",avg_switches ,training_ep )
                tb_writer .add_scalar ("Metrics/surplus_absorption_rate",surplus_absorption_rate ,training_ep )
                tb_writer .add_scalar ("Metrics/supply_cooperation_rate",supply_cooperation_rate ,training_ep )
                if enable_stlimit_metrics :
                    tb_writer .add_scalar ("Metrics/station_limit_steps",station_limit_steps ,training_ep )
                    tb_writer .add_scalar ("Metrics/station_limit_hits",station_limit_hits ,training_ep )
                    tb_writer .add_scalar ("Metrics/station_charge_limit_hits",station_charge_limit_hits ,training_ep )
                    tb_writer .add_scalar ("Metrics/station_discharge_limit_hits",station_discharge_limit_hits ,training_ep )
                    tb_writer .add_scalar ("Metrics/station_limit_step_rate",station_limit_steps /max (steps_in_ep ,1 ),training_ep )
                    tb_writer .add_scalar ("Metrics/station_limit_hit_rate",station_limit_hits /max (steps_in_ep *env .num_stations ,1 ),training_ep )
                    tb_writer .add_scalar ("Metrics/station_limit_penalty_total",station_limit_penalty_total ,training_ep )
                    tb_writer .add_scalar ("Metrics/station_limit_penalty_per_step",station_limit_penalty_per_step ,training_ep )
                    tb_writer .add_scalar ("Metrics/station_limit_penalty_per_hit",station_limit_penalty_per_hit ,training_ep )


            # Ctrl+C による中断フラグが立っていればループを抜ける
            if interrupted :
                break

            # GradientLossVisualizer が収集した Q 値・勾配・損失を TensorBoard に書き込む
            if tb_writer and not _is_warmup :
                visualizer .record_to_tensorboard (training_ep )

                # バッファ充填率・epsilon・OUノイズスケール・更新ステップ数を記録する
                try :

                    if hasattr (agent ,'buf'):
                        buf =agent .buf
                        buf_used =int (getattr (buf ,'size',0 ))
                        buf_cap =int (getattr (buf ,'buf_size',0 ))
                        if buf_cap >0 :
                            tb_writer .add_scalar ("Training/buffer_fill_rate",buf_used /buf_cap ,training_ep )
                        if buf_used >0 :
                            tb_writer .add_scalar ("Training/buffer_size",buf_used ,training_ep )


                    if hasattr (agent ,'epsilon'):
                        tb_writer .add_scalar ("Training/epsilon",float (agent .epsilon ),training_ep )
                    if hasattr (agent ,'ou_noise_scale'):
                        tb_writer .add_scalar ("Training/ou_noise_scale",float (agent .ou_noise_scale ),training_ep )


                    if hasattr (agent ,'update_step'):
                        tb_writer .add_scalar ("Training/update_step",int (agent .update_step ),training_ep )
                except Exception :
                    pass

            # ビジュアライザーのエピソード内データをリセットする
            visualizer .reset_episode_data ()



            # ---------------------------------------------------------------
            # 定期評価・グラフ保存のインターバル判定
            # ---------------------------------------------------------------

            should_run_interim_csv =(not _is_warmup )and (training_ep %INTERIM_TEST_INTERVAL ==0 )
            should_save_interim_graph =(not _is_warmup )and (training_ep %INTERIM_PLOT_INTERVAL ==0 )

            # インターバルに達した場合は評価とモデルスナップショット保存を実施する
            if should_run_interim_csv :
                print (f"------------test{training_ep}")
                test_start =time .time ()

                _ =test (agent ,demand_adjustment =None ,random_window =False ,working_dir =working_dir ,
                test_results =None ,test_episode_num =training_ep ,enable_png =should_save_interim_graph ,
                save_test_detail_files =should_run_interim_csv )

                # モデルスナップショットを results/TEST{training_ep}/ に保存する
                try :
                    save_dir =os .path .join (working_dir ,"results",f"TEST{training_ep}")
                    os .makedirs (save_dir ,exist_ok =True )
                    if hasattr (agent ,"save_actors"):
                        agent .save_actors (save_dir ,episode =training_ep )
                    elif hasattr (agent ,"save_models"):
                        agent .save_models (save_dir ,episode =training_ep )
                except Exception as _e :
                    pass
                test_duration =time .time ()-test_start
                print (f"test_done{training_ep} ({test_duration:.1f}s)")
                print (f"======================================")

                # 学習報酬グラフを中間保存する
                try :
                    interim_results_dir =os .path .join (working_dir ,"results")
                    os .makedirs (interim_results_dir ,exist_ok =True )
                    skip_png_flag =bool (not should_save_interim_graph )

                    if len (all_local_rewards )>0 and len (all_global_rewards )>0 :
                        plot_daily_rewards (all_local_rewards ,all_global_rewards ,
                        interim_results_dir ,episode_num =len (all_local_rewards ),
                        performance_metrics =performance_metrics ,title_prefix ="Train Results",
                        skip_png =skip_png_flag )

                    if performance_metrics and len (performance_metrics .get ('soc_miss_count',[]))>0 :
                        plot_performance_metrics (performance_metrics ,interim_results_dir ,title_prefix ="Train Results",
                        skip_png =skip_png_flag )
                except Exception as _e :
                    pass

    except Exception as e :
        print (f"Training interrupted at episode {len(all_rewards)}: {e}")
        traceback .print_exc ()

    # -----------------------------------------------------------------------
    # 学習完了後の最終グラフ保存
    # -----------------------------------------------------------------------
    results_dir =os .path .join (working_dir ,"results")
    os .makedirs (results_dir ,exist_ok =True )

    # 最終報酬グラフを保存する（skip_png=False で PNG も出力する）
    skip_png_flag =False
    if len (all_local_rewards )>0 and len (all_global_rewards )>0 :
        plot_daily_rewards (all_local_rewards ,all_global_rewards ,
        results_dir ,episode_num =len (all_local_rewards ),
        performance_metrics =performance_metrics ,title_prefix ="Train Results",
        skip_png =skip_png_flag )

    else :
        pass

    # 最終パフォーマンス指標グラフを保存する
    if performance_metrics and len (performance_metrics .get ('soc_miss_count',[]))>0 :
        plot_performance_metrics (performance_metrics ,results_dir ,title_prefix ="Train Results",
        skip_png =skip_png_flag )
    else :
        pass

    # エージェント・累積報酬・指標・エピソードデータ・保存先ディレクトリを返す
    return agent ,all_rewards ,performance_metrics ,all_episode_data ,working_dir


if __name__ =="__main__":


     ag ,all_rewards ,perf ,ep_data ,work_dir =train ()
