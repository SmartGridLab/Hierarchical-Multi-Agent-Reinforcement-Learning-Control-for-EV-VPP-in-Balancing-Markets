"""
Config.py — プロジェクト全体のハイパーパラメータ集約モジュール

学習・環境・探索・出力に関するすべての設定値をここで一元管理する。
他モジュールは `import Config` して各定数を参照する。
"""
import os
import torch


# --- 基本設定 ---
ENV_SEED =2545  # 環境の乱数シード（再現性確保のために固定）

DEVICE =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")  # 計算デバイス（CUDA優先、なければCPU）
PROJECT_ROOT =os .path .dirname (os .path .abspath (__file__ ))  # プロジェクトルートのパス（このファイルが置かれているディレクトリ）


# --- 損失・勾配関連 ---
SMOOTHL1_BETA =0.1  # SmoothL1損失のβ（誤差がβ未満ならL2、以上ならL1として扱う境界値）
GRAD_CLIP_MAX =5.0  # ローカルActor/Criticの勾配クリッピング上限ノルム
GRAD_CLIP_MAX_GLOBAL =2.0  # グローバルCriticの勾配クリッピング上限ノルム
BIAS_GRAD_CLIP_MAX =0.1  # バイアスパラメータ専用の勾配クリッピング上限


# --- 報酬関連 ---
LOCAL_R_SOC_HIT =8.0  # EVが目標SoCを達成して出発した際のローカル報酬
GLOBAL_BALANCE_REWARD =0.5  # グローバル需給調整報酬のスケール係数

# SoC達成ボーナス（出発ステップでの追加報酬）
SOC_HIT_BONUS =4.0

LOCAL_DEFICIT_SHAPING_COEF =0.01  # SoC不足量シェーピング報酬の係数
LOCAL_DEFICIT_SIGMA_POS =5.0  # SoC不足ガウスシェーピングの正方向σ（kWh）
LOCAL_DEFICIT_SIGMA_NEG =3.0  # SoC不足ガウスシェーピングの負方向σ（kWh）
LOCAL_DEFICIT_EXP_CLIP =15.0  # exp計算のオーバーフロー防止クリップ値
LOCAL_DEFICIT_BUFFER =10.0  # SoC不足シェーピングで報酬を与え始める余裕値（kWh）


# --- 学習ハイパーパラメータ ---
NUM_EPISODES =10000  # 総学習エピソード数
BATCH_SIZE =1024  # 1回のミニバッチサイズ
GAMMA =0.985  # 割引率（将来報酬の減衰係数）
TAU =0.02  # ローカルネットワーク（Actor/ローカルCritic）のPolyak更新率




TAU_GLOBAL =0.02  # グローバルCriticのPolyak更新率（別管理）




ACTOR_HIDDEN_SIZE =256  # Actorネットワークの隠れ層ユニット数


LR_ACTOR =1e-6  # Actorの学習率
LR_CRITIC_LOCAL =5e-6  # ローカルCriticの学習率
LOCAL_CRITIC_HIDDEN_SIZE =256  # ローカルCriticの隠れ層ユニット数
MEMORY_SIZE =int (1e5 )  # リプレイバッファの容量（transitions数）

# 学習開始前にバッファを蓄積するステップ数
WARMUP_STEPS =10000

# グローバルCritic用学習率・隠れ層サイズ
LR_GLOBAL_CRITIC =2e-6  # グローバルCriticの学習率
GLOBAL_CRITIC_HIDDEN_SIZE =256  # グローバルCriticの隠れ層ユニット数


# --- TD3関連 ---
TD3_SIGMA_LOCAL =0.2  # ローカルCritic用ターゲットポリシースムージングのノイズ標準偏差
TD3_CLIP_LOCAL =0.5  # ローカルCritic用スムージングノイズのクリップ幅


TD3_SIGMA_GLOBAL =0.20  # グローバルCritic用スムージングノイズの標準偏差
TD3_CLIP_GLOBAL =0.7  # グローバルCritic用スムージングノイズのクリップ幅
POLICY_DELAY =2  # Actorを更新するCritic更新何回ごとに1回か（TD3のdelayed policy update）


# --- 勾配ブレンド ---
# Actorの勾配ブレンド比率
# actor_grad = (1 - w) * local_Q_grad + w * global_Q_grad
Q_MIX_GLOBAL_WEIGHT =0.5  # グローバルQ勾配の重み

# --- グローバルCriticアーキテクチャ選択 ---
# False: QMIX型（per-station u_i → softplus重み付き単調混合 → Q_global）
# True:  集中型JointCritic（global_obs + joint_action flatten → MLP → Q_global）
USE_JOINT_CRITIC =True


# --- 環境設定 ---
TOL_NARROW_METRICS =150.0  # SoC目標達成判定の許容幅（kWh）
SOC_WIDE =20  # 到着時SoC分布の広さパラメータ

EV_CAPACITY =100.0  # EVバッテリー容量（kWh）
EPISODE_STEPS =288  # 1エピソードのステップ数（5分×288=24時間）

# EV充放電電力・時間設定
MAX_EV_POWER_KW =27.5  # EV1台の最大充放電電力（kW）
TIME_STEP_MINUTES =5  # 1ステップの時間長（分）
POWER_TO_ENERGY =TIME_STEP_MINUTES /60.0  # 電力→エネルギー変換係数（kW × 5/60 = kWh）

MAX_EV_PER_STATION =10  # 1充電ステーションあたりの最大EVスロット数
EV_SOC_ARRIVAL_DISTRIBUTION_PATH =os .path .join (PROJECT_ROOT ,"data","input_EVinfo","soc_arrival_distribution.csv")
EV_PROFILE_DATA_PATH =os .path .join (PROJECT_ROOT ,"data","input_EVinfo","WORKPLACE_neededsocanddwelltime.csv")
NUM_EVS =10000  # プロファイルプールのEV総数（到着シミュレーション用）
NUM_STATIONS =5  # 充電ステーション数（エージェント数）

# エピソード開始時に各ステーションに配置するEV数
INITIAL_EVS_PER_STATION =3


# --- OUノイズ探索 ---
OU_THETA =0.15  # Ornstein-Uhlenbeckノイズの回帰速度（平均回帰の強さ）
OU_SIGMA =0.5  # OUノイズの拡散係数（ランダム成分の大きさ）
OU_DT =1.0  # OUノイズの時間刻み
OU_INIT_X =0.0  # OUノイズの初期値
OU_CLIP =1.0  # OUノイズのクリップ上限

OU_NOISE_GAIN =3.0  # OUノイズを行動に加算する際のスケール倍率

OU_NOISE_START_EPISODE =1  # OUノイズ適用開始エピソード
OU_NOISE_END_EPISODE =500  # OUノイズが最終スケールに達するエピソード
OU_NOISE_SCALE_INITIAL =1.0  # OUノイズスケールの初期値（大きい＝探索重視）
OU_NOISE_SCALE_FINAL =0.2  # OUノイズスケールの最終値（小さい＝活用重視）


# --- ε-greedy探索 ---
EPSILON_START_EPISODE =1  # ε-greedy開始エピソード
EPSILON_END_EPISODE =100  # εが最終値に達するエピソード
EPSILON_INITIAL =1.0  # εの初期値（完全ランダム探索）
EPSILON_FINAL =0.05  # εの最終値（ほぼ決定論的行動）
RANDOM_ACTION_RANGE =(-1 ,1 )  # ランダム行動のサンプリング範囲




# --- 出力・評価 ---
DEMAND_ADJUSTMENT_DIR =os .path .join (PROJECT_ROOT ,"data","input.demand_fromPJM","output_5min")  # 需給調整量データ（PJM由来）のディレクトリパス




TRAIN_INTERIM_CSV_INTERVAL_EPISODES =10  # 中間テスト・CSV保存の間隔（エピソード数）

TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES =100  # 中間グラフ保存の間隔（エピソード数）


# --- スイッチング制約 ---
USE_SWITCHING_CONSTRAINTS =False  # EV充放電方向切り替え制約の使用フラグ（Trueは非対応）
MAX_SWITCH_COUNT =10 **9  # 最大切り替え回数（制約OFF時は無限大に相当）
LOCAL_SWITCH_PENALTY =0.0  # 方向切り替えペナルティ係数


# --- 到着プロファイル・ステーション電力制限 ---
# 各ステーションの時刻別EV到着確率CSVパス（実測データ）
PER_STATION_ARRIVAL_PROFILE_PATHS =[
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__COMM VITALITY _ 1400 WALNUT1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__COMM VITALITY _ 1104 SPRUCE1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__COMM VITALITY _ 1500PEARL1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__BOULDER _ CARPENTER PARK1.csv"),
os .path .join (PROJECT_ROOT ,"data","input_EVinfo","arrivals_by_station","Arrival__BOULDER _ N BOULDER REC 1.csv"),
]

# ステーション合計電力制限の使用フラグと設定
USE_STATION_TOTAL_POWER_LIMIT =False  # ステーション合計電力制限の使用フラグ
STATION_MAX_TOTAL_POWER_KW =float (MAX_EV_POWER_KW *MAX_EV_PER_STATION )  # ステーション合計電力上限（kW）
LOCAL_STATION_LIMIT_PENALTY =0.0  # ステーション電力超過ペナルティ係数


# --- エージェント選択フラグ ---
USE_MILP =False  # MILPベンチマークエージェントを使用するか

USE_INDEPENDENT_DDPG =False  # 独立DDPGエージェントを使用するか
GLOBAL_REWARD_WEIGHT =0.5  # 独立DDPGでのグローバル報酬重み

# TrueにするとグローバルCriticのみ使用（純粋MADDPG相当）
REGULAR_MADDPG =False

USE_SHARED_OBS_DDPG =False  # 共有観測DDPGエージェントフラグ
USE_SHARED_OBS_SAC =False  # 共有観測SACエージェントフラグ
MEASURE_LP_STEP_TIME =False  # LPソルバー計算時間計測フラグ
MEASURE_STEP_INDEX =0  # 計測対象のステップインデックス

# --- 自動調整 ---
AUTO_ADJUST_MILP_WEIGHTS =False  # MILPの重みを自動調整するか
AUTO_ADJUST_TARGET_SOC_HIT =90.0  # 自動調整の目標SoC達成率（%）
AUTO_ADJUST_TARGET_DISPATCH =90.0  # 自動調整の目標需給追従率（%）
AUTO_ADJUST_MAX_SWITCHES =10.0  # 自動調整の最大許容スイッチ数
AUTO_ADJUST_MAX_EPISODES =5  # 自動調整の最大試行エピソード数


# --- SharedObs用パラメータ（対応するMADDPGパラメータと同値で初期化） ---
SHARED_OBS_LR_ACTOR =LR_ACTOR  # 共有観測DDPGのActor学習率
SHARED_OBS_LR_CRITIC_LOCAL =LR_CRITIC_LOCAL  # 共有観測DDPGのローカルCritic学習率
SHARED_OBS_GLOBAL_REWARD_WEIGHT =GLOBAL_REWARD_WEIGHT  # 共有観測DDPGのグローバル報酬重み
SHARED_OBS_EPSILON_END_EPISODE =EPSILON_END_EPISODE  # 共有観測DDPGのε終了エピソード
SHARED_OBS_OU_NOISE_END_EPISODE =OU_NOISE_END_EPISODE  # 共有観測DDPGのOUノイズ終了エピソード
SHARED_OBS_OU_NOISE_SCALE_FINAL =OU_NOISE_SCALE_FINAL  # 共有観測DDPGのOUノイズ最終スケール

# --- SAC（Soft Actor-Critic）固有パラメータ ---
SAC_LR_ACTOR =LR_ACTOR  # SACのActor学習率
SAC_LR_CRITIC =LR_CRITIC_LOCAL  # SACのCritic学習率
SAC_LR_ALPHA =1e-4  # エントロピー温度αの学習率
SAC_TAU =TAU  # SACのPolyak更新率
SAC_ALPHA_INIT =0.2  # エントロピー温度αの初期値
SAC_TARGET_ENTROPY_SCALE =1.0  # 目標エントロピーのスケール係数
SAC_GLOBAL_REWARD_WEIGHT =GLOBAL_REWARD_WEIGHT  # SACのグローバル報酬重み


# --- MILP ---
MILP_W_AG =1.0  # MILPの需給調整目標重み
MILP_W_SOC =100.0  # MILPのSoC達成目標重み
MILP_AG_DEADBAND =10  # MILPの需給調整デッドバンド（kW）
MILP_HORIZON =1  # MILPの最適化ホライズン（ステップ数）
MILP_DEADBAND_PENALTY =1.0  # デッドバンド超過ペナルティ係数
MILP_SOC_PENALTY =2.0  # SoC未達ペナルティ係数
