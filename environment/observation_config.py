"""
observation_config.py
=====================
観測の構成フラグと次元計算モジュール。

ローカル観測（各エージェントが自局のみ見る）:
- LOCAL_EVINFO_PRESENCE: EVの在否フラグを観測に含めるか
- LOCAL_EVINFO_SOC: EVのSoCを含めるか
- LOCAL_EVINFO_REMAINING_TIME: 残り滞在時間を含めるか
- LOCAL_EVINFO_NEEDED_SOC: 必要SoCを含めるか
- LOCAL_USE_STATION_POWER: 前ステップのステーション合計電力を末尾特徴に含めるか
- LOCAL_DEMAND_STEPS: 需給調整量先読みステップ数（末尾特徴に追加）
- LOCAL_USE_STEP: 現在ステップ番号を末尾特徴に含めるか

グローバル観測（グローバルCriticが全ステーションの情報を見る）:
- GLOBAL_EVINFO_*: グローバル観測でのEV特徴量フラグ
- GLOBAL_DEMAND_STEPS: グローバル観測の需給先読みステップ数
- GLOBAL_USE_STEP / GLOBAL_USE_TOTAL_POWER: グローバル観測への追加フラグ

安全アサーション:
- USE_SWITCHING_CONSTRAINTS=True は非対応（EV_FEAT_DIM=6になりnormalize.pyとGlobalMLPCriticが壊れる）
- GLOBAL_USE_TOTAL_POWER / GLOBAL_USE_STEP は常にTrueでなければならない（GlobalMLPCritic前提）

EV_FEAT_DIM: 1EVスロットの特徴量次元数（PRESENCE+SOC+TIME+NEED+スイッチ特徴量）
"""




# ─── ローカル観測フラグ ───────────────────────────────────────────────────────
# 各エージェントが自局ステーションのEVについて観測する特徴量を制御する

LOCAL_EVINFO_PRESENCE =True  # EVスロットに車両が存在するかを示すフラグ（0 or 1）
LOCAL_EVINFO_SOC =True       # 現在の充電残量（SoC, kWh）
LOCAL_EVINFO_REMAINING_TIME =True  # 出発までの残りステップ数
LOCAL_EVINFO_NEEDED_SOC =True  # 出発時刻までに追加で必要なSoC量（kWh）


# 前ステップのステーション合計電力（kW）を末尾特徴として追加するか
LOCAL_USE_STATION_POWER =True


# 需給調整量（AG要請）の先読みステップ数
# 1 の場合は現在ステップの需給調整量のみを末尾に付加する
LOCAL_DEMAND_STEPS =1


# 現在のエピソードステップ番号を末尾特徴として追加するか
LOCAL_USE_STEP =True




# ─── グローバル観測フラグ ─────────────────────────────────────────────────────
# GlobalMLPCriticが全ステーションを横断して参照する特徴量フラグ

GLOBAL_EVINFO_PRESENCE =True  # グローバル観測でもEVの在否を含める
GLOBAL_EVINFO_SOC =True       # グローバル観測でもSoCを含める
GLOBAL_EVINFO_REMAINING_TIME =True  # グローバル観測でも残り時間を含める
GLOBAL_EVINFO_NEEDED_SOC =True  # グローバル観測でも必要SoCを含める


# グローバル観測の需給調整量先読みステップ数
GLOBAL_DEMAND_STEPS =1

# 現在ステップ番号をグローバル観測に含めるか（GlobalMLPCriticが必須で使用）
GLOBAL_USE_STEP =True
# 全ステーション合計電力をグローバル観測に含めるか（GlobalMLPCriticが必須で使用）
GLOBAL_USE_TOTAL_POWER =True
# 予測合計電力をグローバル観測に含めるか
GLOBAL_USE_PRED_TOTAL_POWER =True




# ─── 安全アサーション ─────────────────────────────────────────────────────────

try :
    from Config import USE_SWITCHING_CONSTRAINTS
except Exception :
    USE_SWITCHING_CONSTRAINTS =False

# ローカル観測のベースEV特徴量次元数（フラグの合計）
_LOCAL_BASE_EV_FEAT_DIM =(
(1 if LOCAL_EVINFO_PRESENCE else 0 )
+(1 if LOCAL_EVINFO_SOC else 0 )
+(1 if LOCAL_EVINFO_REMAINING_TIME else 0 )
+(1 if LOCAL_EVINFO_NEEDED_SOC else 0 )
)
# グローバル観測のベースEV特徴量次元数（フラグの合計）
_GLOBAL_BASE_EV_FEAT_DIM =(
(1 if GLOBAL_EVINFO_PRESENCE else 0 )
+(1 if GLOBAL_EVINFO_SOC else 0 )
+(1 if GLOBAL_EVINFO_REMAINING_TIME else 0 )
+(1 if GLOBAL_EVINFO_NEEDED_SOC else 0 )
)
# スイッチング制約が有効な場合に追加されるEV特徴量次元数（switch_count, last_dir）
_SWITCH_EXTRA_EV_FEAT_DIM =2 if bool (USE_SWITCHING_CONSTRAINTS )else 0


# USE_SWITCHING_CONSTRAINTS=True はサポート外
# （EV_FEAT_DIM が 6 になり normalize.py および GlobalMLPCritic が壊れるため）
if bool (USE_SWITCHING_CONSTRAINTS ):
    raise AssertionError (
    "USE_SWITCHING_CONSTRAINTS=True is not supported: normalize.py and GlobalMLPCritic "
    "both assume 4 EV features per EV slot.  Keep USE_SWITCHING_CONSTRAINTS=False."
    )

# GLOBAL_USE_TOTAL_POWER は必ず True でなければならない
# （GlobalMLPCritic.forward() が常に total_ev_power を読み取るため）
if not GLOBAL_USE_TOTAL_POWER :
    raise AssertionError (
    "GLOBAL_USE_TOTAL_POWER must be True: GlobalMLPCritic.forward() always reads "
    "total_ev_power from the global observation and has no code path for False."
    )

# GLOBAL_USE_STEP は必ず True でなければならない
# （GlobalMLPCritic.forward() が常に current_step を読み取るため）
if not GLOBAL_USE_STEP :
    raise AssertionError (
    "GLOBAL_USE_STEP must be True: GlobalMLPCritic.forward() always reads "
    "current_step from the global observation and has no code path for False."
    )


# ローカルとグローバルのベースEV特徴量次元数は一致していなければならない
if _LOCAL_BASE_EV_FEAT_DIM !=_GLOBAL_BASE_EV_FEAT_DIM :
    raise ValueError (
    "Local/Global EV base feature dims must match: "
    f"local={_LOCAL_BASE_EV_FEAT_DIM}, global={_GLOBAL_BASE_EV_FEAT_DIM}"
    )

# 1EVスロットあたりの総特徴量次元数（通常4、スイッチング有効時は6）
EV_FEAT_DIM =_LOCAL_BASE_EV_FEAT_DIM +_SWITCH_EXTRA_EV_FEAT_DIM
