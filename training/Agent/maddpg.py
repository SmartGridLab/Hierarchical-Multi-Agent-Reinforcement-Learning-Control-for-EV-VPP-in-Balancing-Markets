"""
MADDPG エージェント（Multi-Agent DDPG with CTDE + TD3 + QMIX）

アーキテクチャ概要:
- CTDE (Centralized Training, Decentralized Execution):
  実行時は各エージェントが自局観測のみで行動し、学習時は全エージェントの情報を利用する。
- n_agent 個の独立した Actor（各充電ステーション = 1 エージェント）を持つ。
- TD3 型のローカル Critic が各エージェントに 2 つ（critics, critics2）。
- 全エージェント共有のグローバル QMIX Critic が 2 つ（global_critic1, global_critic2）。

主要アルゴリズム:
- TD3 双子 Critic: 2 つの Q 値の最小値をターゲットに使用し、過大評価を防ぐ。
- ターゲットポリシースムージング: 次状態行動にクリップ付きノイズを加えて Q 値を平滑化する。
- 遅延ポリシー更新 (Policy Delay): Critic を POLICY_DELAY 回更新するごとに Actor を 1 回更新する。
- Actor 勾配ブレンド: ローカルとグローバルの Q 値から受け取る勾配を w_eff で加重混合する。
- STE (Straight-Through Estimator): SoC クリップ時に勾配を通すための推定器。
- Polyak 平均: ソフトターゲット更新 (τ, τ_global)。
"""
import copy
import math
import os

import torch
import torch .nn as nn
import torch .nn .functional as F
import torch .optim as optim

from environment.normalize import denormalize_soc
from Config import (
NUM_EPISODES ,BATCH_SIZE ,GAMMA ,TAU ,TAU_GLOBAL ,
ACTOR_HIDDEN_SIZE ,LOCAL_CRITIC_HIDDEN_SIZE ,GLOBAL_CRITIC_HIDDEN_SIZE ,
LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC ,
RANDOM_ACTION_RANGE ,SMOOTHL1_BETA ,
EPSILON_START_EPISODE ,EPSILON_END_EPISODE ,EPSILON_INITIAL ,EPSILON_FINAL ,
OU_NOISE_START_EPISODE ,OU_NOISE_END_EPISODE ,
OU_NOISE_SCALE_INITIAL ,OU_NOISE_SCALE_FINAL ,OU_NOISE_GAIN ,
TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,TD3_SIGMA_LOCAL ,TD3_CLIP_LOCAL ,
POLICY_DELAY ,
MEMORY_SIZE ,WARMUP_STEPS ,
Q_MIX_GLOBAL_WEIGHT ,
EV_CAPACITY ,POWER_TO_ENERGY ,
REGULAR_MADDPG ,
MAX_EV_POWER_KW ,MAX_EV_PER_STATION ,
BIAS_GRAD_CLIP_MAX ,GRAD_CLIP_MAX ,GRAD_CLIP_MAX_GLOBAL ,
USE_JOINT_CRITIC ,
)

from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_USE_STATION_POWER ,LOCAL_DEMAND_STEPS ,LOCAL_USE_STEP ,
GLOBAL_DEMAND_STEPS ,GLOBAL_USE_STEP ,GLOBAL_USE_TOTAL_POWER ,
)

try :
    from .actor import Actor
    from .critic import LocalEvMLPCritic ,GlobalMLPCritic ,CentralizedJointCritic
    from .replay_buffer import ReplayBuffer
    from .noise import (
    OUNoise ,
    linear_epsilon_decay ,
    sample_epsilon_random_action ,
    should_take_random_action ,
    )
except Exception :
    from actor import Actor
    from critic import LocalEvMLPCritic ,GlobalMLPCritic ,CentralizedJointCritic
    from replay_buffer import ReplayBuffer
    from noise import (
    OUNoise ,
    linear_epsilon_decay ,
    sample_epsilon_random_action ,
    should_take_random_action ,
    )


# GPU が利用可能であれば CUDA デバイスを使用する
device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
if torch .cuda .is_available ():
    torch .set_default_tensor_type ('torch.cuda.FloatTensor')


def _clip_bias_gradients (model ,max_norm =1.0 ):
    # バイアスパラメータの勾配のみを個別にクリップする（重みとは別管理）
    for name ,param in model .named_parameters ():
        if param .grad is not None and 'bias'in name :
            torch .nn .utils .clip_grad_norm_ (param ,max_norm )


class MADDPG :
    def __init__ (self ,s_dim ,max_evs_per_station ,n_agent ,
    gamma =GAMMA ,
    tau =TAU ,
    batch =BATCH_SIZE ,
    lr_a =LR_ACTOR ,lr_c =LR_CRITIC_LOCAL ,lr_global_c =LR_GLOBAL_CRITIC ,
    num_episodes =NUM_EPISODES ,
    tau_global :float =TAU_GLOBAL ,
    td3_sigma :float =TD3_SIGMA_GLOBAL ,
    td3_clip :float =TD3_CLIP_GLOBAL ,
    td3_sigma_local :float =TD3_SIGMA_LOCAL ,
    td3_clip_local :float =TD3_CLIP_LOCAL ,
    smoothl1_beta =1.0 ,
    **kwargs ):
        # max_evs_per_station と Config の値が一致しない場合はシェイプの不整合が生じるため即座に停止
        if max_evs_per_station !=MAX_EV_PER_STATION :
            raise AssertionError (
            f"max_evs_per_station={max_evs_per_station} != Config.MAX_EV_PER_STATION={MAX_EV_PER_STATION}. "
            "GlobalMLPCritic and normalize.py use Config.MAX_EV_PER_STATION directly; "
            "passing a different value causes silent shape mismatches."
            )
        # 基本次元情報: s_dim=ローカル観測次元, a_dim=EV スロット数（＝行動次元）, n=エージェント数
        self .s_dim ,self .a_dim ,self .n =s_dim ,max_evs_per_station ,n_agent
        self .max_ev_per_station =max_evs_per_station
        # 学習ハイパーパラメータ
        self .gamma ,self .tau ,self .batch =gamma ,tau ,batch
        self .tau_global =tau_global  # グローバル Critic 用の Polyak 係数（ローカルとは別管理）
        self .lr_a =lr_a
        self .lr_c =lr_c
        self .lr_global_c =lr_global_c
        self .total_episodes =num_episodes
        self .current_episode =0

        self .hidden_size =ACTOR_HIDDEN_SIZE

        # ---- ログ用変数（勾配ノルム・損失・クリップ回数など）----
        self .actor_norms =[0 ]*n_agent
        self .critic_norms =[0 ]*n_agent
        self .actor_losses =[]
        self .critic_losses =[]
        self .last_global_critic_loss =0.0
        self .last_actor_loss =0.0
        self .last_critic_loss =0.0
        self .last_actor_grad_norm =0.0
        self .last_local_critic_grad_norm =0.0
        self .last_global_critic_grad_norm =0.0
        self .last_actor_clip_count =0
        self .last_local_critic_clip_count =0
        self .last_global_critic_clip_count =0
        self .local_critic_clip_counts =[0 ]*n_agent
        self .actor_clip_counts =[0 ]*n_agent
        self .critic_norms_before_clip =[0 ]*n_agent
        self .actor_norms_before_clip =[0 ]*n_agent
        # Actor 勾配のローカル/グローバル成分のノルムと比率・コサイン類似度
        self .actor_source_local_norms_before_clip =[0.0 ]*n_agent
        self .actor_source_global_norms_before_clip =[0.0 ]*n_agent
        self .actor_source_global_ratio =[0.0 ]*n_agent
        self .actor_source_cos =[0.0 ]*n_agent
        self .actor_source_cos_valid =[0 ]*n_agent
        self .last_global_critic_grad_norm_before_clip =0.0
        self .last_actor_source_local_grad_norm_before_clip =0.0
        self .last_actor_source_global_grad_norm_before_clip =0.0
        self .last_actor_source_global_ratio =0.0
        self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =0.0
        self .last_local_q_values_per_agent =[0.0 ]*n_agent
        self .last_global_q_value =0.0

        self .env =None
        self .use_tensorboard =False
        self .writer =None

        self .random_action_range =RANDOM_ACTION_RANGE

        # ---- ε-greedy 探索パラメータ（線形減衰） ----
        self .epsilon_start_episode =EPSILON_START_EPISODE
        self .epsilon_end_episode =EPSILON_END_EPISODE
        self .epsilon_initial =EPSILON_INITIAL
        self .epsilon_final =EPSILON_FINAL
        self .epsilon =self .epsilon_initial

        # ---- OUノイズ探索パラメータ（線形スケール減衰） ----
        self .ou_noise_start_episode =OU_NOISE_START_EPISODE
        self .ou_noise_end_episode =OU_NOISE_END_EPISODE
        self .ou_noise_scale_initial =OU_NOISE_SCALE_INITIAL
        self .ou_noise_scale_final =OU_NOISE_SCALE_FINAL
        self .ou_noise_scale =self .ou_noise_scale_initial

        # OUノイズオブジェクト: エージェント数 × EV スロット数の時系列相関ノイズを生成
        self .ou_noise =OUNoise (n_agent ,max_evs_per_station )

        self .test_mode =False

        # リプレイバッファ: 経験を蓄積してミニバッチサンプリングを行う
        self .buf =ReplayBuffer (cap =int (MEMORY_SIZE ))
        self .buf .maddpg_ref =self
        self .max_evs =max_evs_per_station

        self .active_evs =[0 ]*n_agent

        # ---- ローカル観測の次元分解 ----
        # EV 特徴量ブロック + ステーション電力・需要予測・タイムステップなどのテール次元
        self .ev_state_dim =EV_FEAT_DIM
        self .local_tail_dim =(
        (1 if LOCAL_USE_STATION_POWER else 0 )
        +int (LOCAL_DEMAND_STEPS )
        +(1 if LOCAL_USE_STEP else 0 )
        )
        self .station_state_dim =self .ev_state_dim *self .max_evs +self .local_tail_dim

        # ---- Actor ネットワーク（n_agent 個）+ ターゲットネットワーク ----
        self .actors =[
        Actor (s_dim ,max_evs_per_station ,station_state_dim =self .station_state_dim ).to (device )
        for _ in range (n_agent )
        ]
        self .t_actors =[copy .deepcopy (ac )for ac in self .actors ]

        # ---- ローカル Critic ネットワーク（TD3 用の双子構成）+ ターゲットネットワーク ----
        self .critics =[
        LocalEvMLPCritic (
        ev_feat_dim =EV_FEAT_DIM ,
        a_dim =max_evs_per_station ,
        max_evs =max_evs_per_station ,
        hid =LOCAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self .station_state_dim ,
        ).to (device )
        for _ in range (n_agent )
        ]
        self .critics2 =[
        LocalEvMLPCritic (
        ev_feat_dim =EV_FEAT_DIM ,
        a_dim =max_evs_per_station ,
        max_evs =max_evs_per_station ,
        hid =LOCAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self .station_state_dim ,
        ).to (device )
        for _ in range (n_agent )
        ]
        self .t_critics =[copy .deepcopy (cr )for cr in self .critics ]
        self .t_critics2 =[copy .deepcopy (cr )for cr in self .critics2 ]

        # ---- グローバル Critic（全エージェント共有、双子構成）+ ターゲットネットワーク ----
        # USE_JOINT_CRITIC=False: QMIX型（u_i → softplus重み付き単調混合 → Q_global）
        # USE_JOINT_CRITIC=True:  集中型JointCritic（global_obs + joint_action → MLP → Q_global）
        # グローバル観測次元: 全ステーションの EV 特徴量 + 合計電力・タイムステップ・需要予測
        additional_features =(
        (1 if GLOBAL_USE_TOTAL_POWER else 0 )
        +(1 if GLOBAL_USE_STEP else 0 )
        +int (GLOBAL_DEMAND_STEPS )
        )
        self ._global_station_dim =EV_FEAT_DIM *self .max_evs
        global_obs_dim =(self .n *self ._global_station_dim )+additional_features

        # フラグに応じてグローバルCriticクラスを選択
        _GlobalCriticCls =CentralizedJointCritic if USE_JOINT_CRITIC else GlobalMLPCritic
        self .global_critic1 =_GlobalCriticCls (
        global_obs_dim ,max_evs_per_station ,n_agent ,
        hid =GLOBAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self ._global_station_dim ,
        init_gain =0.3 ,
        ).to (device )
        self .global_critic2 =_GlobalCriticCls (
        global_obs_dim ,max_evs_per_station ,n_agent ,
        hid =GLOBAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self ._global_station_dim ,
        init_gain =0.3 ,
        ).to (device )
        self .t_global_critic1 =copy .deepcopy (self .global_critic1 )
        self .t_global_critic2 =copy .deepcopy (self .global_critic2 )

        # 後方互換性のためのエイリアス（global_critic1 を主 Critic として参照）
        self .global_critic =self .global_critic1
        self .t_global_critic =self .t_global_critic1

        # ---- Adam オプティマイザ（Actor・ローカル Critic・グローバル Critic） ----
        self .opt_a =[optim .Adam (self .actors [i ].parameters (),lr =lr_a )for i in range (n_agent )]
        self .opt_c =[optim .Adam (self .critics [i ].parameters (),lr =lr_c )for i in range (n_agent )]
        self .opt_c2 =[optim .Adam (self .critics2 [i ].parameters (),lr =lr_c )for i in range (n_agent )]
        self .opt_global_c1 =optim .Adam (self .global_critic1 .parameters (),lr =self .lr_global_c )
        self .opt_global_c2 =optim .Adam (self .global_critic2 .parameters (),lr =self .lr_global_c )

        # Huber 損失（SmoothL1）: β 以下の誤差は MSE、それ以上は MAE として扱う
        self .loss_fn =nn .SmoothL1Loss (beta =smoothl1_beta )

        self .clip_bias_gradients =_clip_bias_gradients

        self .model_name =""

        # エピソード中の Q 値ログ（グローバル・ローカル・混合）
        self .episode_global_q_values =[]
        self .episode_local_q_values =[]
        self ._ep_q_raw_global =[]
        self ._ep_q_raw_local =[[]for _ in range (n_agent )]
        self ._ep_q_raw_combined =[[]for _ in range (n_agent )]

        # ---- TD3 ハイパーパラメータ ----
        # policy_delay: Critic を何回更新するごとに Actor を 1 回更新するか
        self .policy_delay =max (1 ,int (POLICY_DELAY ))
        self .td3_sigma =td3_sigma      # グローバル Critic 用ターゲットスムージングノイズ標準偏差
        self .td3_clip =td3_clip        # グローバル Critic 用ノイズのクリップ幅
        self .td3_sigma_local =td3_sigma_local  # ローカル Critic 用ターゲットスムージングノイズ標準偏差
        self .td3_clip_local =td3_clip_local    # ローカル Critic 用ノイズのクリップ幅

        self .update_step =0
        self .local_update_interval =1


    def update_active_evs (self ,env ):
        # 現在の環境状態から各ステーションのアクティブ EV 数を取得して更新する
        num_stations =min (self .n ,env .num_stations )
        counts =env .ev_mask [:num_stations ].sum (dim =1 ).cpu ().tolist ()
        if self .n >num_stations :
            # エージェント数が実ステーション数を超える場合は 0 で補完
            counts .extend ([0 ]*(self .n -num_stations ))
        self .active_evs =counts
        self .env =env


    def _convert_to_global_critic_obs (self ,s ,actual_station_powers ):
        """
        ローカル観測スタック s からグローバル Critic 用の観測テンソルを生成する。

        ローカル観測 s (shape: [B, n_agent, station_state_dim]) を変換し、
        以下の要素を結合したグローバル観測テンソルを返す:
          - 全ステーションの EV 特徴量（フラット化）
          - 合計 EV 電力（GLOBAL_USE_TOTAL_POWER=True の場合、正規化済み）
          - 現在タイムステップ（GLOBAL_USE_STEP=True の場合）
          - 需要予測ルックアヘッド（GLOBAL_DEMAND_STEPS > 0 の場合）

        Args:
            s: ローカル観測テンソル [B, n_agent, station_state_dim]
            actual_station_powers: 各ステーションの実際の電力 [B, n_agent]

        Returns:
            global_obs: グローバル Critic 用観測テンソル [B, global_obs_dim]
        """
        B =s .size (0 )

        # EV 特徴量ブロックのみを抽出してフラット化（全エージェント分を結合）
        ev_features_per_station =self .max_evs *EV_FEAT_DIM
        ev_features_all =s [:,:,:ev_features_per_station ]
        station_features_flat =ev_features_all .reshape (B ,-1 )

        if GLOBAL_USE_TOTAL_POWER :
            # 全ステーションの電力合計を最大可能電力で正規化し [-1, 1] にクリップ
            MAX_POSSIBLE_POWER =MAX_EV_POWER_KW *self .n *self .max_evs
            total_ev_power_raw =actual_station_powers .sum (dim =1 ,keepdim =True )
            total_ev_power =torch .clamp (total_ev_power_raw /MAX_POSSIBLE_POWER ,-1.0 ,1.0 )
        else :
            total_ev_power =s .new_zeros (B ,1 )

        if GLOBAL_USE_STEP :
            # ローカル観測の最終次元をタイムステップ特徴として流用
            current_step =s [:,0 ,-1 :]
        else :
            current_step =s .new_zeros (B ,1 )

        # 需要予測ルックアヘッドをローカル観測のテール部分から抽出
        L_global =int (GLOBAL_DEMAND_STEPS )
        if L_global >0 :
            tail =s [:,0 ,ev_features_per_station :]
            tail_idx =0
            if LOCAL_USE_STATION_POWER :
                tail_idx +=1

            ag_local =None
            if LOCAL_DEMAND_STEPS >0 and tail .size (1 )>=tail_idx +LOCAL_DEMAND_STEPS :
                ag_local =tail [:,tail_idx :tail_idx +LOCAL_DEMAND_STEPS ]

            if ag_local is not None :
                if LOCAL_DEMAND_STEPS >=L_global :
                    # ローカルの需要予測が十分長い場合は先頭 L_global ステップを使用
                    ag_lookahead =ag_local [:,:L_global ]
                else :
                    # 短い場合はゼロパディングで L_global まで補完
                    pad =s .new_zeros (B ,L_global -LOCAL_DEMAND_STEPS )
                    ag_lookahead =torch .cat ([ag_local ,pad ],dim =1 )
            else :
                ag_lookahead =s .new_zeros (B ,L_global )
        else :
            ag_lookahead =s .new_zeros (B ,0 )

        # 有効なフラグに従って各パーツを結合してグローバル観測を構築
        parts =[station_features_flat ]
        if GLOBAL_USE_TOTAL_POWER :
            parts .append (total_ev_power )
        if GLOBAL_USE_STEP :
            parts .append (current_step )
        parts .append (ag_lookahead )
        global_obs =torch .cat (parts ,dim =1 )
        return global_obs


    def _apply_soc_constraint (self ,actions_kw ,current_socs ,ev_padding_mask =None ,use_ste =False ):
        """
        行動（kW）を SoC 制約でクリップし、各ステーションの合計電力を計算する。

        充電量は (EV_CAPACITY - SoC) を、放電量は SoC を上限としてクリップする。
        use_ste=True の場合、STE（Straight-Through Estimator）を適用し、
        前向きパスはクリップ後の値、後向きパスはクリップ前の値をそのまま使う。
        これにより Actor の勾配計算時に SoC 制約を通じて勾配を伝播させられる。

        Args:
            actions_kw: 提案行動 [B, n_agent, max_evs] (kW 単位)
            current_socs: 現在の SoC [B, n_agent, max_evs] (kWh 単位)
            ev_padding_mask: EV 非存在スロットのマスク [B, n_agent, max_evs]
            use_ste: True の場合 STE を適用して勾配を通す

        Returns:
            clamped_actions_kw: SoC 制約適用後の行動 [B, n_agent, max_evs] (kW)
            station_powers_kw: 各ステーションの合計電力 [B, n_agent] (kW)
        """
        proposed_delta_kwh =actions_kw *POWER_TO_ENERGY
        # 充電上限: EV_CAPACITY - 現在 SoC、放電上限: 現在 SoC
        max_charge =EV_CAPACITY -current_socs
        max_discharge =current_socs
        clamped_kwh =torch .clamp (proposed_delta_kwh ,-max_discharge ,max_charge )
        if use_ste :
            # STE: 前向きはクリップ後、後向きはクリップ前をそのまま通す
            # clamped = proposed + (clipped - proposed).detach() という実装
            clamped_kwh =proposed_delta_kwh +(clamped_kwh -proposed_delta_kwh ).detach ()
        clamped_actions_kw =clamped_kwh /POWER_TO_ENERGY
        if ev_padding_mask is not None :
            try :
                # EV が存在しないスロットの行動を 0 に設定
                clamped_actions_kw =clamped_actions_kw .masked_fill (ev_padding_mask ,0.0 )
            except Exception :
                pass
        # 各ステーション内の全 EV 電力を合計してステーション単位の電力を算出
        station_powers_kw =clamped_actions_kw .sum (dim =2 )
        return clamped_actions_kw ,station_powers_kw


    def act (self ,state ,env =None ,noise =True ):
        # テストモード時はノイズを無効にして決定論的行動を返す
        if hasattr (self ,'test_mode')and self .test_mode :
            noise =False

        if env is not None :
            # 環境から各ステーションのアクティブ EV 数を更新
            self .update_active_evs (env )

        # EV が 1 台以上いるエージェントのみ行動を計算する
        active_agents =[i for i ,num in enumerate (self .active_evs )if num >0 ]
        tensor_actions =torch .zeros ((self .n ,self .max_evs ),dtype =torch .float32 ,device =device )

        with torch .no_grad ():
            ou_step_noise =None
            for agent_idx in active_agents :
                num_active =self .active_evs [agent_idx ]
                agent_state =state [agent_idx :agent_idx +1 ]
                # Actor の前向き計算（CTDE の分散実行フェーズ: 自局観測のみを入力）
                a =self .actors [agent_idx ](agent_state )

                if noise and not getattr (self ,'test_mode',False ):
                    try :
                        if should_take_random_action (self .epsilon ):
                            # ε-greedy: ε確率でランダム行動を選択
                            a [0 ,:num_active ]=sample_epsilon_random_action (
                            num_active ,
                            action_range =self .random_action_range ,
                            like_tensor =a [0 ,:num_active ],
                            )
                        else :
                            # OUノイズを行動に加算（時系列相関のある探索ノイズ）
                            if hasattr (self ,'ou_noise')and getattr (self ,'ou_noise_scale',0.0 )>0.0 :
                                if ou_step_noise is None :
                                    # 全エージェント分のノイズを一括サンプリング
                                    ou_step_noise =self .ou_noise .sample (self .active_evs )
                                a [0 ,:num_active ]=(
                                a [0 ,:num_active ]
                                +(OU_NOISE_GAIN *self .ou_noise_scale )
                                *ou_step_noise [agent_idx ,:num_active ]
                                )
                    except Exception :
                        pass

                # 行動を [-1, 1] の正規化範囲にクリップ
                a =torch .clamp (a ,-1.0 ,1.0 )
                tensor_actions [agent_idx ,:num_active ]=a .squeeze (0 )[:num_active ]

        return tensor_actions


    def _zero_update_logs (self ):
        # ウォームアップ中やテストモード時にログ変数をゼロリセットする
        self .last_actor_loss =0.0
        self .last_critic_loss =0.0
        self .last_actor_grad_norm =0.0
        self .last_local_critic_grad_norm =0.0
        self .last_global_critic_grad_norm =0.0
        self .last_actor_source_local_grad_norm_before_clip =0.0
        self .last_actor_source_global_grad_norm_before_clip =0.0
        self .last_actor_source_global_ratio =0.0
        self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =0.0


    def _build_update_ctx (self ,s ,s2 ,a ,r_local ,r_global ,d ,
    actual_station_powers ,actual_ev_power_kw ):
        # リプレイバッファからサンプルした経験をもとに、各更新サブメソッドで共有するコンテキスト辞書を生成する
        batch_size =s .size (0 )
        max_evs =self .a_dim

        # 実際の EV 電力が提供されている場合は正規化して使用し、なければ Actor 出力をそのまま使う
        a_actual =(
        torch .clamp (actual_ev_power_kw /MAX_EV_POWER_KW ,-1.0 ,1.0 )
        if actual_ev_power_kw is not None else a
        )

        # EV 特徴量ブロックを分解してパディングマスクを生成
        # presence_mask: SoC 特徴量（インデックス 0）が 0.5 以下のスロットを非存在と判定
        ev_block =s [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,self .max_evs ,EV_FEAT_DIM )
        presence_mask =(ev_block [...,0 ]<=0.5 )
        ev_padding_mask =presence_mask
        # key_padding_mask: ステーション内の全 EV スロットが空の場合に True
        key_padding_mask =presence_mask .all (dim =2 )

        # 次状態 s2 のパディングマスクも同様に生成
        ev_block_s2 =s2 [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,self .max_evs ,EV_FEAT_DIM )
        presence_mask_s2 =(ev_block_s2 [...,0 ]<=0.5 )
        ev_padding_mask_s2 =presence_mask_s2
        key_padding_mask_s2 =presence_mask_s2 .all (dim =2 )

        # skip_local / skip_global フラグ: Q_MIX_GLOBAL_WEIGHT の端点（0.0, 1.0）で不要な計算をスキップ
        skip_local =(Q_MIX_GLOBAL_WEIGHT ==1.0 )
        skip_global =(Q_MIX_GLOBAL_WEIGHT ==0.0 )
        if REGULAR_MADDPG :
            # REGULAR_MADDPG=True の場合はグローバル Q のみを使用（ローカルをスキップ）
            skip_local =True
            skip_global =False

        max_station_power =MAX_EV_POWER_KW *MAX_EV_PER_STATION
        # w_eff: Actor 勾配のグローバル Q 成分の重み（REGULAR_MADDPG=True なら 1.0 固定）
        w_eff =1.0 if REGULAR_MADDPG else Q_MIX_GLOBAL_WEIGHT

        return {
        's':s ,'s2':s2 ,'a':a ,
        'r_local':r_local ,'r_global':r_global ,'d':d ,
        'actual_station_powers':actual_station_powers ,
        'a_actual':a_actual ,
        'batch_size':batch_size ,'max_evs':max_evs ,
        'ev_block_s2':ev_block_s2 ,
        'ev_padding_mask':ev_padding_mask ,
        'ev_padding_mask_s2':ev_padding_mask_s2 ,
        'key_padding_mask':key_padding_mask ,
        'key_padding_mask_s2':key_padding_mask_s2 ,
        'skip_local':skip_local ,'skip_global':skip_global ,
        'max_station_power':max_station_power ,
        'w_eff':w_eff ,
        }


    def _update_local_critics (self ,ctx ):
        """
        各エージェントのローカル Critic（TD3 双子構成）を更新する。

        TD3 ターゲット計算:
          1. ターゲット Actor で次状態の行動を計算し、クリップ付きノイズを加える（ターゲットポリシースムージング）
          2. SoC 制約を適用してクリップ後の行動と次状態ステーション電力を得る
          3. 双子ターゲット Critic の最小 Q 値を使ってベルマンターゲット y を計算（過大評価防止）
          4. 現在の Critic の Q 値と y の SmoothL1 損失を逆伝播し、勾配クリップ後に Adam で更新

        skip_local=True（Q_MIX_GLOBAL_WEIGHT==1.0 または REGULAR_MADDPG=True）の場合は何もしない。

        Returns:
            local_critic_clip_count: 勾配クリップが発生した回数の合計
        """
        # グローバル Q のみを使う設定の場合はローカル Critic 更新をスキップ
        if ctx ['skip_local']:
            self .critic_losses =[]
            return 0

        s ,s2 =ctx ['s'],ctx ['s2']
        a_actual =ctx ['a_actual']
        r_local ,d =ctx ['r_local'],ctx ['d']
        actual_station_powers =ctx ['actual_station_powers']
        ev_padding_mask_s2 =ctx ['ev_padding_mask_s2']
        max_station_power =ctx ['max_station_power']
        ev_block_s2 =ctx ['ev_block_s2']
        local_critic_clip_count =0

        with torch .no_grad ():
            # ターゲット Actor で次状態行動を計算（全エージェント一括）
            next_actions_all =torch .stack (
            [self .t_actors [i ](s2 [:,i ,:])for i in range (self .n )],dim =1
            )
            # TD3 ターゲットポリシースムージング: クリップ付きガウスノイズを加算
            ta_noise =torch .randn_like (next_actions_all )*self .td3_sigma_local
            ta_noise =torch .clamp (ta_noise ,-self .td3_clip_local ,self .td3_clip_local )
            next_actions_all =torch .clamp (next_actions_all +ta_noise ,-1.0 ,1.0 )

            # 次状態の SoC を復元し、行動を SoC 制約でクリップして実際の電力を計算
            next_actions_kw =next_actions_all *MAX_EV_POWER_KW
            current_socs_s2 =denormalize_soc (ev_block_s2 [...,1 ])
            next_actions_clamped ,next_agent_powers =self ._apply_soc_constraint (
            next_actions_kw ,current_socs_s2 ,ev_padding_mask_s2
            )
            next_powers_norm =torch .clamp (next_agent_powers /max_station_power ,-1.0 ,1.0 )
            next_actions_normalized =torch .clamp (
            next_actions_clamped /MAX_EV_POWER_KW ,-1.0 ,1.0
            )
            # 双子ターゲット Critic の min Q をターゲットとして使用（TD3 の過大評価防止）
            target_qs =[]
            for i in range (self .n ):
                tq1 =self .t_critics [i ](
                s2 [:,i ,:],next_actions_normalized [:,i ,:],
                actual_station_powers =next_powers_norm [:,i ],
                )
                tq2 =self .t_critics2 [i ](
                s2 [:,i ,:],next_actions_normalized [:,i ,:],
                actual_station_powers =next_powers_norm [:,i ],
                )
                target_q =torch .min (tq1 ,tq2 )
                target_qs .append (torch .nan_to_num_ (target_q ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        # ベルマンターゲット: y = r + γ * min(Q_target1, Q_target2) * (1 - done)
        y_targets =[]
        for i in range (self .n ):
            y =r_local [:,i :i +1 ]+self .gamma *target_qs [i ]*(1 -d [:,i :i +1 ])
            y_targets .append (torch .nan_to_num_ (y ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided.")
        # 実際のステーション電力を正規化
        powers_norm =torch .clamp (actual_station_powers /max_station_power ,-1.0 ,1.0 )

        # 現在の双子 Critic の Q 値を計算
        q_vals ,q_vals2 =[],[]
        for i in range (self .n ):
            q_val =self .critics [i ](
            s [:,i ,:],a_actual [:,i ,:],actual_station_powers =powers_norm [:,i ]
            )
            q_val2 =self .critics2 [i ](
            s [:,i ,:],a_actual [:,i ,:],actual_station_powers =powers_norm [:,i ]
            )
            q_vals .append (torch .nan_to_num_ (q_val ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))
            q_vals2 .append (torch .nan_to_num_ (q_val2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        # 各エージェントの Critic を SmoothL1 損失で更新
        self .critic_losses =[]
        for i in range (self .n ):
            loss_c1 =F .smooth_l1_loss (q_vals [i ],y_targets [i ],beta =SMOOTHL1_BETA ,reduction ='mean')
            loss_c2 =F .smooth_l1_loss (q_vals2 [i ],y_targets [i ],beta =SMOOTHL1_BETA ,reduction ='mean')
            loss_c1 =torch .nan_to_num_ (loss_c1 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            loss_c2 =torch .nan_to_num_ (loss_c2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            self .critic_losses .append (loss_c1 .item ()if torch .isfinite (loss_c1 )else 0.0 )

            if self .update_step %self .local_update_interval ==0 :
                if torch .isfinite (loss_c1 ):
                    self .opt_c [i ].zero_grad ()
                    loss_c1 .backward (retain_graph =True )
                    # バイアス勾配を個別クリップ後、全体勾配ノルムクリップ
                    self .clip_bias_gradients (self .critics [i ],max_norm =BIAS_GRAD_CLIP_MAX )
                    gn1 =torch .nn .utils .clip_grad_norm_ (
                    self .critics [i ].parameters (),max_norm =GRAD_CLIP_MAX
                    )
                    gn1f =float (gn1 )
                    if gn1f >5.0 :
                        local_critic_clip_count +=1
                    self .critic_norms_before_clip [i ]=gn1f
                    self .critic_norms [i ]=min (gn1f ,GRAD_CLIP_MAX )
                    self .opt_c [i ].step ()
                if torch .isfinite (loss_c2 ):
                    self .opt_c2 [i ].zero_grad ()
                    loss_c2 .backward ()
                    self .clip_bias_gradients (self .critics2 [i ],max_norm =BIAS_GRAD_CLIP_MAX )
                    _ =torch .nn .utils .clip_grad_norm_ (
                    self .critics2 [i ].parameters (),max_norm =GRAD_CLIP_MAX
                    )
                    self .opt_c2 [i ].step ()

        return local_critic_clip_count


    def _update_actors (self ,ctx ):
        """
        各エージェントの Actor をローカル/グローバル Q 値の加重混合勾配で更新する。

        勾配ブレンド:
          - ローカル Q（LocalEvMLPCritic）とグローバル Q（GlobalMLPCritic）の
            勾配をそれぞれ個別に計算し、w_eff で加重混合する。
          - g_mix = (1 - w_eff) * local_grad + w_eff * global_grad

        STE の適用:
          - _apply_soc_constraint を use_ste=True で呼び出すことで、
            SoC クリップを通じて勾配を Actor まで伝播させる。

        ログ:
          - 各エージェントのローカル/グローバル勾配ノルム・比率・コサイン類似度を記録する。

        Returns:
            actor_clip_count: 勾配クリップが発生した Actor の累計数
        """
        s =ctx ['s']
        ev_padding_mask =ctx ['ev_padding_mask']
        key_padding_mask =ctx ['key_padding_mask']
        batch_size ,max_evs =ctx ['batch_size'],ctx ['max_evs']
        skip_local ,skip_global =ctx ['skip_local'],ctx ['skip_global']
        max_station_power =ctx ['max_station_power']
        w_eff =ctx ['w_eff']
        actor_clip_count =0

        # 全エージェントの Actor を同時に前向き計算し、行動をスタック
        current_actions =[self .actors [i ](s [:,i ,:])for i in range (self .n )]
        cur_a_all_new =torch .stack (current_actions ,dim =1 )
        # 正規化行動（[-1, 1]）を kW 単位に変換
        actions_all =cur_a_all_new *MAX_EV_POWER_KW

        # EV 特徴量ブロックから SoC を復元（SoC 制約クリップに必要）
        ev_feats =s [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,max_evs ,EV_FEAT_DIM )
        current_socs_all =denormalize_soc (ev_feats [...,1 ])

        # EV が存在しないスロットの行動をゼロマスク
        if ev_padding_mask is not None :
            actions_all =actions_all .masked_fill (ev_padding_mask ,0.0 )

        # STE 付きで SoC 制約を適用（勾配を Actor まで通すため use_ste=True）
        clamped_actions_all ,recomputed_actual_station_powers =self ._apply_soc_constraint (
        actions_all ,current_socs_all ,ev_padding_mask ,use_ste =True
        )
        clamped_actions_all_critic =clamped_actions_all

        # グローバル Critic 用の観測を事前に計算（skip_global=False の場合のみ）
        s_global_actor =None
        recomputed_actual_station_powers_normalized =None
        if not skip_global :
            s_global_actor =self ._convert_to_global_critic_obs (
            s ,recomputed_actual_station_powers )
            recomputed_actual_station_powers_normalized =torch .clamp (
            recomputed_actual_station_powers /max_station_power ,-1.0 ,1.0
            )

        # ログ変数の初期化
        self .actor_losses =[0.0 ]*self .n
        self .actor_clip_counts =[0 ]*self .n
        self .actor_source_local_norms_before_clip =[0.0 ]*self .n
        self .actor_source_global_norms_before_clip =[0.0 ]*self .n
        self .actor_source_global_ratio =[0.0 ]*self .n
        self .actor_source_cos =[0.0 ]*self .n
        self .actor_source_cos_valid =[0 ]*self .n

        for i in range (self .n ):
            params =list (self .actors [i ].parameters ())

            agent_actual_power =recomputed_actual_station_powers [:,i ]
            agent_actual_power_normalized =torch .clamp (
            agent_actual_power /max_station_power ,-1.0 ,1.0
            )

            # ローカル Q 値の計算（skip_local=True の場合はゼロ）
            if skip_local :
                q_local =torch .zeros ((batch_size ,1 ),device =device )
            else :
                s_flat_i =s [:,i ,:]
                agent_a =torch .clamp (
                clamped_actions_all_critic [:,i ,:]/MAX_EV_POWER_KW ,-1.0 ,1.0 )
                # 双子ローカル Critic の最小 Q 値を使用（TD3 の過大評価防止）
                ql1 =self .critics [i ](
                s_flat_i ,agent_a ,actual_station_powers =agent_actual_power_normalized )
                ql2 =self .critics2 [i ](
                s_flat_i ,agent_a ,actual_station_powers =agent_actual_power_normalized )
                q_local =torch .min (ql1 ,ql2 )

            if isinstance (q_local ,(tuple ,list )):
                q_local =q_local [0 ]
            q_local =torch .nan_to_num_ (q_local ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

            # エージェント i の行動のみ勾配グラフに残し、他エージェントは detach して固定
            a_all_kw =clamped_actions_all .detach ().clone ()
            a_all_kw [:,i ,:]=clamped_actions_all [:,i ,:]
            try :
                a_all_kw_masked =a_all_kw .masked_fill (ev_padding_mask ,0.0 )
            except Exception :
                a_all_kw_masked =a_all_kw
            a_all_kw_normalized =torch .clamp (a_all_kw_masked /MAX_EV_POWER_KW ,-1.0 ,1.0 )

            # グローバル Q 値の計算（skip_global=True の場合はゼロ）
            if skip_global :
                q_global =torch .zeros ((batch_size ,1 ),device =device )
            else :
                # 双子グローバル Critic の最小 Q 値を使用
                q1 ,_ =self .global_critic1 (
                s_global_actor ,a_all_kw_normalized ,key_padding_mask ,
                actual_station_powers =recomputed_actual_station_powers_normalized ,
                )
                q2 ,_ =self .global_critic2 (
                s_global_actor ,a_all_kw_normalized ,key_padding_mask ,
                actual_station_powers =recomputed_actual_station_powers_normalized ,
                )
                q_global =torch .min (q1 ,q2 )

            if isinstance (q_global ,(tuple ,list )):
                q_global =q_global [0 ]
            q_global =torch .nan_to_num_ (q_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

            q_l_mean =q_local .mean ()
            q_g_mean =q_global .mean ()

            # エピソード中の Q 値をロギング
            try :
                self ._ep_q_raw_local [i ].append (q_l_mean .detach ())
                if i ==0 :
                    self ._ep_q_raw_global .append (q_g_mean .detach ())
                self ._ep_q_raw_combined [i ].append (
                (1.0 -w_eff )*q_l_mean .detach ()+w_eff *q_g_mean .detach ()
                )
            except Exception :
                pass

            self .opt_a [i ].zero_grad ()
            local_grads =[None ]*len (params )
            global_grads =[None ]*len (params )

            # ローカル Q に対する勾配（Actor を Q 最大化する方向 → -Q を最小化）
            if not skip_local :
                local_grads =list (torch .autograd .grad (
                -q_l_mean ,params ,retain_graph =True ,allow_unused =True
                ))
            # グローバル Q に対する勾配
            if not skip_global :
                global_grads =list (torch .autograd .grad (
                -q_g_mean ,params ,retain_graph =True ,allow_unused =True
                ))

            # ローカル/グローバル勾配のノルムとコサイン類似度を計算（診断ログ用）
            sq_l =torch .zeros ((),device =device )
            sq_g =torch .zeros ((),device =device )
            dot_lg =torch .zeros ((),device =device )
            has_l ,has_g =False ,False
            for g_l ,g_g in zip (local_grads ,global_grads ):
                if g_l is not None :
                    sq_l =sq_l +(g_l .detach ()*g_l .detach ()).sum ()
                    has_l =True
                if g_g is not None :
                    sq_g =sq_g +(g_g .detach ()*g_g .detach ()).sum ()
                    has_g =True
                if g_l is not None and g_g is not None :
                    dot_lg =dot_lg +(g_l .detach ()*g_g .detach ()).sum ()

            norm_l =float (torch .sqrt (torch .clamp (sq_l ,min =0.0 )).item ())if has_l else 0.0
            norm_g =float (torch .sqrt (torch .clamp (sq_g ,min =0.0 )).item ())if has_g else 0.0
            # ratio_g: グローバル勾配の比率（グローバルの強さ / 合計の強さ）
            ratio_g =norm_g /max (norm_l +norm_g ,1e-12 )

            cos_lg =0.0
            cos_valid =0
            if has_l and has_g and norm_l >1e-12 and norm_g >1e-12 :
                # コサイン類似度: ローカルとグローバル勾配の方向の一致度
                cos_tensor =dot_lg /(
                torch .sqrt (torch .clamp (sq_l ,min =1e-24 ))*
                torch .sqrt (torch .clamp (sq_g ,min =1e-24 ))
                )
                cos_lg =float (torch .clamp (cos_tensor ,-1.0 ,1.0 ).item ())
                cos_valid =1

            self .actor_source_local_norms_before_clip [i ]=norm_l
            self .actor_source_global_norms_before_clip [i ]=norm_g
            self .actor_source_global_ratio [i ]=ratio_g
            self .actor_source_cos [i ]=cos_lg
            self .actor_source_cos_valid [i ]=cos_valid

            # 各パラメータの混合勾配を計算して設定する
            # g_mix = (1 - w_eff) * local_grad + w_eff * global_grad
            for idx ,p in enumerate (params ):
                g_l =local_grads [idx ]
                g_g =global_grads [idx ]
                if g_l is not None and g_g is not None :
                    g_mix =(1.0 -w_eff )*g_l +w_eff *g_g
                elif g_l is not None :
                    g_mix =g_l
                elif g_g is not None :
                    g_mix =g_g
                else :
                    g_mix =None
                p .grad =g_mix .clone ()if g_mix is not None else None

            # NaN/Inf 勾配が含まれる場合は更新をスキップ
            has_nan_inf =any (
            p .grad is not None and not torch .isfinite (p .grad ).all ()
            for p in params
            )
            if has_nan_inf :
                self .actor_norms [i ]=0.0
                continue

            # バイアス勾配を個別クリップ後、全体勾配ノルムクリップ
            self .clip_bias_gradients (self .actors [i ],max_norm =BIAS_GRAD_CLIP_MAX )
            grad_norm_before =torch .nn .utils .clip_grad_norm_ (
            self .actors [i ].parameters (),max_norm =GRAD_CLIP_MAX
            ).item ()
            grad_norm_after =min (grad_norm_before ,GRAD_CLIP_MAX )

            if grad_norm_before >5.0 :
                actor_clip_count +=1
                if i <len (self .actor_clip_counts ):
                    self .actor_clip_counts [i ]=1

            if math .isfinite (grad_norm_after ):
                self .actor_norms [i ]=grad_norm_after
                self .actor_norms_before_clip [i ]=grad_norm_before
                self .opt_a [i ].step ()
            else :
                self .actor_norms [i ]=0.0

            # Actor の損失 = -(ローカル/グローバル Q 値の加重平均)
            self .actor_losses [i ]=-(
            (1.0 -w_eff )*q_l_mean .detach ()+w_eff *q_g_mean .detach ()
            ).item ()

        return actor_clip_count


    def _polyak_update_targets (self ):
        """
        Polyak 平均（ソフトターゲット更新）でターゲットネットワークを更新する。

        ローカル系（Actor・ローカル Critic）: τ = TAU
          target_param = (1 - τ) * target_param + τ * param
        グローバル Critic: τ_global = TAU_GLOBAL（別管理）

        torch._foreach_lerp_ を使って全パラメータを一括で高効率に更新する。
        """
        # ローカル系: 全エージェントの Actor・Critic1・Critic2 を一括 Polyak 更新
        src_local ,tgt_local =[],[]
        for i in range (self .n ):
            for p ,tp in zip (self .actors [i ].parameters (),self .t_actors [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
            for p ,tp in zip (self .critics [i ].parameters (),self .t_critics [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
            for p ,tp in zip (self .critics2 [i ].parameters (),self .t_critics2 [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
        # lerp_: tgt = tgt + τ * (src - tgt) = (1 - τ) * tgt + τ * src
        torch ._foreach_lerp_ (tgt_local ,src_local ,self .tau )

        # グローバル Critic: τ_global で独立して Polyak 更新
        src_global ,tgt_global =[],[]
        for p ,tp in zip (self .global_critic1 .parameters (),self .t_global_critic1 .parameters ()):
            src_global .append (p .data );tgt_global .append (tp .data )
        for p ,tp in zip (self .global_critic2 .parameters (),self .t_global_critic2 .parameters ()):
            src_global .append (p .data );tgt_global .append (tp .data )
        torch ._foreach_lerp_ (tgt_global ,src_global ,self .tau_global )


    def _aggregate_update_logs (self ,actor_update_due ,local_critic_clip_count ,actor_clip_count ):
        # 各更新サブメソッドで収集した勾配ノルム・損失・クリップ回数を集約してログ変数に保存する
        if self .critic_losses :
            self .last_critic_loss =sum (self .critic_losses )/len (self .critic_losses )
        if self .actor_losses :
            self .last_actor_loss =sum (self .actor_losses )/len (self .actor_losses )

        # ローカル Critic 勾配ノルムの平均
        if self .critic_norms :
            avg_cn =sum (self .critic_norms )/len (self .critic_norms )
            self .last_local_critic_grad_norm =avg_cn if math .isfinite (avg_cn )else 0.0
        else :
            self .last_local_critic_grad_norm =0.0
        self .last_local_critic_clip_count =local_critic_clip_count

        # Actor 勾配ノルムの平均
        if self .actor_norms :
            avg_an =sum (self .actor_norms )/len (self .actor_norms )
            self .last_actor_grad_norm =avg_an if math .isfinite (avg_an )else 0.0
        else :
            self .last_actor_grad_norm =0.0
        self .last_actor_clip_count =actor_clip_count

        # Actor 勾配のローカル/グローバル成分ノルムの平均
        src_local =self .actor_source_local_norms_before_clip
        self .last_actor_source_local_grad_norm_before_clip =(
        float (sum (src_local )/len (src_local ))if src_local else 0.0
        )
        src_global_norms =self .actor_source_global_norms_before_clip
        self .last_actor_source_global_grad_norm_before_clip =(
        float (sum (src_global_norms )/len (src_global_norms ))if src_global_norms else 0.0
        )
        # グローバル勾配の相対比率の平均
        self .last_actor_source_global_ratio =(
        float (sum (self .actor_source_global_ratio )/len (self .actor_source_global_ratio ))
        if self .actor_source_global_ratio else 0.0
        )

        # コサイン類似度の計算（有効なエージェントのみ）
        cos_valid_count =int (sum (self .actor_source_cos_valid ))if self .actor_source_cos_valid else 0
        if cos_valid_count >0 :
            cos_sum =sum (
            float (v )for v ,flag in zip (self .actor_source_cos ,self .actor_source_cos_valid )
            if flag
            )
            self .last_actor_source_cos =cos_sum /float (cos_valid_count )
        else :
            self .last_actor_source_cos =0.0
        # コサイン類似度が有効だったエージェントの割合
        self .last_actor_source_cos_valid_fraction =(
        float (cos_valid_count )/float (len (self .actor_source_cos_valid ))
        if self .actor_source_cos_valid else 0.0
        )

        # 直近更新ステップのエージェントごとのローカル/グローバル Q 値を保存
        self .last_local_q_values_per_agent =[
        float (self ._ep_q_raw_local [i ][-1 ].item ())if self ._ep_q_raw_local [i ]else 0.0
        for i in range (self .n )
        ]
        self .last_global_q_value =(
        float (self ._ep_q_raw_global [-1 ].item ())if self ._ep_q_raw_global else 0.0
        )


    def _update_global_critic (self ,ctx ):
        """
        グローバル QMIX Critic（双子構成）を TD3 スタイルで更新する。

        ターゲット計算:
          1. ターゲット Actor で次状態行動を計算し、クリップ付きグローバルノイズを加える
          2. SoC 制約を適用し、グローバル観測を構築
          3. 双子ターゲット Critic の最小 Q をベルマンターゲットに使用
          4. 現在の双子 Critic の Q 値との SmoothL1 損失を逆伝播して更新

        skip_global=True（Q_MIX_GLOBAL_WEIGHT==0.0）の場合は何もしない。
        """
        # ローカル Q のみを使う設定の場合はグローバル Critic 更新をスキップ
        if ctx ['skip_global']:
            return

        s ,s2 =ctx ['s'],ctx ['s2']
        a_actual =ctx ['a_actual']
        r_global ,d =ctx ['r_global'],ctx ['d']
        actual_station_powers =ctx ['actual_station_powers']
        ev_padding_mask =ctx ['ev_padding_mask']
        ev_padding_mask_s2 =ctx ['ev_padding_mask_s2']
        key_padding_mask =ctx ['key_padding_mask']
        key_padding_mask_s2 =ctx ['key_padding_mask_s2']
        max_station_power =ctx ['max_station_power']
        ev_block_s2 =ctx ['ev_block_s2']

        with torch .no_grad ():
            # ターゲット Actor で次状態行動を計算（全エージェント一括）
            next_a_all =torch .stack (
            [self .t_actors [i ](s2 [:,i ,:])for i in range (self .n )],dim =1
            )
            # グローバル用 TD3 ターゲットポリシースムージング（ローカルとは異なる σ, clip を使用）
            g_noise =torch .randn_like (next_a_all )*self .td3_sigma
            g_noise =torch .clamp (g_noise ,-self .td3_clip ,self .td3_clip )
            next_a_all =torch .clamp (next_a_all +g_noise ,-1.0 ,1.0 )

            # SoC 制約を適用して次状態のクリップ行動とステーション電力を計算
            next_a_kw =next_a_all *MAX_EV_POWER_KW
            current_socs_s2_g =denormalize_soc (ev_block_s2 [...,1 ])
            clamped_actions_next ,next_station_powers =self ._apply_soc_constraint (
            next_a_kw ,current_socs_s2_g ,ev_padding_mask_s2
            )
            # 次状態のグローバル観測を構築
            s2_global =self ._convert_to_global_critic_obs (s2 ,next_station_powers )

            try :
                next_a_kw_masked =clamped_actions_next .masked_fill (ev_padding_mask_s2 ,0.0 )
            except Exception :
                next_a_kw_masked =clamped_actions_next
            next_a_kw_normalized =torch .clamp (next_a_kw_masked /MAX_EV_POWER_KW ,-1.0 ,1.0 )
            next_station_powers_normalized =torch .clamp (
            next_station_powers /max_station_power ,-1.0 ,1.0
            )

            # 双子ターゲット Critic の最小 Q 値をターゲットとして使用
            tq1_s ,_ =self .t_global_critic1 (
            s2_global ,next_a_kw_normalized ,key_padding_mask_s2 ,
            actual_station_powers =next_station_powers_normalized ,
            )
            tq2_s ,_ =self .t_global_critic2 (
            s2_global ,next_a_kw_normalized ,key_padding_mask_s2 ,
            actual_station_powers =next_station_powers_normalized ,
            )
            target_q_global =torch .min (tq1_s ,tq2_s )

            # 全エージェントの done フラグの最大値をグローバルな終了判定に使用
            done_mask_global =torch .max (d ,dim =1 )[0 ].unsqueeze (1 )
            y_global =r_global +self .gamma *target_q_global *(1 -done_mask_global )
            y_global =torch .nan_to_num_ (y_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        # 現在状態のグローバル観測を構築して双子 Critic の Q 値を計算
        s_global =self ._convert_to_global_critic_obs (s ,actual_station_powers )
        try :
            a_actual_global =a_actual .masked_fill (ev_padding_mask ,0.0 )
        except Exception :
            a_actual_global =a_actual
        actual_station_powers_normalized =torch .clamp (
        actual_station_powers /max_station_power ,-1.0 ,1.0
        )

        q1_s ,_ =self .global_critic1 (
        s_global ,a_actual_global ,key_padding_mask ,
        actual_station_powers =actual_station_powers_normalized ,
        )
        q2_s ,_ =self .global_critic2 (
        s_global ,a_actual_global ,key_padding_mask ,
        actual_station_powers =actual_station_powers_normalized ,
        )
        q1_s =torch .nan_to_num_ (q1_s ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        q2_s =torch .nan_to_num_ (q2_s ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        # SmoothL1 損失で双子 Critic を同時に更新（合計損失で一括逆伝播）
        loss_g1 =self .loss_fn (q1_s ,y_global )
        loss_g2 =self .loss_fn (q2_s ,y_global )
        loss_g1 =torch .nan_to_num_ (loss_g1 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        loss_g2 =torch .nan_to_num_ (loss_g2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        loss_g_total =loss_g1 +loss_g2
        if not torch .isfinite (loss_g_total ):
            return

        self .opt_global_c1 .zero_grad ()
        self .opt_global_c2 .zero_grad ()
        loss_g_total .backward ()

        # NaN/Inf 勾配チェック後にグローバル Critic1 を更新
        has_nan_inf =any (
        p .grad is not None and not torch .isfinite (p .grad ).all ()
        for p in self .global_critic1 .parameters ()
        )
        if not has_nan_inf :
            self .clip_bias_gradients (self .global_critic1 ,max_norm =BIAS_GRAD_CLIP_MAX )
            gn_b =torch .nn .utils .clip_grad_norm_ (
            self .global_critic1 .parameters (),max_norm =GRAD_CLIP_MAX_GLOBAL
            ).item ()
            gn_a =min (gn_b ,GRAD_CLIP_MAX_GLOBAL )
            self .last_global_critic_clip_count =1 if gn_b >GRAD_CLIP_MAX_GLOBAL else 0
            self .last_global_critic_grad_norm =gn_a if math .isfinite (gn_a )else 0.0
            self .last_global_critic_grad_norm_before_clip =gn_b if math .isfinite (gn_b )else 0.0
            self .last_global_critic_loss =loss_g1 .item ()if torch .isfinite (loss_g1 )else 0.0
            if math .isfinite (gn_a ):
                self .opt_global_c1 .step ()

        # NaN/Inf 勾配チェック後にグローバル Critic2 を更新
        has_nan_inf2 =any (
        p .grad is not None and not torch .isfinite (p .grad ).all ()
        for p in self .global_critic2 .parameters ()
        )
        if not has_nan_inf2 :
            self .clip_bias_gradients (self .global_critic2 ,max_norm =BIAS_GRAD_CLIP_MAX )
            gn_b2 =torch .nn .utils .clip_grad_norm_ (
            self .global_critic2 .parameters (),max_norm =GRAD_CLIP_MAX_GLOBAL
            ).item ()
            gn_a2 =min (gn_b2 ,GRAD_CLIP_MAX_GLOBAL )
            if math .isfinite (gn_a2 ):
                self .opt_global_c2 .step ()


    def update (self ):
        """
        1 ステップ分の学習更新を実行するメインエントリポイント。

        更新フロー:
          1. _update_local_critics: 各エージェントのローカル Critic を TD3 スタイルで更新
          2. _update_actors（policy_delay ステップごと）: Actor をローカル/グローバル Q 混合勾配で更新
          3. _polyak_update_targets（Actor 更新と同タイミング）: ターゲットネットワークをソフト更新
          4. _aggregate_update_logs: ログ変数を集約
          5. _update_global_critic: グローバル QMIX Critic を更新

        テストモードまたはウォームアップ中（バッファ蓄積量 < WARMUP_STEPS）は何もしない。
        """
        # テストモード時は更新しない
        if hasattr (self ,'test_mode')and self .test_mode :
            self ._zero_update_logs ()
            return

        # ウォームアップ: バッファに十分な経験が蓄積されるまで更新しない
        if self .buf .size <WARMUP_STEPS :
            self ._zero_update_logs ()
            return

        self .update_step +=1
        # policy_delay ステップに 1 回だけ Actor と Polyak 更新を行う（TD3 遅延ポリシー更新）
        actor_update_due =(self .update_step %self .policy_delay ==0 )

        # リプレイバッファからランダムにミニバッチをサンプリング
        s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_power_kw =self .buf .sample (self .batch )
        # サンプルから各サブメソッドで使う全テンソルをコンテキスト辞書として生成
        ctx =self ._build_update_ctx (s ,s2 ,a ,r_local ,r_global ,d ,
        actual_station_powers ,actual_ev_power_kw )

        # ステップ 1: ローカル Critic 更新
        local_critic_clip_count =self ._update_local_critics (ctx )

        # ステップ 2 & 3: Actor 更新と Polyak ソフト更新（policy_delay ごと）
        self .actor_losses =[0.0 ]*self .n
        actor_clip_count =0
        if actor_update_due :
            actor_clip_count =self ._update_actors (ctx )
            self ._polyak_update_targets ()

        # ステップ 4: ログ集約
        self ._aggregate_update_logs (actor_update_due ,local_critic_clip_count ,actor_clip_count )
        # ステップ 5: グローバル Critic 更新
        self ._update_global_critic (ctx )

    def episode_start (self ):
        # エピソード開始時に探索パラメータ（ε・OUノイズスケール）を線形スケジュールで更新する
        if not getattr (self ,'test_mode',False ):
            self .current_episode +=1

        # エピソード内の Q 値ログをリセット
        self .episode_global_q_values =[]
        self .episode_local_q_values =[]
        self ._ep_q_raw_global =[]
        self ._ep_q_raw_local =[[]for _ in range (self .n )]
        self ._ep_q_raw_combined =[[]for _ in range (self .n )]

        if getattr (self ,'test_mode',False ):
            # テストモード: 探索ノイズを完全に無効化
            self .epsilon =0.0
            if hasattr (self ,'ou_noise_scale'):
                self .ou_noise_scale =0.0
        else :
            ep_in_phase =self .current_episode
            # ε-greedy の線形減衰: EPSILON_START_EPISODE → EPSILON_END_EPISODE の区間で減衰
            self .epsilon =linear_epsilon_decay (
            ep_in_phase ,
            self .epsilon_start_episode ,self .epsilon_end_episode ,
            self .epsilon_initial ,self .epsilon_final ,
            )

            # OUノイズスケールの線形減衰
            s0n ,s1n =self .ou_noise_start_episode ,self .ou_noise_end_episode
            n0 ,n1 =self .ou_noise_scale_initial ,self .ou_noise_scale_final
            if ep_in_phase <s0n :
                # OU ノイズ開始エピソード前はスケールを 0 に固定
                self .ou_noise_scale =0.0
            elif ep_in_phase >=s1n :
                # OU ノイズ終了エピソード以降は最終スケールに固定
                self .ou_noise_scale =n1
            else :
                # 開始〜終了エピソード区間で線形補間
                rn =(ep_in_phase -s0n )/max (1 ,(s1n -s0n ))
                self .ou_noise_scale =n0 +(n1 -n0 )*rn
            # スケールが最終値を下回らないように保証
            self .ou_noise_scale =max (self .ou_noise_scale_final ,self .ou_noise_scale )

        if hasattr (self ,'ou_noise'):
            try :
                # OUノイズの内部状態をリセット（エピソードごとに相関をリセット）
                self .ou_noise .reset ()
            except Exception :
                pass

    def episode_end (self ):
        # テストモード時はエピソード終了処理を行わない
        if getattr (self ,'test_mode',False ):
            return

    def set_test_mode (self ,mode :bool ):
        # テストモード（評価用）と学習モードを切り替え、全ネットワークの eval/train を設定する
        self .test_mode =mode
        self .training =not mode

        if mode :
            # テストモード: 全ネットワークを評価モードに設定（BatchNorm・Dropout の挙動が変わる）
            for actor in self .actors :
                actor .eval ()
            for critic in self .critics :
                critic .eval ()
            for critic2 in self .critics2 :
                critic2 .eval ()
            for t_actor in self .t_actors :
                t_actor .eval ()
            for t_critic in self .t_critics :
                t_critic .eval ()
            for t_critic2 in self .t_critics2 :
                t_critic2 .eval ()
            self .global_critic1 .eval ()
            self .global_critic2 .eval ()
            self .t_global_critic1 .eval ()
            self .t_global_critic2 .eval ()
        else :
            # 学習モード: 全ネットワークを訓練モードに戻す
            for actor in self .actors :
                actor .train ()
            for critic in self .critics :
                critic .train ()
            for critic2 in self .critics2 :
                critic2 .train ()
            for t_actor in self .t_actors :
                t_actor .train ()
            for t_critic in self .t_critics :
                t_critic .train ()
            for t_critic2 in self .t_critics2 :
                t_critic2 .train ()
            self .global_critic1 .train ()
            self .global_critic2 .train ()
            self .t_global_critic1 .train ()
            self .t_global_critic2 .train ()

    def cache_experience (self ,s ,s2 ,a ,r_local ,r_global ,d ,
    actual_station_powers =None ,actual_ev_power_kw =None ):
        # テストモード時は経験を保存しない（リプレイバッファへの書き込みをスキップ）
        if hasattr (self ,'test_mode')and self .test_mode :
            return
        self .buf .cache (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_power_kw )

    def save_actors (self ,path ,episode ):
        # 全エージェントの Actor の重みをファイルに保存する
        os .makedirs (path ,exist_ok =True )
        for i in range (self .n ):
            torch .save (self .actors [i ].state_dict (),os .path .join (path ,f"actor_{i}_ep{episode}.pth"))

    def load_actors (self ,path ,episode ,map_location =None ):
        # 全エージェントの Actor の重みをファイルから読み込み、ターゲット Actor にもコピーする
        for i in range (self .n ):
            sd =torch .load (
            os .path .join (path ,f"actor_{i}_ep{episode}.pth"),
            map_location =map_location if map_location is not None else device ,
            )
            self .actors [i ].load_state_dict (sd )
            if i <len (self .t_actors ):
                # ターゲット Actor も同一の重みで初期化（評価時の一貫性のため）
                self .t_actors [i ].load_state_dict (self .actors [i ].state_dict ())
