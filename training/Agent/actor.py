"""
actor.py - DeepSets型置換不変Actorネットワーク

充電ステーション内の全EVスロットの状態を受け取り、各EVへの充放電行動を出力する。
EVスロットの並び順に依存しない置換不変（permutation-invariant）設計を採用。
  - ev_token_encoder: 各EVスロットを独立にエンコードする共有MLP
  - ev_action_head: トークン特徴量 + 集約表現 + tail特徴量から各EVの行動を生成
"""
import torch
import torch .nn as nn
from Config import ACTOR_HIDDEN_SIZE
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_USE_STATION_POWER ,
LOCAL_DEMAND_STEPS ,
LOCAL_USE_STEP ,
)


class Actor (nn .Module ):
    def __init__ (self ,s_dim ,max_evs_per_station ,hid =ACTOR_HIDDEN_SIZE ,station_state_dim =None ,init_gain =0.1 ):
        """
        Actorネットワークを初期化する。

        Parameters
        ----------
        s_dim : int
            入力状態ベクトルの総次元数。
            [EV slots × EV_FEAT_DIM + tail features] のフラットなベクトル。
        max_evs_per_station : int
            1ステーション当たりの最大EVスロット数（固定長パディング含む）。
        hid : int
            隠れ層のサイズ（ACTOR_HIDDEN_SIZEがデフォルト）。
        station_state_dim : int or None
            ステーション状態次元数。Noneの場合はs_dimを使用。
        init_gain : float
            Xavier初期化のゲイン値。小さい値で出力を抑えてトレーニング安定化。
        """
        super (Actor ,self ).__init__ ()
        self .s_dim =s_dim
        self .max_evs =max_evs_per_station
        self .hidden_size =hid
        self .station_state_dim =station_state_dim if station_state_dim is not None else s_dim
        self .init_gain =init_gain

        # EV特徴量の次元数（presence, soc, remaining_time, needed_soc の4特徴）
        self .ev_feat_dim =EV_FEAT_DIM
        # 全EVスロット分のフラット次元数（ev_feat_dim × max_evs）
        self .ev_state_dim =self .ev_feat_dim *self .max_evs
        # tail特徴量の次元数（observation_configの設定に依存）
        # - station_power: ステーション合計電力（任意）
        # - demand_lookahead: 需要予測ステップ数（LOCAL_DEMAND_STEPS個）
        # - current_step: 現在ステップ（任意）
        self .tail_dim =(
        (1 if LOCAL_USE_STATION_POWER else 0 )
        +int (LOCAL_DEMAND_STEPS )
        +(1 if LOCAL_USE_STEP else 0 )
        )

        # EVトークンエンコーダ: 各EVスロットを独立にエンコードする共有MLP
        # 入力: (batch, max_evs, EV_FEAT_DIM) → 出力: (batch, max_evs, set_hid)
        set_hid =max (hid //2 ,32 )
        self .ev_token_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )
        # 行動ヘッド: [token_feat, pooled, tail] の結合から各EVの行動スカラーを生成
        # 入力次元: set_hid (トークン) + set_hid (プール集約) + tail_dim (tail特徴量)
        self .ev_action_head =nn .Sequential (
        nn .Linear (set_hid *2 +self .tail_dim ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,1 ),
        )

        # 全Linearレイヤに Xavier 初期化を適用
        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        """Linearレイヤに Xavier 一様初期化を適用し、バイアスをゼロで初期化する。"""
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,x ):
        """
        Actor の順伝播。

        Parameters
        ----------
        x : Tensor
            形状 (batch, s_dim) または (s_dim,) の状態ベクトル。
            先頭 ev_state_dim 要素が EV スロット特徴量、残りが tail 特徴量。

        Returns
        -------
        Tensor
            形状 (batch, max_evs) の充放電行動テンソル（[-1, 1] の tanh 出力）。
            不在EVスロット（presence <= 0.5）は 0 でマスクされる。
        """
        # 1D入力をバッチ次元に拡張（単一サンプル対応）
        if x .dim ()==1 :
            x =x .unsqueeze (0 )
            squeeze_output =True
        else :
            squeeze_output =False

        batch_size =x .shape [0 ]
        x_normalized =x

        # --- EV スロット特徴量の抽出とトークン化 ---
        # 先頭 ev_state_dim 要素を (batch, max_evs, ev_feat_dim) 形状に reshape
        ev_flat =x_normalized [:,:self .ev_state_dim ]
        ev_tokens =ev_flat .view (batch_size ,self .max_evs ,self .ev_feat_dim )
        # presence（在否）マスク: 特徴量の第0要素が 0.5 超なら在車中（1.0）、それ以外は0
        presence =(ev_tokens [...,0 :1 ]>0.5 ).float ()

        # --- tail 特徴量の抽出（station_power, demand_lookahead, current_step）---
        tail_start =self .ev_state_dim
        tail_end =min (tail_start +self .tail_dim ,x_normalized .size (1 ))
        if tail_end >tail_start :
            tail =x_normalized [:,tail_start :tail_end ]
            # tail_dim に満たない場合はゼロパディング
            if tail .size (1 )<self .tail_dim :
                pad =x_normalized .new_zeros (batch_size ,self .tail_dim -tail .size (1 ))
                tail =torch .cat ([tail ,pad ],dim =1 )
        else :
            # tail 特徴量が存在しない場合は全ゼロテンソルを使用
            tail =x_normalized .new_zeros (batch_size ,self .tail_dim )

        # --- DeepSets トークンエンコード ---
        # 各EVスロットを独立に共有MLPでエンコード: (batch, max_evs, set_hid)
        token_feat =self .ev_token_encoder (ev_tokens )

        # --- マスク付き平均プーリング（置換不変の集約）---
        # 在車中EVのみを平均して集約表現を生成（ゼロ除算防止でclamp(min=1.0)）
        denom =presence .sum (dim =1 ,keepdim =True ).clamp (min =1.0 )
        pooled =(token_feat *presence ).sum (dim =1 ,keepdim =True )/denom

        # --- 行動ヘッドへの入力を構築 ---
        # 集約表現を全スロットに展開: (batch, max_evs, set_hid)
        pooled_expand =pooled .expand (-1 ,self .max_evs ,-1 )
        # tail特徴量を全スロットに展開: (batch, max_evs, tail_dim)
        tail_expand =tail .unsqueeze (1 ).expand (-1 ,self .max_evs ,-1 )
        # [token_feat, pooled, tail] を結合: (batch, max_evs, set_hid*2 + tail_dim)
        token_input =torch .cat ([token_feat ,pooled_expand ,tail_expand ],dim =-1 )

        # --- 行動生成と不在EVスロットのマスキング ---
        # tanh で [-1, 1] に正規化し、不在スロット（presence=0）は 0 でマスク
        raw_actions =torch .tanh (self .ev_action_head (token_input ).squeeze (-1 ))
        raw_actions =raw_actions *presence .squeeze (-1 )

        # バッチ次元を追加した場合は元の形状に戻す
        if squeeze_output :
            raw_actions =raw_actions .squeeze (0 )

        return raw_actions
