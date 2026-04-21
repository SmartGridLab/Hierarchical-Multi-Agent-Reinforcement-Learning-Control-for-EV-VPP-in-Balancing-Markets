"""
critic.py - ローカル・グローバル Q 値ネットワーク

3種類のCriticを提供する。

LocalEvMLPCritic
    1ステーション（1エージェント）のローカル観測と行動から Q 値を推定する。
    DeepSets 型のトークンエンコーダでEVスロットを処理し、
    マスク付き平均プーリングで集約した特徴量から Q 値スカラーを出力する。

GlobalMLPCritic  （QMIX型、USE_JOINT_CRITIC=False 時に使用）
    全エージェントの状態・行動を統合してグローバル Q 値を推定する。
    各ステーションのユーティリティ u_i を計算し、状態依存の重み w で
    単調混合（w > 0 保証）することでグローバル Q 値を生成する。

CentralizedJointCritic  （集中型JointCritic、USE_JOINT_CRITIC=True 時に使用）
    global_obs と joint_action をフラット結合して MLP に通すシンプルな構造。
    QMIX の単調性制約を持たない代わりに表現力が高い。
    GlobalMLPCritic と同一の forward インターフェースを持つ。
"""
import torch
import torch .nn as nn
from Config import (
LOCAL_CRITIC_HIDDEN_SIZE ,
GLOBAL_CRITIC_HIDDEN_SIZE ,
MAX_EV_PER_STATION ,
)
from environment.observation_config import (
LOCAL_DEMAND_STEPS ,
GLOBAL_DEMAND_STEPS ,
)


class LocalEvMLPCritic (nn .Module ):
    def __init__ (self ,ev_feat_dim ,a_dim ,max_evs ,hid =LOCAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        """
        LocalEvMLPCritic を初期化する。

        Parameters
        ----------
        ev_feat_dim : int
            1EVスロット当たりの特徴量次元数（EV_FEAT_DIM）。
        a_dim : int
            ローカル行動次元数（max_evs と同一）。
        max_evs : int
            1ステーション当たりの最大EVスロット数。
        hid : int
            隠れ層サイズ（LOCAL_CRITIC_HIDDEN_SIZEがデフォルト）。
        station_state_dim : int or None
            ステーション状態次元数（Noneの場合は ev_feat_dim*max_evs+4 を使用）。
        init_gain : float
            Xavier 初期化のゲイン値。
        """
        super ().__init__ ()
        self .ev_feat_dim =ev_feat_dim
        self .a_dim =a_dim
        self .max_evs =max_evs
        self .hid =hid
        self .station_state_dim =station_state_dim if station_state_dim is not None else (ev_feat_dim *max_evs +4 )
        self .init_gain =init_gain

        # q_head の追加入力次元: 実ステーション電力(1) + 現在ステップ(1) + 需要予測(LOCAL_DEMAND_STEPS)
        additional_features =1 +1 +int (LOCAL_DEMAND_STEPS )
        set_hid =max (hid //2 ,32 )

        # トークンエンコーダ: [EV特徴量, 行動_i] を各スロットでエンコードする共有MLP
        # 入力: (B, max_evs, ev_feat_dim+1)  →  出力: (B, max_evs, set_hid)
        self .token_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim +1 ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )

        # Q 値ヘッド: [pooled, 実電力, 現在ステップ, 需要予測] → Q 値スカラー
        # 入力: set_hid + additional_features
        self .q_head =nn .Sequential (
        nn .Linear (set_hid +additional_features ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,1 ),
        )

        # 全Linearレイヤに Xavier 初期化を適用
        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        """Linearレイヤに Xavier 一様初期化を適用し、バイアスをゼロで初期化する。"""
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s_flat ,a_flat ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ):
        """
        ローカル Q 値を計算する順伝播。

        Parameters
        ----------
        s_flat : Tensor
            形状 (B, station_state_dim) のローカル観測フラットベクトル。
        a_flat : Tensor
            形状 (B, max_evs) の各EVスロットの行動 [-1, 1]。
        key_padding_mask : Tensor or None
            未使用（インターフェース統一のため保持）。
        return_attn : bool
            未使用（インターフェース統一のため保持）。
        actual_station_powers : Tensor
            形状 (B,) または (B, 1) の実際のステーション合計電力（SoC制約適用後）。
            必須引数。

        Returns
        -------
        Tensor
            形状 (B, 1) の Q 値。NaN/Inf は 0.0 に置換される。
        """
        # 1D入力をバッチ次元に拡張（単一サンプル対応）
        if s_flat .dim ()==1 :
            s_flat =s_flat .unsqueeze (0 )
        if a_flat .dim ()==1 :
            a_flat =a_flat .unsqueeze (0 )

        B =s_flat .size (0 )

        # actual_station_powers は必須（SoC制約後の実電力を使用）
        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for LocalEvMLPCritic")
        # 形状を (B, 1) に統一
        if actual_station_powers .dim ()>1 :
            total_ev_power =actual_station_powers .sum (dim =1 ,keepdim =True )
        else :
            total_ev_power =actual_station_powers .unsqueeze (-1 )

        # --- EV スロット特徴量の抽出 ---
        ev_dim =self .max_evs *self .ev_feat_dim
        ev_flat =s_flat [:,:ev_dim ]
        # (B, max_evs, ev_feat_dim) に reshape
        ev_tokens =ev_flat .view (B ,self .max_evs ,self .ev_feat_dim )
        # presence マスク: 第0特徴量が 0.5 超なら在車中
        presence =(ev_tokens [...,0 :1 ]>0.5 ).float ()

        # 行動テンソルのバリデーションとクランプ
        if a_flat .size (1 )!=self .max_evs :
            raise ValueError (f"Invalid local action shape: {tuple(a_flat.shape)} expected second dim {self.max_evs}")
        # 行動を [-1, 1] にクランプして (B, max_evs, 1) に拡張
        a_tokens =torch .clamp (a_flat ,-1.0 ,1.0 ).unsqueeze (-1 )

        # --- トークンエンコード ---
        # [EV特徴量, 行動_i] を結合して共有MLPでエンコード
        token_input =torch .cat ([ev_tokens ,a_tokens ],dim =-1 )
        token_feat =self .token_encoder (token_input )

        # --- マスク付き平均プーリング ---
        # 在車中EVのみを平均して集約表現を生成（ゼロ除算防止でclamp(min=1.0)）
        denom =presence .sum (dim =1 ,keepdim =False ).clamp (min =1.0 )
        pooled =(token_feat *presence ).sum (dim =1 )/denom

        # --- tail 特徴量の取り出し（末尾から需要予測 + 現在ステップ）---
        L =int (LOCAL_DEMAND_STEPS )
        tail =L +1  # 需要予測ステップ数 + 現在ステップ
        if s_flat .size (1 )>=(self .max_evs *self .ev_feat_dim +tail ):
            # 末尾から現在ステップ（最後の1要素）と需要予測（末尾tail-1要素）を取得
            current_step =s_flat [:,-1 :]
            ag_lookahead =s_flat [:,-tail :-1 ]
        else :
            # 観測が短い場合はゼロで補完
            current_step =s_flat .new_zeros (B ,1 )
            ag_lookahead =s_flat .new_zeros (B ,L )if L >0 else s_flat .new_zeros (B ,0 )

        # 実電力を [-1, 1] にスケールし、pooledと結合してQ値を計算
        scaled_total_ev_power =torch .clamp (total_ev_power ,-1.0 ,1.0 )
        feat =torch .cat ([pooled ,scaled_total_ev_power ,current_step ,ag_lookahead ],dim =1 )
        out =self .q_head (feat )
        # NaN/Inf を 0 に置換して数値安定性を確保
        out =torch .nan_to_num_ (out ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        return out


class GlobalMLPCritic (nn .Module ):
    def __init__ (self ,s_dim ,a_dim ,n_agent ,hid =GLOBAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        """
        GlobalMLPCritic を初期化する。

        Parameters
        ----------
        s_dim : int
            グローバル観測次元数（全エージェントのEV特徴量 + 共通情報）。
        a_dim : int
            1エージェント当たりの行動次元数（max_evs）。
        n_agent : int
            エージェント数（充電ステーション数）。
        hid : int
            隠れ層サイズ（GLOBAL_CRITIC_HIDDEN_SIZEがデフォルト）。
        station_state_dim : int or None
            未使用（インターフェース統一のため保持）。
        init_gain : float
            Xavier 初期化のゲイン値。
        """
        super ().__init__ ()
        self .s_dim =s_dim
        self .a_dim =a_dim
        self .n_agent =n_agent
        self .hid =hid
        # 1ステーション当たりのEV特徴量次元数（4特徴 × MAX_EV_PER_STATION）
        self .ev_features_per_station =4 *MAX_EV_PER_STATION
        self .station_state_dim =self .ev_features_per_station
        self .init_gain =init_gain

        self .station_emb_dim =hid //2
        # ステーション埋め込みMLP: [EV特徴量, 行動] → ステーション埋め込みベクトル
        # 各ステーションの (EV特徴量, 行動) ペアを独立にエンコード
        self .per_station_sa =nn .Sequential (
        nn .Linear (self .ev_features_per_station +a_dim ,self .station_emb_dim ),
        nn .LayerNorm (self .station_emb_dim ),
        nn .LeakyReLU (0.1 ),
        )

        # 各エージェントのユーティリティスカラー u_i を出力するヘッド
        # (B×n_agent, station_emb_dim) → (B×n_agent, 1)
        self .per_agent_head =nn .Sequential (
        nn .Linear (self .station_emb_dim ,self .station_emb_dim //2 ),
        nn .LayerNorm (self .station_emb_dim //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (self .station_emb_dim //2 ,1 ),
        )

        # --- QMIX Mixer ---
        # 共通特徴量: total_ev_power(1) + current_step(1) + demand_lookahead(GLOBAL_DEMAND_STEPS) + station_power_vec(n_agent)
        # mixer_w: 各エージェントの混合重みを状態依存で計算（Softplus で w>0 を保証し単調性を確保）
        # mixer_b: グローバルバイアス（制約なし）
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
        # Softplus: w = softplus(mixer_w(...)) > 0 を保証（QMIX 単調性条件）
        self .softplus =nn .Softplus ()

        # 全Linearレイヤに Xavier 初期化を適用
        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        """Linearレイヤに Xavier 一様初期化を適用し、バイアスをゼロで初期化する。"""
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s ,a ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ,attn_per_head =False ):
        """
        グローバル Q 値を計算する順伝播。

        Parameters
        ----------
        s : Tensor
            形状 (B, global_obs_dim) のグローバル観測ベクトル。
            先頭 n_agent*ev_features_per_station 要素が全ステーションのEV特徴量、
            続いて total_ev_power, current_step, demand_lookahead が格納される。
        a : Tensor
            形状 (B, n_agent, a_dim) の全エージェントの行動テンソル。
        key_padding_mask : Tensor or None
            形状 (B, n_agent) の無効エージェントマスク（Trueで該当エージェントのu_iを0に）。
        return_attn : bool
            Trueの場合 (q_global, u, None) を返す（インターフェース統一用）。
        actual_station_powers : Tensor
            形状 (B, n_agent) の各ステーション実電力（SoC制約適用後）。必須引数。
        attn_per_head : bool
            未使用（インターフェース統一のため保持）。

        Returns
        -------
        Tensor or tuple
            return_attn=False: (q_global, u) のタプル。
            return_attn=True: (q_global, u, None) のタプル。
            q_global の形状は (B, 1)、u の形状は (B, n_agent)。
        """
        # 1D入力をバッチ次元に拡張（単一サンプル対応）
        if s .dim ()==1 :
            s =s .unsqueeze (0 )
        B =s .size (0 )

        # --- グローバル観測の分解 ---
        # 先頭: 全ステーションのEV特徴量フラット (B, n_agent * ev_features_per_station)
        ev_features_total =self .n_agent *self .ev_features_per_station
        ev_features_flat =s [:,:ev_features_total ]

        # EV特徴量の直後: 共通情報 (total_ev_power, current_step, demand_lookahead)
        common_info_start =ev_features_total
        total_ev_power =s [:,common_info_start :common_info_start +1 ]
        current_step =s [:,common_info_start +1 :common_info_start +2 ]
        L =int (GLOBAL_DEMAND_STEPS )
        ag_lookahead =s [:,common_info_start +2 :common_info_start +2 +L ]if L >0 else s .new_zeros (B ,0 )

        # actual_station_powers のバリデーションと形状確認
        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for GlobalMLPCritic")
        # 各ステーション実電力を [-1, 1] にクランプ
        station_power_vec =torch .clamp (actual_station_powers ,-1.0 ,1.0 )
        if station_power_vec .dim ()==1 :
            station_power_vec =station_power_vec .unsqueeze (0 )
        if station_power_vec .dim ()!=2 or station_power_vec .size (1 )!=self .n_agent :
            raise ValueError (
            f"Invalid actual_station_powers shape: {tuple(station_power_vec.shape)} "
            f"expected (B, {self.n_agent})"
            )

        # 行動テンソルのバリデーションとクランプ
        if a is None :
            raise ValueError ("action tensor 'a' must be provided for GlobalMLPCritic")
        actions_vec =torch .clamp (a ,-1.0 ,1.0 )
        if actions_vec .dim ()!=3 or actions_vec .size (1 )!=self .n_agent or actions_vec .size (2 )!=self .a_dim :
            raise ValueError (f"Invalid action shape: {tuple(actions_vec.shape)} expected (B, n, {self.a_dim})")

        # --- 各ステーションの (EV特徴量, 行動) をエンコード ---
        # (B, n_agent, ev_features_per_station) に reshape
        ev_per_station =ev_features_flat .reshape (B ,self .n_agent ,self .ev_features_per_station )
        # [EV特徴量, 行動] を結合: (B, n_agent, ev_features_per_station + a_dim)
        station_input =torch .cat ([ev_per_station ,actions_vec ],dim =2 )
        # (B*n_agent, ...) でバッチ処理し (B, n_agent, station_emb_dim) に戻す
        station_feats =self .per_station_sa (
        station_input .reshape (B *self .n_agent ,-1 )
        ).reshape (B ,self .n_agent ,self .station_emb_dim )

        # --- 各エージェントのユーティリティ u_i を計算 ---
        # (B, n_agent, 1) → squeeze して (B, n_agent)
        u_tokens =self .per_agent_head (station_feats )
        u =u_tokens .squeeze (-1 )
        # key_padding_mask が指定された無効エージェントのユーティリティをゼロにマスク
        if key_padding_mask is not None :
            u =u .masked_fill (key_padding_mask ,0.0 )

        # --- QMIX 単調混合 ---
        # 共通特徴量: [total_ev_power, current_step, demand_lookahead, station_power_vec]
        common_features =torch .cat ([total_ev_power ,current_step ,ag_lookahead ,station_power_vec ],dim =1 )

        # 混合重み w = softplus(...) > 0 （単調性保証）
        w =self .mixer_w (common_features )
        w =self .softplus (w )
        # バイアス b（制約なし）
        b =self .mixer_b (common_features )

        # グローバル Q 値: Q_global = mean(w ⊙ u) + b
        q_global =(w *u ).mean (dim =1 ,keepdim =True )+b

        # NaN/Inf を 0 に置換して数値安定性を確保
        if not torch .isfinite (q_global ).all ():
            q_global =torch .nan_to_num (q_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        if not torch .isfinite (u ).all ():
            u =torch .nan_to_num (u ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        # return_attn=True の場合はアテンション互換のタプルで返す
        if return_attn :
            return q_global ,u ,None

        return q_global ,u


class CentralizedJointCritic (nn .Module ):
    """
    集中型 Joint Critic（USE_JOINT_CRITIC=True 時に GlobalMLPCritic の代替として使用）。

    アーキテクチャ:
        global_obs  (B, s_dim)
        joint_action (B, n_agent, a_dim) → flatten → (B, n_agent * a_dim)
        actual_station_powers (B, n_agent)
        → concat → (B, s_dim + n_agent*a_dim + n_agent)
        → Linear → LayerNorm → LeakyReLU
        → Linear → LayerNorm → LeakyReLU
        → Linear → scalar Q  (B, 1)

    QMIX との違い:
        - u_i / mixer_w / mixer_b / softplus を持たない
        - 単調性制約なし（全結合MLPで直接 Q スカラーを出力）
        - GlobalMLPCritic と同一の forward シグネチャを持つため
          maddpg.py 側の呼び出しコードを変更せずに差し替え可能

    Returns
    -------
    (q_global, None) または (q_global, None, None)
        QMIX との互換性のため第2戻り値は常に None。
    """

    def __init__ (self ,s_dim ,a_dim ,n_agent ,hid =GLOBAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        """
        Parameters
        ----------
        s_dim : int
            グローバル観測次元数。
        a_dim : int
            1エージェント当たりの行動次元数（max_evs）。
        n_agent : int
            エージェント数。
        hid : int
            隠れ層サイズ。
        station_state_dim : int or None
            未使用（インターフェース統一のため保持）。
        init_gain : float
            Xavier 初期化のゲイン値。
        """
        super ().__init__ ()
        self .s_dim =s_dim
        self .a_dim =a_dim
        self .n_agent =n_agent
        self .hid =hid
        self .init_gain =init_gain

        # 入力次元: global_obs + joint_action_flat + actual_station_powers
        in_dim =s_dim +n_agent *a_dim +n_agent

        self .mlp =nn .Sequential (
        nn .Linear (in_dim ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,hid ),
        nn .LayerNorm (hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid ,1 ),
        )

        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        """Linearレイヤに Xavier 一様初期化を適用し、バイアスをゼロで初期化する。"""
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s ,a ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ,attn_per_head =False ):
        """
        グローバル Q 値を計算する順伝播。

        Parameters
        ----------
        s : Tensor
            形状 (B, s_dim) のグローバル観測ベクトル。
        a : Tensor
            形状 (B, n_agent, a_dim) の全エージェント行動テンソル。
        key_padding_mask : Tensor or None
            未使用（インターフェース統一のため保持）。
        return_attn : bool
            True の場合 (q, None, None) を返す（インターフェース統一用）。
        actual_station_powers : Tensor
            形状 (B, n_agent) の各ステーション実電力。必須引数。
        attn_per_head : bool
            未使用（インターフェース統一のため保持）。

        Returns
        -------
        tuple
            (q_global, None) または (q_global, None, None)。
            q_global の形状は (B, 1)。
        """
        if s .dim ()==1 :
            s =s .unsqueeze (0 )
        B =s .size (0 )

        # joint_action: (B, n_agent, a_dim) → (B, n_agent * a_dim)
        if a is None :
            raise ValueError ("action tensor 'a' must be provided for CentralizedJointCritic")
        a_flat =torch .clamp (a ,-1.0 ,1.0 ).reshape (B ,-1 )

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for CentralizedJointCritic")
        sp =torch .clamp (actual_station_powers ,-1.0 ,1.0 )
        if sp .dim ()==1 :
            sp =sp .unsqueeze (0 )

        # [global_obs, joint_action_flat, station_powers] を結合して MLP に入力
        x =torch .cat ([s ,a_flat ,sp ],dim =1 )
        q =self .mlp (x )
        q =torch .nan_to_num_ (q ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        if return_attn :
            return q ,None ,None
        return q ,None
