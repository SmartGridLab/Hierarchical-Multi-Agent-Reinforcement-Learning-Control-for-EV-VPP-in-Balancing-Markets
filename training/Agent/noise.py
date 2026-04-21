"""
noise.py - 探索ノイズと ε-greedy 戦略

連続行動空間での探索に使用するノイズクラスとユーティリティ関数を提供する。

OUNoise
    Ornstein-Uhlenbeck 過程ノイズ。
    時系列相関を持つため、連続行動空間の探索（特に慣性のある系）に適している。
    各エージェント・各 EV スロット独立の OU プロセスを並列管理する。
    更新式: dx = -θ(x - μ)dt + σ√dt * ε  (ε ~ N(0,1), μ=0)

linear_epsilon_decay
    ε を指定エピソード区間で線形減衰させる。

sample_epsilon_random_action
    指定範囲のランダム行動テンソルをサンプリングする。

should_take_random_action
    ε-greedy 判定: epsilon 確率でランダム行動を選択すべきか判定する。
"""
import random
import torch
from Config import DEVICE ,OU_THETA ,OU_SIGMA ,OU_DT ,OU_INIT_X ,OU_CLIP


class OUNoise :
    """
    Ornstein-Uhlenbeck 過程ノイズ（探索ノイズ）。

    全エージェント × 全 EV スロットの OU プロセスを GPU テンソルで並列管理する。
    各スロット独立に状態を保持し、reset() でエピソード開始時に初期値に戻す。

    OU プロセス更新式:
        dx = -θ (x - 0) dt + σ √dt * N(0,1)
        x ← x + dx

    Parameters
    ----------
    n_agents : int
        エージェント数（充電ステーション数）。
    max_evs_per_station : int
        1ステーション当たりの最大EVスロット数。
    theta : float
        平均回帰速度（OU_THETA）。大きいほど速く平均に回帰する。
    sigma : float
        拡散係数（OU_SIGMA）。大きいほどノイズが強い。
    dt : float
        時間刻み（OU_DT）。
    x0 : float
        OU プロセスの初期値（OU_INIT_X）。
    """
    def __init__ (self ,n_agents :int ,max_evs_per_station :int ,
    theta :float =OU_THETA ,sigma :float =OU_SIGMA ,
    dt :float =OU_DT ,x0 :float =OU_INIT_X ):
        self .n_agents =int (n_agents )
        self .max_evs =int (max_evs_per_station )
        self .theta =float (theta )
        self .sigma =float (sigma )
        self .dt =float (dt )
        self .x0 =float (x0 )
        self .device =DEVICE
        # OU プロセスの状態テンソル: (n_agents, max_evs), 初期値は x0 で埋める
        self .state =torch .full ((self .n_agents ,self .max_evs ),float (self .x0 ),
        dtype =torch .float32 ,device =self .device )

    def reset (self ):
        """OU プロセスの状態を初期値 x0 にリセットする（エピソード開始時に呼ぶ）。"""
        self .state .fill_ (float (self .x0 ))

    @torch .no_grad ()
    def sample (self ,active_evs_per_agent ):
        """
        OU ノイズを1ステップ進め、アクティブなEVスロット分のノイズを返す。

        active_evs_per_agent: list[int] or 1D tensor (len=n_agents)
        Returns: Tensor shape (n_agents, max_evs)

        不在スロット（active_evs_per_agent[i] 以降）のノイズは 0 になる。
        OU_CLIP が設定されている場合はノイズをクランプする。
        """
        # active_evs_per_agent を GPU テンソルに変換
        if not torch .is_tensor (active_evs_per_agent ):
            active =torch .as_tensor (active_evs_per_agent ,device =self .device )
        else :
            active =active_evs_per_agent .to (self .device )

        # --- OU プロセスの1ステップ更新 ---
        # dx = -θ * x * dt + σ * √dt * N(0,1)  （μ=0 なので x - μ = x）
        noise =torch .randn_like (self .state )
        dx =self .theta *(0.0 -self .state )*self .dt +self .sigma *(noise *(self .dt **0.5 ))
        self .state =self .state +dx

        # --- アクティブスロットのみノイズを出力（不在スロットは 0）---
        out =torch .zeros_like (self .state )
        for i in range (self .n_agents ):
            k =int (active [i ].item ())if active .dim ()>0 else int (active .item ())
            if k >0 :
                out [i ,:k ]=self .state [i ,:k ]

        # OU_CLIP が設定されている場合はノイズ値をクランプ
        if OU_CLIP is not None and OU_CLIP >0 :
            out =torch .clamp (out ,-float (OU_CLIP ),float (OU_CLIP ))
        return out


def linear_epsilon_decay (current_episode :int ,
start_episode :int ,
end_episode :int ,
epsilon_initial :float ,
epsilon_final :float )->float :
    """
    ε-greedy の探索率 ε を線形減衰させる。

    start_episode から end_episode の間で epsilon_initial から epsilon_final へ
    線形補間する。区間外では端点の値にクランプする。

    Parameters
    ----------
    current_episode : int
        現在のエピソード番号。
    start_episode : int
        線形減衰を開始するエピソード番号（これ以前は epsilon_initial を返す）。
    end_episode : int
        線形減衰を終了するエピソード番号（これ以降は epsilon_final を返す）。
    epsilon_initial : float
        減衰開始時の ε 値（通常 1.0 など探索重視の値）。
    epsilon_final : float
        減衰終了時の ε 値（通常 0.05 など活用重視の値）。

    Returns
    -------
    float
        現在エピソードに対応する ε 値（[min(e0,e1), max(e0,e1)] にクランプ済み）。
    """
    ep =int (current_episode )
    s0 =int (start_episode )
    s1 =int (end_episode )
    e0 =float (epsilon_initial )
    e1 =float (epsilon_final )

    if ep <=s0 :
        # 開始エピソード以前: 初期 ε を返す
        eps =e0
    elif ep >=s1 :
        # 終了エピソード以降: 最終 ε を返す
        eps =e1
    else :
        # 線形補間: r = (ep - s0) / (s1 - s0) で [0, 1] に正規化
        r =(ep -s0 )/max (1 ,(s1 -s0 ))
        eps =e0 +(e1 -e0 )*r

    # 数値誤差対策: [min(e0,e1), max(e0,e1)] の範囲にクランプ
    lo =min (e0 ,e1 )
    hi =max (e0 ,e1 )
    return max (lo ,min (hi ,eps ))


def sample_epsilon_random_action (num_active :int ,
action_range =(-1.0 ,1.0 ),
like_tensor :torch .Tensor =None )->torch .Tensor :
    """
    指定範囲のランダム行動テンソルをサンプリングする（ε-greedy 用）。

    Parameters
    ----------
    num_active : int
        サンプリングする行動の個数（アクティブなEVスロット数）。
    action_range : tuple of float
        行動の範囲 (low, high)。デフォルトは (-1.0, 1.0)。
    like_tensor : Tensor or None
        デバイス・dtype の参照テンソル。Noneの場合は DEVICE と float32 を使用。

    Returns
    -------
    Tensor
        形状 (num_active,) の一様分布ランダム行動テンソル。
    """
    low ,high =float (action_range [0 ]),float (action_range [1 ])
    # like_tensor が指定されていればそのデバイスと dtype を継承
    if like_tensor is not None :
        out =torch .empty (num_active ,device =like_tensor .device ,dtype =like_tensor .dtype )
    else :
        out =torch .empty (num_active ,device =DEVICE ,dtype =torch .float32 )
    # [low, high] の一様分布でテンソルを埋める
    out .uniform_ (low ,high )
    return out


def should_take_random_action (epsilon :float )->bool :
    """
    ε-greedy 判定: epsilon 確率でランダム行動を選択すべきかを返す。

    Parameters
    ----------
    epsilon : float
        ランダム行動を選択する確率 [0.0, 1.0]。

    Returns
    -------
    bool
        True: ランダム行動を選択すべき（確率 epsilon）。
        False: ポリシーに従う行動を選択すべき（確率 1 - epsilon）。
    """
    return random .random ()<float (epsilon )
