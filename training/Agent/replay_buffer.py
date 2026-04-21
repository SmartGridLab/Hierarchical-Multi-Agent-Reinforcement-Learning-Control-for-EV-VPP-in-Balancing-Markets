"""
replay_buffer.py - 経験再生バッファ（Experience Replay Buffer）

オフポリシー強化学習（MADDPG等）で使用する循環型リプレイバッファ。
全テンソルをGPUに保持することで高速なミニバッチサンプリングを実現する。

設計上の特徴:
  - 遅延初期化: 最初の cache() 呼び出し時に入力形状を確定し GPU テンソルを確保する。
    これにより、バッファ生成時にエージェント数・状態次元数が未決定でも動作する。
  - temp_buf: GPU テンソル確保前に受け取ったデータを一時的に CPU リストで保持し、
    初期化完了後にまとめて GPU テンソルへ転送する。
  - ポインタ方式の循環バッファ: ptr % buf_size により古いデータを上書きする。

格納データ（GPU テンソル）:
  s, s2               : (buf_size, n_agents, s_dim)    現在/次状態
  a                   : (buf_size, n_agents, max_evs)  行動
  r_local             : (buf_size, n_agents)           ローカル報酬
  r_global            : (buf_size, 1)                  グローバル報酬
  d                   : (buf_size, n_agents)           エピソード完了フラグ
  actual_station_powers : (buf_size, n_agents)         SoC制約後の実ステーション電力
  actual_ev_soc_changes : (buf_size, n_agents, max_evs) 実際のEV SoC変化量
"""
import torch
import numpy as np

# 利用可能なら GPU を使用し、なければ CPU にフォールバック
device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


class ReplayBuffer :
    def __init__ (self ,cap =int (1e5 )):
        """
        ReplayBuffer を初期化する。

        Parameters
        ----------
        cap : int
            バッファの最大容量（エントリ数）。デフォルトは 100,000。
            バッファが満杯になると古いデータを上書きする（循環バッファ）。
        """
        self .buf_size =cap

        # --- 遅延初期化用の形状パラメータ（最初の cache() で確定） ---
        self .s_dim =None   # 状態次元数
        self .a_dim =None   # 行動次元数（未使用、max_evs で代替）
        self .n_agents =None  # エージェント数
        self .max_evs =None  # 1エージェント当たりの最大EVスロット数

        # --- 循環バッファのポインタとサイズ管理 ---
        self .ptr =0   # 次に書き込む位置（0 ～ buf_size-1 の循環）
        self .size =0  # 現在格納されているエントリ数（最大 buf_size）

        # --- GPU テンソル（遅延初期化、最初の cache() で確保） ---
        self .s =None    # 現在状態
        self .s2 =None   # 次状態
        self .a =None    # 行動

        self .r_local =None   # ローカル報酬（エージェントごと）
        self .r_global =None  # グローバル報酬（需給追従など全体報酬）
        self .d =None         # 完了フラグ（エピソード終了で 1.0）

        self .actual_station_powers =None  # SoC制約適用後の実ステーション電力

        # 推定メモリ使用量（GB）のログ用変数
        self .estimated_memory_gb =0.0

        # GPU テンソル確保前のデータを一時保存する CPU リスト（遅延初期化対応）
        self .temp_buf =[]

    def cache (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers =None ,actual_ev_soc_changes =None ):
        """
        1ステップ分の経験をバッファに追加する。

        GPU テンソルが未初期化の場合（最初の呼び出し時）、入力データの形状から
        テンソルサイズを確定し GPU テンソルを確保する。それ以前のデータは
        temp_buf から一括で転送される。

        Parameters
        ----------
        s : array-like or Tensor
            形状 (n_agents, s_dim) の現在状態。
        s2 : array-like or Tensor
            形状 (n_agents, s_dim) の次状態。
        a : array-like or Tensor
            形状 (n_agents, max_evs) の行動テンソル。
        r_local : array-like or Tensor
            形状 (n_agents,) のローカル報酬。
        r_global : scalar or Tensor
            グローバル報酬（スカラーまたは (1,) テンソル）。
        d : array-like or Tensor
            形状 (n_agents,) の完了フラグ（終了: 1.0、継続: 0.0）。
        actual_station_powers : array-like or Tensor, optional
            形状 (n_agents,) の SoC 制約適用後の実ステーション電力。必須。
        actual_ev_soc_changes : array-like or Tensor, optional
            形状 (n_agents, max_evs) の実際の EV SoC 変化量。
            None の場合は行動テンソルで代替。
        """
        # --- 遅延初期化: 最初の cache() 呼び出し時のみ実行 ---
        if self .s is None :
            self .n_agents =len (s )
            self .s_dim =s [0 ].shape [0 ]if isinstance (s [0 ],np .ndarray )else len (s [0 ])

            # max_evs の確定（行動テンソルの形状から推定）
            if isinstance (a ,np .ndarray ):
                self .max_evs =a .shape [2 ]if a .ndim ==3 else a .shape [1 ]
            else :
                # リスト・tuple・Tensor など各形式に対応したフォールバック処理
                try :
                    if isinstance (a ,tuple )or isinstance (a ,list ):
                        if len (a )>0 and hasattr (a [0 ],'shape')and len (a [0 ].shape )>0 :
                            self .max_evs =a [0 ].shape [1 ]
                        else :
                            self .max_evs =5  # デフォルト値
                    elif isinstance (a ,torch .Tensor ):
                        self .max_evs =a .shape [1 ]if len (a .shape )>1 else 5
                    else :
                        self .max_evs =5  # デフォルト値
                except (IndexError ,AttributeError ):
                    self .max_evs =5  # 形状取得失敗時のデフォルト値

            # --- GPU テンソルの一括確保 ---
            self .s =torch .zeros ((self .buf_size ,self .n_agents ,self .s_dim ),
            dtype =torch .float32 ,device =device )
            self .s2 =torch .zeros ((self .buf_size ,self .n_agents ,self .s_dim ),
            dtype =torch .float32 ,device =device )
            self .a =torch .zeros ((self .buf_size ,self .n_agents ,self .max_evs ),
            dtype =torch .float32 ,device =device )

            self .r_local =torch .zeros ((self .buf_size ,self .n_agents ),
            dtype =torch .float32 ,device =device )

            self .r_global =torch .zeros ((self .buf_size ,1 ),
            dtype =torch .float32 ,device =device )
            self .d =torch .zeros ((self .buf_size ,self .n_agents ),
            dtype =torch .float32 ,device =device )

            self .actual_station_powers =torch .zeros ((self .buf_size ,self .n_agents ),
            dtype =torch .float32 ,device =device )

            # 実際のSoC変化量テンソル（行動と同形状）
            self .actual_ev_soc_changes =torch .zeros ((self .buf_size ,self .n_agents ,self .max_evs ),
            dtype =torch .float32 ,device =device )

            # --- temp_buf に溜まっていたデータを GPU テンソルへ一括転送 ---
            for temp_data in self .temp_buf :
                if len (temp_data )==8 :
                    temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d ,temp_actual_powers ,temp_actual_ev_soc_changes =temp_data
                elif len (temp_data )==7 :
                    temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d ,temp_actual_powers =temp_data
                    temp_actual_ev_soc_changes =None
                else :
                    temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d =temp_data
                    temp_actual_powers =None
                    temp_actual_ev_soc_changes =None
                self ._cache_to_tensor (temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d ,temp_actual_powers ,temp_actual_ev_soc_changes )
            self .temp_buf =[]  # 一時バッファをクリア

        # --- 実際のデータ格納 ---
        if self .s is not None :
            # GPU テンソル確保済みなら直接書き込む
            self ._cache_to_tensor (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes )
        else :
            # 初期化前（通常は起こらないが念のため）は temp_buf に追加
            if actual_ev_soc_changes is not None :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes ))
            elif actual_station_powers is not None :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ))
            else :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ))

    def _cache_to_tensor (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers =None ,actual_ev_soc_changes =None ):
        """
        1エントリ分のデータを GPU テンソルの現在位置（ptr）に書き込む。

        Tensor / ndarray / リストなど各形式に対応し、デバイスが異なる場合は
        GPU へ転送してから格納する。

        Parameters
        ----------
        ※ cache() と同じシグネチャ。actual_station_powers は必須（None 禁止）。
        """
        idx =self .ptr

        # --- 現在状態 s の格納 ---
        if isinstance (s ,torch .Tensor ):
            if s .device ==device :
                self .s [idx ]=s .detach ()
            else :
                self .s [idx ]=s .detach ().to (device )
        else :
            self .s [idx ]=torch .tensor (s ,dtype =torch .float32 ,device =device )

        # --- 次状態 s2 の格納 ---
        if isinstance (s2 ,torch .Tensor ):
            if s2 .device ==device :
                self .s2 [idx ]=s2 .detach ()
            else :
                self .s2 [idx ]=s2 .detach ().to (device )
        else :
            self .s2 [idx ]=torch .tensor (s2 ,dtype =torch .float32 ,device =device )

        # --- 行動 a の格納 ---
        if isinstance (a ,torch .Tensor ):
            if a .device ==device :
                self .a [idx ]=a .detach ()
            else :
                self .a [idx ]=a .detach ().to (device )
        else :
            self .a [idx ]=torch .tensor (a ,dtype =torch .float32 ,device =device )

        # --- ローカル報酬 r_local の格納 ---
        if isinstance (r_local ,torch .Tensor ):
            if r_local .device ==device :
                self .r_local [idx ]=r_local .detach ()
            else :
                self .r_local [idx ]=r_local .detach ().to (device )
        else :
            self .r_local [idx ]=torch .tensor (r_local ,dtype =torch .float32 ,device =device )

        # --- グローバル報酬 r_global の格納（スカラーを (1,) に統一） ---
        if isinstance (r_global ,torch .Tensor ):
            if r_global .device ==device :
                # flatten して先頭1要素のみ格納（スカラー・ベクトル両対応）
                self .r_global [idx ]=r_global .detach ().flatten ()[:1 ]
            else :
                self .r_global [idx ]=r_global .detach ().to (device ).flatten ()[:1 ]
        else :
            # Python スカラーをリスト経由でテンソル化
            self .r_global [idx ]=torch .tensor ([r_global ],dtype =torch .float32 ,device =device )

        # --- 完了フラグ d の格納 ---
        if isinstance (d ,torch .Tensor ):
            if d .device ==device :
                self .d [idx ]=d .detach ()
            else :
                self .d [idx ]=d .detach ().to (device )
        else :
            self .d [idx ]=torch .tensor (d ,dtype =torch .float32 ,device =device )

        # --- 実ステーション電力 actual_station_powers の格納（必須） ---
        if actual_station_powers is not None :
            if isinstance (actual_station_powers ,torch .Tensor ):
                if actual_station_powers .device ==device :
                    self .actual_station_powers [idx ]=actual_station_powers .detach ()
                else :
                    self .actual_station_powers [idx ]=actual_station_powers .detach ().to (device )
            else :
                self .actual_station_powers [idx ]=torch .tensor (actual_station_powers ,dtype =torch .float32 ,device =device )
        else :
            # SoC制約後の実電力は必須であり、環境から必ず提供されなければならない
            raise ValueError ("actual_station_powers must be provided. Environment should provide actual station powers after applying SoC constraints.")

        # --- 実EV SoC変化量 actual_ev_soc_changes の格納 ---
        if actual_ev_soc_changes is not None :
            if isinstance (actual_ev_soc_changes ,torch .Tensor ):
                if actual_ev_soc_changes .device ==device :
                    self .actual_ev_soc_changes [idx ]=actual_ev_soc_changes .detach ()
                else :
                    self .actual_ev_soc_changes [idx ]=actual_ev_soc_changes .detach ().to (device )
            else :
                self .actual_ev_soc_changes [idx ]=torch .tensor (actual_ev_soc_changes ,dtype =torch .float32 ,device =device )
        else :
            # actual_ev_soc_changes が未提供の場合は行動テンソルで代替
            self .actual_ev_soc_changes [idx ]=self .a [idx ].clone ()

        # --- 循環バッファのポインタ更新 ---
        # ptr を 1 進め、buf_size で折り返す（古いデータを上書き）
        self .ptr =(self .ptr +1 )%self .buf_size
        # size は buf_size を上限として増加
        self .size =min (self .size +1 ,self .buf_size )

    def sample (self ,batch ):
        """
        バッファからランダムにミニバッチをサンプリングする。

        GPU テンソルから直接インデックスアクセスするため高速。

        Parameters
        ----------
        batch : int
            サンプリングするエントリ数。

        Returns
        -------
        tuple of Tensor
            (s, s2, a, r_local, r_global, d, actual_station_powers, actual_ev_soc_changes)
            各テンソルはデバイス（GPU）上にある。

        Raises
        ------
        RuntimeError
            GPU テンソルが未初期化（cache() が一度も呼ばれていない）の場合。
        """
        # GPU テンソルが未初期化の場合はエラー
        if self .s is None :
            raise RuntimeError ("ReplayBuffer is not initialized. cache() must be called to initialize GPU buffers before sampling.")

        # size の範囲内でランダムにインデックスを選択（重複あり）
        idxs =torch .randint (0 ,self .size ,(batch ,),device =device )

        # GPU テンソルから直接バッチ取得（高速インデックスアクセス）
        s_batch =self .s [idxs ]
        s2_batch =self .s2 [idxs ]
        a_batch =self .a [idxs ]
        r_local_batch =self .r_local [idxs ]
        r_global_batch =self .r_global [idxs ]
        d_batch =self .d [idxs ]
        actual_station_powers_batch =self .actual_station_powers [idxs ]

        actual_ev_soc_changes_batch =self .actual_ev_soc_changes [idxs ]
        # non_blocking=True でデバイス転送を非同期化（既にGPU上なら即時）
        return tuple (t .to (device ,non_blocking =True )for t in (s_batch ,s2_batch ,a_batch ,r_local_batch ,r_global_batch ,d_batch ,actual_station_powers_batch ,actual_ev_soc_changes_batch ))
