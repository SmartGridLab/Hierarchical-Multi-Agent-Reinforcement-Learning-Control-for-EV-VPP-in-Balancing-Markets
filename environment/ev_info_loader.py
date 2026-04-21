"""
ev_info_loader.py
=================
EV到着・プロファイル情報のローダー。

- load_accurate_ev_info: 実データ由来のEVプロファイル（SoC分布・滞在時間分布）を読み込む。
  - EV_SOC_ARRIVAL_DISTRIBUTION_PATH: 到着時SoC分布CSV（累積分布関数に変換して使用）
  - EV_PROFILE_DATA_PATH: EV識別子・必要SoC・滞在時間CSVのパス
- load_arrival_probabilities: ステーション別の時刻別EV到着確率を読み込む。
  - 1行=1タイムステップの到着確率（[0,1]）として正規化する

AccurateEVInfo: 読み込んだEVプロファイル情報を保持するデータクラス（イミュータブル）。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass (frozen =True )
class AccurateEVInfo :
    """実データ由来のEVプロファイル情報を保持するイミュータブルデータクラス。

    Attributes:
        arrival_probabilities: 各タイムステップのEV到着確率配列（shape: episode_steps）
        soc_values: 到着時SoC分布のサポート値（kWh相当の整数列）
        soc_cdf: soc_values に対応する累積分布関数（CDF）配列
        needed_soc_values: 必要SoCのヒストグラムのサポート値
        needed_soc_pmf: 必要SoCの確率質量関数（PMF）配列
        dwell_values: 滞在時間（5分単位）のサポート値
        dwell_pmf: 滞在時間の確率質量関数（PMF）配列
        profile_ev_ids: プロファイルCSVから読み込んだEV識別子配列
        profile_dwell_units: プロファイルCSVから読み込んだ滞在単位数配列
        profile_needed_soc: プロファイルCSVから読み込んだ必要SoCパーセント配列
    """
    arrival_probabilities :np .ndarray
    soc_values :np .ndarray
    soc_cdf :np .ndarray
    needed_soc_values :np .ndarray
    needed_soc_pmf :np .ndarray
    dwell_values :np .ndarray
    dwell_pmf :np .ndarray
    profile_ev_ids :np .ndarray
    profile_dwell_units :np .ndarray
    profile_needed_soc :np .ndarray


def _read_numeric_column (path :str ,value_index :int )->Tuple [np .ndarray ,np .ndarray ]:
    """CSVファイルを読み込み、0列目（キー）と value_index 列目（値）を配列として返す。

    ファイルが存在しない場合や空の場合は空配列を返す（エラーにしない）。
    """
    if not path :
        return np .empty (0 ,dtype =np .int64 ),np .empty (0 ,dtype =np .float64 )
    file_path =Path (path )
    if not file_path .exists ():
        return np .empty (0 ,dtype =np .int64 ),np .empty (0 ,dtype =np .float64 )
    keys =[]
    vals =[]
    with file_path .open (newline ="",encoding ="utf-8-sig")as f :
        reader =csv .reader (f )
        header_skipped =False
        for row in reader :
            # ヘッダ行をスキップする
            if not header_skipped :
                header_skipped =True
                continue
            if len (row )<=value_index :
                continue
            try :
                key =int (row [0 ].strip ())
                val =float (row [value_index ])
            except ValueError :
                continue
            keys .append (key )
            vals .append (val )
    return np .asarray (keys ,dtype =np .int64 ),np .asarray (vals ,dtype =np .float64 )


def _ensure_length (arr :np .ndarray ,target_len :int )->np .ndarray :
    """配列を target_len の長さに調整して返す。

    短い場合はタイル繰り返し、長い場合はトリムする。
    空配列の場合はゼロ配列を返す。
    """
    if len (arr )==target_len :
        return arr
    if len (arr )==0 :
        return np .zeros (target_len ,dtype =np .float64 )
    reps =int (np .ceil (target_len /len (arr )))
    tiled =np .tile (arr ,reps )[:target_len ]
    return tiled .astype (np .float64 )


def _load_arrival_probabilities (path :str ,episode_steps :int ,baseline_prob :float )->np .ndarray :
    """CSVから到着確率ウェイトを読み込み、baseline_prob を総量基準として正規化した確率配列を返す。

    ウェイトの合計が 0 の場合は一様分布（全ステップで baseline_prob）にフォールバックする。
    """
    _ ,weights =_read_numeric_column (path ,1 )
    weights =np .clip (weights ,a_min =0.0 ,a_max =None )
    weights =_ensure_length (weights ,episode_steps )
    total =float (weights .sum ())
    if total <=0 :
        # ウェイトが無効な場合は一様確率にフォールバックする
        return np .clip (np .full (episode_steps ,baseline_prob ,dtype =np .float64 ),0.0 ,1.0 )
    # baseline_prob を全ステップに均等に割り当てるようスケーリングする
    scale =baseline_prob *episode_steps /total
    probs =np .clip (weights *scale ,0.0 ,1.0 )
    return probs


def load_arrival_probabilities (path :str ,episode_steps :int ,baseline_prob :float )->np .ndarray :
    """ステーション別の時刻別EV到着確率配列を読み込んで返す。

    Args:
        path: 到着確率CSVファイルのパス（1行=1タイムステップのウェイト）。
        episode_steps: エピソード長（タイムステップ数）。
        baseline_prob: 全ステップの合計到着期待値の基準確率。

    Returns:
        shape (episode_steps,) の到着確率配列（値は [0, 1]）。
    """
    return _load_arrival_probabilities (path ,episode_steps ,baseline_prob )


def _load_soc_cdf (path :str )->Tuple [np .ndarray ,np .ndarray ]:
    """到着時SoC分布CSVを読み込み、SoC値と累積分布関数（CDF）を返す。

    パーセンタイル値をCDFに変換する。データが無効な場合は一様分布にフォールバックする。
    """
    soc_values ,percents =_read_numeric_column (path ,2 )
    percents =np .clip (percents ,a_min =0.0 ,a_max =None )
    if len (percents )==0 or float (percents .sum ())<=0 :
        # データが空または全ゼロの場合は一様CDF（0〜100%）にフォールバックする
        soc_values =np .arange (0 ,101 ,dtype =np .int64 )
        cdf =np .linspace (0.0 ,1.0 ,len (soc_values ),dtype =np .float64 )
        return soc_values ,cdf
    cdf =np .cumsum (percents )
    cdf /=cdf [-1 ]
    return soc_values ,cdf


def _load_survival_to_pmf (path :str ,max_value :int =100 )->Tuple [np .ndarray ,np .ndarray ]:
    """生存関数（Survival Function）CSVを読み込み、確率質量関数（PMF）に変換して返す。

    生存関数の差分をとることでPMFを計算する。単調減少でないデータは補正する。
    データが無効な場合は一様分布にフォールバックする。
    """
    values ,survival =_read_numeric_column (path ,1 )
    survival =np .clip (survival ,a_min =0.0 ,a_max =100.0 )
    grid =np .zeros (max_value +2 ,dtype =np .float64 )
    for v ,s in zip (values ,survival ):
        if 0 <=v <=max_value :
            grid [v ]=s
    grid [max_value +1 ]=0.0
    # 単調減少でない箇所を補正する（前の値より大きければ前の値に揃える）
    for idx in range (1 ,len (grid )):
        if grid [idx ]>grid [idx -1 ]:
            grid [idx ]=grid [idx -1 ]
    pmf =np .maximum (grid [:-1 ]-grid [1 :],0.0 )
    total =float (pmf .sum ())
    if total <=0 :
        # データが無効な場合は一様PMFにフォールバックする
        values =np .arange (0 ,max_value +1 ,dtype =np .int64 )
        pmf =np .ones_like (values ,dtype =np .float64 )
        pmf /=pmf .sum ()
        return values ,pmf
    pmf /=total
    support =np .arange (0 ,max_value +1 ,dtype =np .int64 )
    return support ,pmf


def _load_ev_profile_csv (path :str )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """EVプロファイルCSVを読み込み、EV識別子・滞在単位数・必要SoCを配列として返す。

    読み込むカラム:
        - ev_id: EV識別子（整数）
        - connection_minutes_5min: 5分単位の接続ステップ数（浮動小数点）
        - required_soc_percent: 必要SoCのパーセント値（浮動小数点）

    ファイルが見つからない場合や行が不正な場合は空配列を返す。
    """
    if not path :
        return (
        np .empty (0 ,dtype =np .int64 ),
        np .empty (0 ,dtype =np .float64 ),
        np .empty (0 ,dtype =np .float64 ),
        )
    ev_ids =[]
    dwell_units =[]
    needed_soc =[]
    try :
        with open (path ,newline ="",encoding ="utf-8-sig")as f :
            reader =csv .DictReader (f )
            for row in reader :
                try :
                    ev_id =int (row .get ("ev_id"))
                    dwell =float (row .get ("connection_minutes_5min"))
                    needed =float (row .get ("required_soc_percent"))
                except (TypeError ,ValueError ):
                    continue
                ev_ids .append (ev_id )
                dwell_units .append (dwell )
                needed_soc .append (needed )
    except FileNotFoundError :
        return (
        np .empty (0 ,dtype =np .int64 ),
        np .empty (0 ,dtype =np .float64 ),
        np .empty (0 ,dtype =np .float64 ),
        )
    return (
    np .asarray (ev_ids ,dtype =np .int64 ),
    np .asarray (dwell_units ,dtype =np .float64 ),
    np .asarray (needed_soc ,dtype =np .float64 ),
    )


def _build_hist (values :np .ndarray ,lower :float ,upper :float )->tuple [np .ndarray ,np .ndarray ]:
    """値配列からヒストグラム（PMF）を構築して返す。

    値を [lower, upper] にクリップし、ユニーク値ごとの出現頻度からPMFを計算する。
    値が空の場合は一様分布にフォールバックする。
    """
    if values .size ==0 :
        # 値が空の場合は等間隔グリッドの一様PMFを返す
        grid =np .linspace (lower ,upper ,num =int (upper -lower +1 ),dtype =np .float64 )
        pmf =np .ones_like (grid ,dtype =np .float64 )
        pmf /=pmf .sum ()
        return grid ,pmf

    vals =np .clip (values .astype (np .float64 ),lower ,upper ).reshape (-1 )
    unique_vals ,counts =np .unique (vals ,return_counts =True )
    pmf =counts .astype (np .float64 )
    pmf /=pmf .sum ()
    return unique_vals ,pmf


def load_accurate_ev_info (
arrival_path :str ,
soc_arrival_path :str ,
profile_csv_path :str ,
episode_steps :int ,
baseline_arrival_prob :float ,
)->AccurateEVInfo :
    """実データ由来のEVプロファイル情報を読み込み AccurateEVInfo として返す。

    Args:
        arrival_path: 到着確率ウェイトCSVのパス（空文字の場合は baseline_arrival_prob を一様適用）。
        soc_arrival_path: 到着時SoC分布CSVのパス（EV_SOC_ARRIVAL_DISTRIBUTION_PATH）。
        profile_csv_path: EVプロファイルCSVのパス（ev_id, connection_minutes_5min, required_soc_percent）。
        episode_steps: エピソード長（タイムステップ数）。
        baseline_arrival_prob: 到着確率の基準値（_load_arrival_probabilities に渡す）。

    Returns:
        読み込んだ全プロファイル情報を格納した AccurateEVInfo インスタンス。
    """
    # 到着確率プロファイルを読み込む
    arrival_probs =_load_arrival_probabilities (arrival_path ,episode_steps ,baseline_arrival_prob )
    # 到着時SoC分布（CDF）を読み込む
    soc_values ,soc_cdf =_load_soc_cdf (soc_arrival_path )
    # EVプロファイルCSV（EV識別子・滞在・必要SoC）を読み込む
    profile_ev_ids ,profile_dwell_units ,profile_needed_soc =_load_ev_profile_csv (profile_csv_path )

    # 必要SoCと滞在時間のヒストグラム（PMF）を構築する
    needed_values ,needed_pmf =_build_hist (profile_needed_soc ,0.0 ,100.0 )
    dwell_values ,dwell_pmf =_build_hist (profile_dwell_units ,0.0 ,1_000.0 )
    return AccurateEVInfo (
    arrival_probabilities =arrival_probs ,
    soc_values =soc_values ,
    soc_cdf =soc_cdf ,
    needed_soc_values =needed_values ,
    needed_soc_pmf =needed_pmf ,
    dwell_values =dwell_values ,
    dwell_pmf =dwell_pmf ,
    profile_ev_ids =profile_ev_ids ,
    profile_dwell_units =profile_dwell_units ,
    profile_needed_soc =profile_needed_soc ,
    )
