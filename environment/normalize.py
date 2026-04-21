"""
normalize.py
============
観測データの正規化・逆正規化ユーティリティ。

正規化の意図:
- ニューラルネットワークへの入力をおおよそ [-1, 1] に収める
- SoC: soc_kWh / 100.0 で [0, 1] に正規化（EV容量 100kWh 相当として扱う）
- 残り時間: steps_remaining / EPISODE_STEPS で [0, 1] に正規化
- 必要SoC: needed_kWh / 100.0 で [-1, 1] にクランプ
- ステーション電力: power_kw / (MAX_EV_POWER_KW * MAX_EV_PER_STATION) で正規化
- AG要請: (raw - center) / scale で [-1, 1] に正規化
- denormalize_soc: 正規化SoC → kWh（Actorのアップデート時にSoC制約適用のために使用）
"""

import numpy as np
import torch
from Config import EPISODE_STEPS ,MAX_EV_PER_STATION ,MAX_EV_POWER_KW
from environment.readcsv import TARGET_MIN ,TARGET_MAX
from environment.observation_config import (
LOCAL_USE_STATION_POWER ,
LOCAL_DEMAND_STEPS ,
LOCAL_USE_STEP ,
)


# AG要請値の正規化に使う中心値・スケール
# readcsv.py の TARGET_MIN / TARGET_MAX から計算する
AG_REQUEST_CENTER =float ((TARGET_MIN +TARGET_MAX )/2.0 )
AG_REQUEST_SCALE =float ((TARGET_MAX -TARGET_MIN )/2.0 )


def normalize_observation (obs ):
   """観測テンソル (num_stations, state_dim) を正規化して返す。

   各ステーションごとに以下の順序で特徴量を正規化する:
   1. EVスロットブロック（presence, soc, remaining_time, needed_soc）
   2. ステーション合計電力（LOCAL_USE_STATION_POWER が True の場合）
   3. AG要請先読み値（LOCAL_DEMAND_STEPS > 0 の場合）
   4. 現在ステップ番号（LOCAL_USE_STEP が True の場合）
   """
   if isinstance (obs ,np .ndarray ):
       obs =torch .from_numpy (obs ).float ()

   normalized_obs =obs .clone ()

   # 各ステーションの観測を個別に正規化する
   for station_idx in range (obs .shape [0 ]):
       station_obs =obs [station_idx ]
       state_dim =station_obs .shape [-1 ]

       # EVスロットブロックは 4特徴量×MAX_EV_PER_STATION 次元
       ev_block_dim =4 *MAX_EV_PER_STATION
       ev_block_end =min (ev_block_dim ,state_dim )

       # 各EVスロットを正規化する
       num_evs =ev_block_end //4
       for j in range (num_evs ):
           base =j *4
           if base +3 <ev_block_end :
               # presence は正規化不要（0 or 1）
               normalized_obs [station_idx ,base ]=obs [station_idx ,base ]
               # SoC: kWh を 100.0 で割って [0, 1] に正規化
               normalized_obs [station_idx ,base +1 ]=obs [station_idx ,base +1 ]/100.0
               # 残り時間: ステップ数を EPISODE_STEPS で割って [0, 1] に正規化
               normalized_obs [station_idx ,base +2 ]=obs [station_idx ,base +2 ]/EPISODE_STEPS
               # 必要SoC: kWh を 100.0 で割り [-1, 1] にクランプ
               normalized_obs [station_idx ,base +3 ]=torch .clamp (obs [station_idx ,base +3 ]/100.0 ,-1.0 ,1.0 )


       idx =ev_block_end

       # ステーション合計電力を正規化（最大電力で割り [-1, 1] にクランプ）
       if LOCAL_USE_STATION_POWER and idx <state_dim :
           max_station_power =MAX_EV_POWER_KW *MAX_EV_PER_STATION
           if max_station_power <=0 :
               max_station_power =1.0
           normalized_obs [station_idx ,idx ]=torch .clamp (
           obs [station_idx ,idx ]/max_station_power ,-1.0 ,1.0
           )
           idx +=1


       # AG要請先読み値を正規化（中心・スケールで [-1, 1] にクランプ）
       if LOCAL_DEMAND_STEPS >0 and idx <state_dim :
           L_local =int (LOCAL_DEMAND_STEPS )
           end_ag =min (idx +L_local ,state_dim -(1 if LOCAL_USE_STEP else 0 ))
           if end_ag >idx :
               ag_slice =slice (idx ,end_ag )
               normalized_obs [station_idx ,ag_slice ]=torch .clamp (
               (obs [station_idx ,ag_slice ]-AG_REQUEST_CENTER )/AG_REQUEST_SCALE ,
               -1.0 ,
               1.0 ,
               )
               idx =end_ag


       # 現在ステップ番号を正規化（EPISODE_STEPS で割り [0, 1] にクランプ）
       if LOCAL_USE_STEP and idx <state_dim :
           normalized_obs [station_idx ,idx ]=torch .clamp (
           obs [station_idx ,idx ]/EPISODE_STEPS ,0.0 ,1.0
           )

   return normalized_obs


def denormalize_observation (normalized_obs ,archive_dir =None ):
   """正規化済み観測テンソルを人間が読める形式に逆正規化して辞書で返す。

   station_1 のみを対象とし、各EVスロットの SoC / 残り時間 / 必要SoC および
   AG要請先読みリストと現在ステップをデコードする。
   """
   if isinstance (normalized_obs ,np .ndarray ):
       normalized_obs =torch .from_numpy (normalized_obs ).float ()

   # 先頭ステーション（station_idx=0）のみ逆正規化する
   station_idx =0
   station_obs =normalized_obs [station_idx ]
   state_dim =station_obs .shape [-1 ]

   # EVスロットブロックの次元数を計算する
   ev_block_dim =4 *MAX_EV_PER_STATION
   ev_block_end =min (ev_block_dim ,state_dim )
   num_ev_features =ev_block_end //4

   station_data ={
   'station_id':station_idx +1 ,
   'evs':[],
   'ag_requests':{}
   }

   # EVスロットを逆正規化して辞書リストを作成する
   for j in range (num_ev_features ):
       base =j *4
       if base +3 <state_dim -4 :
           presence =normalized_obs [station_idx ,base ].item ()
           # 正規化SoC → kWh（× 100.0）
           raw_soc =normalized_obs [station_idx ,base +1 ].item ()*100.0
           # 正規化残り時間 → ステップ数（× EPISODE_STEPS）
           raw_remaining_time =normalized_obs [station_idx ,base +2 ].item ()*EPISODE_STEPS
           # 正規化必要SoC → kWh（× 100.0）
           raw_needed_soc =normalized_obs [station_idx ,base +3 ].item ()*100.0

           if presence >=0.5 :
               station_data ['evs'].append ({
               'ev_index':j +1 ,
               'soc':raw_soc ,
               'remaining_time':raw_remaining_time ,
               'needed_soc':raw_needed_soc
               })


   # 末尾特徴量のインデックスを進める
   tail_idx =ev_block_end
   if LOCAL_USE_STATION_POWER and tail_idx <state_dim :
       tail_idx +=1

   # AG要請先読みリストを逆正規化する
   if LOCAL_DEMAND_STEPS >0 and tail_idx <state_dim :
       L_local =int (LOCAL_DEMAND_STEPS )
       end_ag =min (tail_idx +L_local ,state_dim -(1 if LOCAL_USE_STEP else 0 ))
       if end_ag >tail_idx :
           ag_norm =normalized_obs [station_idx ,tail_idx :end_ag ]
           ag_list =(ag_norm *AG_REQUEST_SCALE +AG_REQUEST_CENTER ).cpu ().tolist ()
       else :
           ag_list =[]
   else :
       ag_list =[]


   # 現在ステップ番号を逆正規化する（末尾インデックス）
   if LOCAL_USE_STEP and state_dim >0 :
       step_val =normalized_obs [station_idx ,state_dim -1 ].item ()*EPISODE_STEPS
   else :
       step_val =0.0

   if ag_list :
       station_data ['ag_requests']={
       'lookahead':ag_list ,
       'current_step':step_val ,
       }

   return {'station_1':station_data }


def denormalize_soc (normalized_soc ):
   """正規化SoC値 → kWh に逆変換する。

   Actorのアップデート時にSoC制約を適用するために使用する。
   正規化は soc_kWh / 100.0 で行われているため、× 100.0 で復元する。
   """
   return normalized_soc *100.0


def normalize_soc (raw_soc ):
   """SoC（kWh） → 正規化値 [0, 1] に変換する。

   raw_soc / 100.0 で計算する。
   """
   return raw_soc /100.0


def denormalize_remaining_time (normalized_time ):
   """正規化済み残り時間 → ステップ数に逆変換する。

   × EPISODE_STEPS で復元する。
   """
   return normalized_time *EPISODE_STEPS


def normalize_remaining_time (raw_time ):
   """残り時間（ステップ数） → 正規化値 [0, 1] に変換する。

   raw_time / EPISODE_STEPS で計算する。
   """
   return raw_time /EPISODE_STEPS


def denormalize_ag_request (normalized_ag ):
   """正規化済みAG要請値 → 元のkW値に逆変換する。

   normalized_ag * AG_REQUEST_SCALE + AG_REQUEST_CENTER で復元する。
   """
   return normalized_ag *AG_REQUEST_SCALE +AG_REQUEST_CENTER


def normalize_ag_request (raw_ag ):
   """AG要請値（kW） → 正規化値 [-1, 1] に変換する。

   (raw_ag - AG_REQUEST_CENTER) / AG_REQUEST_SCALE で計算し、[-1, 1] にクランプする。
   """
   return torch .clamp ((raw_ag -AG_REQUEST_CENTER )/AG_REQUEST_SCALE ,-1.0 ,1.0 )
