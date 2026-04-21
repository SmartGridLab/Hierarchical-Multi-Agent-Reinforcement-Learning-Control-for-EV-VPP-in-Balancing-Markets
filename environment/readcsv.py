"""
readcsv.py
==========
需給調整量データ（AG要請）のCSV読み込みユーティリティ。

- load_multiple_demand_files: 指定ディレクトリ内の複数CSVファイルを読み込み、
  訓練/テスト分割して返す。train_split引数で訓練用ファイル数を指定する（残りはテスト用）。
- get_random_demand_episode: データプールからランダムに1エピソード分のシーケンスを取得する。

CSVフォーマット:
  各行が1タイムステップ（5分刻み）の需給調整量（kW）を表す。
  'demand_adjustment' カラムを優先的に読み込み、存在しない場合は最初の数値カラムを使用する。

正規化:
  各ファイルの需給データを [TARGET_MIN, TARGET_MAX] の範囲に線形スケーリングする。
"""

from __future__ import annotations

import os
from typing import List ,Dict
import glob

import numpy as np
import pandas as pd


# 需給調整量の正規化後の出力範囲（kW）
# 全CSVファイルのデータをこの範囲に線形スケーリングして揃える
TARGET_MIN =-500.0
TARGET_MAX =1000.0


def load_multiple_demand_files (directory :str =None ,train_split :int =25 )->Dict [str ,List [np .ndarray ]]:
   """指定ディレクトリ内の全CSVファイルを読み込み、訓練/テスト分割した辞書を返す。

   Args:
       directory: CSVファイルが格納されているディレクトリパス。
                  None の場合は Config.DEMAND_ADJUSTMENT_DIR を使用する。
       train_split: 訓練データとして使用するファイル数（先頭 train_split 個）。
                    残りはテストデータとして分類される。

   Returns:
       {'train': [ndarray, ...], 'test': [ndarray, ...]} の辞書。
       各 ndarray は 288 ステップ分の正規化済み需給調整量データ。
   """
   from Config import DEMAND_ADJUSTMENT_DIR

   # directory が None の場合は Config から読み込む
   if directory is None :
       directory =DEMAND_ADJUSTMENT_DIR

   if not os .path .exists (directory ):
       error_message =f"Demand directory does not exist: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   # ディレクトリ内のCSVファイルをソート済みリストで取得する
   pattern =os .path .join (directory ,"*.csv")
   files =sorted (glob .glob (pattern ))

   if len (files )==0 :
       error_message =f"No demand CSV files found in directory: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   # 訓練データとテストデータを格納するリスト
   train_data =[]
   test_data =[]

   for i ,file_path in enumerate (files ,start =1 ):
       try :
           # CSVをDataFrameとして読み込む
           df =pd .read_csv (file_path )

           # 'demand_adjustment' カラムを優先して取得する
           if 'demand_adjustment'in df .columns :
               data =df ['demand_adjustment'].to_numpy (float )
           else :
               # 存在しない場合は最初の数値カラムを使用する
               numeric_cols =df .select_dtypes (include =['number']).columns
               if len (numeric_cols )>0 :
                   data =df [numeric_cols [0 ]].to_numpy (float )
               else :
                   continue

           # エピソード長（288ステップ = 24時間）に切り詰める
           data_288 =data [:288 ]if len (data )>=288 else data

           # [TARGET_MIN, TARGET_MAX] の範囲に線形スケーリングする
           a =float (data_288 .min ())
           b =float (data_288 .max ())

           if b -a <1e-6 :
               # 全値が同じ場合は中央値で埋める
               data_288 =np .full_like (data_288 ,(TARGET_MIN +TARGET_MAX )/2.0 )
           else :
               # 線形スケーリング: data_288 を [TARGET_MIN, TARGET_MAX] に変換する
               SCALE =(TARGET_MAX -TARGET_MIN )/(b -a )
               OFFSET =TARGET_MIN /SCALE -a
               data_288 =SCALE *(data_288 +OFFSET )

           # ファイル番号が train_split 以下なら訓練データ、超えたらテストデータ
           if i <=train_split :
               train_data .append (data_288 )
           else :
               test_data .append (data_288 )

       except Exception :
           continue

   return {
   'train':train_data ,
   'test':test_data
   }


def get_random_demand_episode (data_list :List [np .ndarray ],episode_steps :int =288 )->np .ndarray :
   """データプールからランダムに1エピソード分の需給調整量シーケンスを返す。

   Args:
       data_list: load_multiple_demand_files で取得したデータリスト（train or test）。
       episode_steps: エピソード長（デフォルト 288 ステップ = 24時間）。

   Returns:
       shape (episode_steps,) の需給調整量配列。
       データが episode_steps より短い場合はゼロパディングする。
   """
   import random

   if not data_list :
       raise ValueError ("Error: invalid runtime state.")

   # データプールからランダムに1件選択する
   data =random .choice (data_list )

   # episode_steps に合わせてトリムまたはゼロパディングして返す
   if len (data )>=episode_steps :
       return data [:episode_steps ]
   else :
       # データが短い場合はゼロパディングして長さを揃える
       padded =np .zeros (episode_steps ,dtype =float )
       padded [:len (data )]=data
       return padded
