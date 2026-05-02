"""
readcsv.py
==========
CSV loading utilities for demand-adjustment (AG request) data.

- `load_multiple_demand_files`: load all CSV files in a directory and split
  them into train/test groups.
- `get_random_demand_episode`: sample one episode-length demand sequence from
  a preloaded data pool.

CSV format:
  Each row is one 5-minute time step of demand adjustment in kW.
  The `demand_adjustment` column is preferred; otherwise the first numeric
  column is used.

Normalization:
  Each file is linearly scaled into the range `[TARGET_MIN, TARGET_MAX]`.
"""

from __future__ import annotations

import os
from typing import List ,Dict
import glob

import numpy as np
import pandas as pd


# Output range after demand normalization (kW).
TARGET_MIN =-500.0
TARGET_MAX =1000.0


def load_multiple_demand_files (directory :str =None ,train_split :int =25 )->Dict [str ,List [np .ndarray ]]:
   """
   Load all demand CSV files from a directory and split them into train/test sets.
   """
   from Config import DEMAND_ADJUSTMENT_DIR

   if directory is None :
       directory =DEMAND_ADJUSTMENT_DIR

   if not os .path .exists (directory ):
       error_message =f"Demand directory does not exist: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   pattern =os .path .join (directory ,"*.csv")
   files =sorted (glob .glob (pattern ))

   if len (files )==0 :
       error_message =f"No demand CSV files found in directory: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   train_data =[]
   test_data =[]

   for i ,file_path in enumerate (files ,start =1 ):
       df =pd .read_csv (file_path )

       if 'demand_adjustment'in df .columns :
           data =df ['demand_adjustment'].to_numpy (float )
       else :
           numeric_cols =df .select_dtypes (include =['number']).columns
           if len (numeric_cols )==0 :
               raise ValueError (f"No numeric demand column found in {file_path}")
           data =df [numeric_cols [0 ]].to_numpy (float )

       data_288 =data [:288 ]if len (data )>=288 else data
       if len (data_288 )==0 :
           raise ValueError (f"Demand CSV has no rows: {file_path}")

       a =float (data_288 .min ())
       b =float (data_288 .max ())

       if b -a <1e-6 :
           data_288 =np .full_like (data_288 ,(TARGET_MIN +TARGET_MAX )/2.0 )
       else :
           scale =(TARGET_MAX -TARGET_MIN )/(b -a )
           offset =TARGET_MIN /scale -a
           data_288 =scale *(data_288 +offset )

       if i <=train_split :
           train_data .append (data_288 )
       else :
           test_data .append (data_288 )

   return {
   'train':train_data ,
   'test':test_data
   }


def get_random_demand_episode (data_list :List [np .ndarray ],episode_steps :int =288 )->np .ndarray :
   """
   Return one random episode-length demand sequence from a preloaded data list.
   """
   import random

   if not data_list :
       raise ValueError ("data_list is empty; cannot sample a demand episode.")

   data =random .choice (data_list )

   if len (data )>=episode_steps :
       return data [:episode_steps ]
   else :
       padded =np .zeros (episode_steps ,dtype =float )
       padded [:len (data )]=data
       return padded
