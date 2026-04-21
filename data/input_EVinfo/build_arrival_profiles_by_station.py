

"Documentation."

from __future__ import annotations 

import argparse 
import csv 
import re 
from collections import defaultdict 
from dataclasses import dataclass 
from datetime import datetime 
from pathlib import Path 


EPISODE_STEPS_DEFAULT =288 
STEP_MINUTES_DEFAULT =5 


@dataclass (frozen =True )
class Args :
    input_csv :str 
    out_dir :str 
    episode_steps :int 
    step_minutes :int 
    station_col :str 
    start_dt_col :str 
    start_tz_col :str 


def _parse_args ()->Args :
    p =argparse .ArgumentParser ()
    p .add_argument (
    "--input",
    default =str (
    Path ("input_EVinfo")
    /"Electric_Vehicle_Charging_Station_Data_-8671638762898357044.csv"
    ),
    help ="Text.",
    )
    p .add_argument (
    "--out-dir",
    default =str (Path ("input_EVinfo")/"arrivals_by_station"),
    help ="Text.",
    )
    p .add_argument ("--episode-steps",type =int ,default =EPISODE_STEPS_DEFAULT )
    p .add_argument ("--step-minutes",type =int ,default =STEP_MINUTES_DEFAULT )
    p .add_argument ("--station-col",default ="Station_Name")
    p .add_argument ("--start-dt-col",default ="Start_Date___Time")
    p .add_argument ("--start-tz-col",default ="Start_Time_Zone")
    ns =p .parse_args ()
    return Args (
    input_csv =ns .input ,
    out_dir =ns .out_dir ,
    episode_steps =ns .episode_steps ,
    step_minutes =ns .step_minutes ,
    station_col =ns .station_col ,
    start_dt_col =ns .start_dt_col ,
    start_tz_col =ns .start_tz_col ,
    )


def _sanitize_filename (name :str )->str :
    s =name .strip ()
    
    s =re .sub (r'[<>:"/\\\\|?*]+',"_",s )
    s =re .sub (r"\s+"," ",s ).strip ()
    
    s =s .strip (" .")
    if not s :
        s ="UNKNOWN_STATION"
        
    return s [:120 ]


def _parse_start_dt (raw :str )->datetime |None :
    "Documentation."
    if not raw :
        return None 
    s =raw .strip ()
    
    for fmt in ("%m/%d/%Y %H:%M","%m/%d/%y %H:%M"):
        try :
            return datetime .strptime (s ,fmt )
        except ValueError :
            continue 
    return None 


def _step_from_dt (dt :datetime ,step_minutes :int ,episode_steps :int )->int :
    minute_of_day =dt .hour *60 +dt .minute 
    step =(minute_of_day //step_minutes )+1 
    if step <1 :
        return 1 
    if step >episode_steps :
        return episode_steps 
    return int (step )


def main ()->None :
    args =_parse_args ()
    in_path =Path (args .input_csv )
    out_dir =Path (args .out_dir )
    out_dir .mkdir (parents =True ,exist_ok =True )

    
    counts_by_station :dict [str ,list [int ]]=defaultdict (
    lambda :[0 ]*(args .episode_steps +1 )
    )

    total_rows =0 
    used_rows =0 
    missing_station =0 
    missing_dt =0 
    tz_values =defaultdict (int )

    with in_path .open (newline ="",encoding ="utf-8-sig")as f :
        reader =csv .DictReader (f )
        if reader .fieldnames is None :
            raise ValueError ("Error: invalid runtime state.")
        for need in (args .station_col ,args .start_dt_col ):
            if need not in reader .fieldnames :
                raise ValueError ("Error: invalid runtime state.")

        for row in reader :
            total_rows +=1 
            station =(row .get (args .station_col )or "").strip ()
            if not station :
                missing_station +=1 
                continue 
            raw_dt =(row .get (args .start_dt_col )or "").strip ()
            dt =_parse_start_dt (raw_dt )
            if dt is None :
                missing_dt +=1 
                continue 

            tz =(row .get (args .start_tz_col )or "").strip ()
            if tz :
                tz_values [tz ]+=1 

            step =_step_from_dt (dt ,args .step_minutes ,args .episode_steps )
            counts_by_station [station ][step ]+=1 
            used_rows +=1 

            
    written =0 
    for station ,counts in sorted (counts_by_station .items (),key =lambda kv :kv [0 ]):
        safe =_sanitize_filename (station )
        out_path =out_dir /f"Arrival__{safe}.csv"
        with out_path .open ("w",newline ="",encoding ="utf-8-sig")as fo :
            w =csv .writer (fo )
            w .writerow (["arrivaltime","workplace"])
            for step in range (1 ,args .episode_steps +1 ):
                w .writerow ([step ,float (counts [step ])])
        written +=1 

        
    top_tz =sorted (tz_values .items (),key =lambda kv :kv [1 ],reverse =True )[:10 ]
    print (f"Input: {in_path}")
    print (f"Rows: total={total_rows}, used={used_rows}, missing_station={missing_station}, missing_start_dt={missing_dt}")
    print (f"Stations written: {written} -> {out_dir}")
    if top_tz :
        print ("Start_Time_Zone (top):",", ".join ([f"{k}:{v}"for k ,v in top_tz ]))


if __name__ =="__main__":
    main ()

