import pandas as pd 
from pathlib import Path 
import re 


INPUT_DIR =Path (".")
GLOB_PATTERN ="*pal.csv"
TARGET_REGION ="CAPITL"
STEP_START =0 


def find_col (cols ,target ):
    "Documentation."
    norm =lambda s :re .sub (r"[^a-z0-9]","",str (s ).lower ())
    t =norm (target )
    for c in cols :
        if norm (c )==t :
            return c 
    return None 

def sanitize (name :str )->str :
    "Documentation."
    s =str (name ).upper ()
    s =s .replace ("N.Y.C.","NYC")
    s =re .sub (r"[^\w]+","_",s )
    s =re .sub (r"_+","_",s ).strip ("_")
    return s 

    
files =sorted (INPUT_DIR .glob (GLOB_PATTERN ))
if not files :
    raise FileNotFoundError ("Error: invalid runtime state.")

for fp in files :

    df =pd .read_csv (fp ,sep =None ,engine ="python")

    
    col_ts =find_col (df .columns ,"Time Stamp")
    col_name =find_col (df .columns ,"Name")
    col_load =find_col (df .columns ,"Load")
    if not all ([col_ts ,col_name ,col_load ]):
        raise ValueError ("Error: invalid runtime state.")

        
    reg_norm =lambda s :re .sub (r"\s+","",str (s )).upper ()
    mask =df [col_name ].apply (lambda x :reg_norm (x )==reg_norm (TARGET_REGION ))
    g =df .loc [mask ,[col_ts ,col_load ]].copy ()
    if g .empty :
        print ("Info.")
        continue 

        
    g [col_ts ]=pd .to_datetime (g [col_ts ],errors ="coerce",infer_datetime_format =True )
    g =g .dropna (subset =[col_ts ])
    g ["date"]=g [col_ts ].dt .strftime ("%Y%m%d")

    
    for date_key ,sub in g .groupby ("date"):
        sub =sub .sort_values (col_ts ).reset_index (drop =True )
        
        sub .insert (0 ,"step",range (STEP_START ,STEP_START +len (sub )))
        
        out =sub [["step",col_load ]].copy ()
        out .columns =["step","demand_adjustment"]

        
        out_name =f"NYISO_{date_key}_{sanitize(TARGET_REGION)}.csv"
        out_path =INPUT_DIR /out_name 
        out .to_csv (out_path ,index =False )
        print (f"saved: {out_name}  rows={len(out)}")
