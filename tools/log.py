import os 
from Config import MAX_EV_PER_STATION ,MAX_EV_POWER_KW ,POWER_TO_ENERGY 


_last_end_snapshot_by_ep ={}
_last_end_snapshot_by_ep_global ={}


def _open_log (archive_dir ,episode ):
    os .makedirs (archive_dir ,exist_ok =True )
    path =os .path .join (archive_dir ,f"debug_local_critic_obs_ep{episode}.txt")
    is_new =not os .path .exists (path )
    f =open (path ,'a',encoding ='utf-8')
    if is_new :
        f .write ("="*80 +"\n")
        f .write ("Text.")
        f .write ("="*80 +"\n\n")
    return f ,path 


def _open_log_global (archive_dir ,episode ):
    os .makedirs (archive_dir ,exist_ok =True )
    path =os .path .join (archive_dir ,f"debug_global_critic_obs_ep{episode}.txt")
    is_new =not os .path .exists (path )
    f =open (path ,'a',encoding ='utf-8')
    if is_new :
        f .write ("="*80 +"\n")
        f .write ("Text.")
        f .write ("="*80 +"\n\n")
    return f ,path 


def _ids_from_snapshot (items ):
    return [int (x .get ('id',0 ))for x in items ]


def _write_base_info (f ,info ,include_wave =True ):
    step =info .get ('step_count')
    ag_cur =info .get ('net_demand',0.0 )
    if include_wave :
        f .write ("Text.")
        f .write (f"  step_count: {step}\n")
        if 'wave_offset'in info :
            f .write (f"  wave_offset: {info['wave_offset']}\n")
        if 'wave_phase'in info :
            f .write (f"  wave_phase: {info['wave_phase']}\n")
    else :
        f .write ("Text.")
        f .write (f"  step_count: {step}\n")
    f .write ("Text.")
    
    f .write ("Text.")
    f .write ("Text.")
    if 'ag_lookahead'in info and isinstance (info ['ag_lookahead'],(list ,tuple )):
    
        try :
            seq =', '.join (f"{float(x):.2f}"for x in info ['ag_lookahead'])
        except Exception :
            seq =str (info ['ag_lookahead'])
        f .write ("Text.")
    f .write ("Text.")


def _pad_rows (rows ,width ):
    padded =list (rows )
    while len (padded )<width :
    
        padded .append (["0"]+["0.00"]*(len (rows [0 ])-1 )if rows else ["0"])
    return padded [:width ]


def _write_table_aligned (f ,header_cols ,rows ,widths ):
    def fmt_cell (s ,w ):
        return str (s ).rjust (w )
    header ="    | "+" | ".join (fmt_cell (h ,w )for h ,w in zip (header_cols ,widths ))+" |\n"
    sep ="    |"+"".join ("-"*(w +2 )+"|"for w in widths )+"\n"
    f .write (header )
    f .write (sep )
    for r in rows :
        f .write ("    | "+" | ".join (fmt_cell (c ,w )for c ,w in zip (r ,widths ))+" |\n")


def write_step_log (archive_dir ,episode ,info ,actor_raw_outputs =None ,include_wave =True ):

    f ,_ =_open_log (archive_dir ,episode )
    step =info .get ('step_count')
    f .write (f"--- Step {step} ---\n")

    
    _write_base_info (f ,info ,include_wave =include_wave )

    
    if 'local_rewards'in info or 'global_reward'in info :
        f .write ("Text.")
        if 'local_rewards'in info and isinstance (info ['local_rewards'],list ):
            try :
                f .write ("Text.")
            except Exception :
                pass 
        if 'global_reward'in info :
            f .write ("Text.")

            
        rb =info .get ('reward_breakdown',{})
        if rb :
            g =rb .get ('global',{})
            if g :
                f .write ("Text.")
                f .write (f"    balance_reward:   {float(g.get('balance_reward', 0.0)):8.2f}\n")
                f .write ("Text.")
                if 'abs_deviation'in g :
                    f .write (f"    abs_deviation:    {float(g.get('abs_deviation', 0.0)):8.2f}\n")
                f .write (f"    net_demand:       {float(g.get('net_demand', 0.0)):8.2f} \n")
                f .write ("Text.")
            ps =rb .get ('per_station',[])
            if ps :
                st0 =ps [0 ]if len (ps )>0 else {}
                f .write ("Text.")
                f .write (f"    station_power:    {float(st0.get('station_power', 0.0)):8.2f} \n")
                f .write (f"    direction_case:   {st0.get('direction_case', '')}\n")
                f .write (f"    direction_reward: {float(st0.get('direction_reward', 0.0)):8.2f}\n")
                if 'departure_reward'in st0 :
                    f .write (f"    departure_reward: {float(st0.get('departure_reward', 0.0)):8.2f}\n")
                if 'local_total'in st0 :
                    f .write (f"    local_total:      {float(st0.get('local_total', 0.0)):8.2f}\n")
        f .write ("\n")

        
    prev_end =_last_end_snapshot_by_ep .get (episode ,{}).get (0 ,[])
    prev_end_ids =set (_ids_from_snapshot (prev_end ))

    
    f .write ("Text.")
    pre =info .get ('snapshot_pre',{}).get (0 ,[])
    pre_rows =[]
    for item in pre :
        soc =float (item .get ('soc',0.0 ))
        needed_soc =float (item .get ('needed_soc',0.0 ))
        target_soc =float (item .get ('target_soc',None ))
        if target_soc is None :
            raise ValueError (f"EV {item.get('id', 'unknown')}: target_soc not found in log data. Cannot proceed without target SoC information.")
        pre_rows .append ([
        str (item .get ('id',0 )),
        f"{soc:.2f}",
        f"{float(item.get('remaining_time', 0.0)):.2f}",
        f"{needed_soc:.2f}",
        f"{target_soc:.2f}",
        str (item .get ('switch_count',0 )),
        ])
    pre_rows =_pad_rows (pre_rows ,MAX_EV_PER_STATION )
    _write_table_aligned (f ,
    ["EV#","SOC(%)","Text.","Text.","Text.","Text."],
    pre_rows ,
    [4 ,8 ,8 ,10 ,10 ,8 ]
    )
    arrived =[pid for pid in _ids_from_snapshot (pre )if pid not in prev_end_ids and pid !=0 ]
    f .write ("Text.")

    
    f .write ("Text.")
    after =info .get ('snapshot_after',{}).get (0 ,[])
    after_rows =[]
    for item in after :
        action_scaled =item .get ('action_scaled',0.0 )
        actor_out_30x =""
        if actor_raw_outputs :
            idx =len (after_rows )
            if idx <len (actor_raw_outputs ):
                try :
                    actor_kw =float (actor_raw_outputs [idx ])*MAX_EV_POWER_KW 
                    actor_out_30x =f"{actor_kw * POWER_TO_ENERGY:9.2f}"
                except Exception :
                    actor_out_30x =""
        prev_soc =float (item .get ('prev_soc',0.0 ))
        needed_soc =float (item .get ('needed_soc',0.0 ))
        target_soc =float (item .get ('target_soc',None ))
        if target_soc is None :
            raise ValueError (f"EV {item.get('id', 'unknown')}: target_soc not found in log data. Cannot proceed without target SoC information.")
        after_rows .append ([
        str (item .get ('id',0 )),
        f"{float(item.get('new_soc', 0.0)):.2f}",
        f"{float(item.get('remaining_time', 0.0)):.2f}",
        f"{needed_soc:.2f}",
        f"{target_soc:.2f}",
        f"{prev_soc:.2f}",
        actor_out_30x ,
        f"{float(item.get('delta_soc', 0.0)):.2f}",
        f"{float(item.get('critic_input', 0.0)) * POWER_TO_ENERGY:.2f}",
        str (item .get ('switch_count',0 )),
        ])
    after_rows =_pad_rows (after_rows ,MAX_EV_PER_STATION )
    _write_table_aligned (f ,
    [
    "EV#","Text.","Text.","Text.","Text.",
    "Text.","Text.","Text.","Text.","Text."
    ],
    after_rows ,
    [4 ,8 ,8 ,10 ,10 ,8 ,12 ,10 ,10 ,8 ]
    )
    f .write ("\n")

    
    f .write ("Text.")
    end =info .get ('snapshot_end',{}).get (0 ,[])
    end_rows =[]
    for item in end :
        soc =float (item .get ('soc',0.0 ))
        needed_soc =float (item .get ('needed_soc',0.0 ))
        target_soc =float (item .get ('target_soc',None ))
        if target_soc is None :
            raise ValueError (f"EV {item.get('id', 'unknown')}: target_soc not found in log data. Cannot proceed without target SoC information.")
        end_rows .append ([
        str (item .get ('id',0 )),
        f"{soc:.2f}",
        f"{needed_soc:.2f}",
        f"{target_soc:.2f}",
        f"{float(item.get('remaining_time', 0.0)):.2f}",
        str (item .get ('switch_count',0 )),
        ])
    end_rows =_pad_rows (end_rows ,MAX_EV_PER_STATION )
    _write_table_aligned (f ,
    ["EV#","SOC(%)","Text.","Text.","Text.","Text."],
    end_rows ,
    [4 ,8 ,10 ,10 ,8 ,8 ]
    )

    
    after_ids =set (_ids_from_snapshot (after ))
    end_ids =set (_ids_from_snapshot (end ))
    departed_cnt =len ([i for i in after_ids if i not in end_ids and i !=0 ])
    f .write ("Text.")

    f .write ("\n"+"-"*60 +"\n")
    f .close ()

    
    _last_end_snapshot_by_ep .setdefault (episode ,{})[0 ]=end 


def write_step_log_global (archive_dir ,episode ,info ,include_wave =True ):

    f ,_ =_open_log_global (archive_dir ,episode )
    step =info .get ('step_count')
    f .write (f"--- Step {step} ---\n")

    
    _write_base_info (f ,info ,include_wave =include_wave )

    
    if 'local_rewards'in info or 'global_reward'in info :
        f .write ("Text.")
        if 'local_rewards'in info and isinstance (info ['local_rewards'],list ):
            try :
                f .write ("Text.")
            except Exception :
                pass 
        if 'global_reward'in info :
            f .write ("Text.")

        rb =info .get ('reward_breakdown',{})
        if rb :
            g =rb .get ('global',{})
            if g :
                f .write ("Text.")
                f .write (f"    balance_reward:   {float(g.get('balance_reward', 0.0)):8.2f}\n")
                f .write ("Text.")
                if 'abs_deviation'in g :
                    f .write (f"    abs_deviation:    {float(g.get('abs_deviation', 0.0)):8.2f}\n")
                f .write (f"    net_demand:       {float(g.get('net_demand', 0.0)):8.2f} \n")
                f .write ("Text.")
        f .write ("\n")

        
    num_stations =int (info .get ('num_stations',len (info .get ('snapshot_pre',{}))))
    total_slots =MAX_EV_PER_STATION *num_stations 

    def _flatten_snapshot (key ):
        data =info .get (key ,{})
        rows =[]
        for st in range (num_stations ):
            rows .extend (data .get (st ,[]))
        return rows 

        
    prev_end =_last_end_snapshot_by_ep_global .get (episode ,[])
    prev_end_ids =set (_ids_from_snapshot (prev_end ))

    
    f .write ("Text.")
    pre =_flatten_snapshot ('snapshot_pre')
    pre_rows =[]
    for item in pre :
        soc =float (item .get ('soc',0.0 ))
        needed_soc =float (item .get ('needed_soc',0.0 ))
        target_soc =float (item .get ('target_soc',None ))
        if target_soc is None :
            raise ValueError (f"EV {item.get('id', 'unknown')}: target_soc not found in log data. Cannot proceed without target SoC information.")
        pre_rows .append ([
        str (item .get ('id',0 )),
        f"{soc:.2f}",
        f"{float(item.get('remaining_time', 0.0)):.2f}",
        f"{needed_soc:.2f}",
        f"{target_soc:.2f}",
        str (item .get ('switch_count',0 )),
        ])
    pre_rows =_pad_rows (pre_rows ,total_slots )
    _write_table_aligned (
    f ,
    ["EV#","SOC(%)","Text.","Text.","Text.","Text."],
    pre_rows ,
    [4 ,8 ,8 ,10 ,10 ,8 ]
    )
    arrived =[pid for pid in _ids_from_snapshot (pre )if pid not in prev_end_ids and pid !=0 ]
    f .write ("Text.")

    
    f .write ("Text.")
    after =_flatten_snapshot ('snapshot_after')
    after_rows =[]
    for item in after :
        prev_soc =float (item .get ('prev_soc',0.0 ))
        needed_soc =float (item .get ('needed_soc',0.0 ))
        target_soc =float (item .get ('target_soc',None ))
        if target_soc is None :
            raise ValueError (f"EV {item.get('id', 'unknown')}: target_soc not found in log data. Cannot proceed without target SoC information.")
        after_rows .append ([
        str (item .get ('id',0 )),
        f"{float(item.get('new_soc', 0.0)):.2f}",
        f"{float(item.get('remaining_time', 0.0)):.2f}",
        f"{needed_soc:.2f}",
        f"{target_soc:.2f}",
        f"{prev_soc:.2f}",
        "",
        f"{float(item.get('delta_soc', 0.0)):.2f}",
        f"{float(item.get('critic_input', 0.0)) * POWER_TO_ENERGY:.2f}",
        str (item .get ('switch_count',0 )),
        ])
    after_rows =_pad_rows (after_rows ,total_slots )
    _write_table_aligned (
    f ,
    [
    "EV#","Text.","Text.","Text.","Text.",
    "Text.","Text.","Text.","Text.","Text."
    ],
    after_rows ,
    [4 ,8 ,8 ,10 ,10 ,8 ,12 ,10 ,10 ,8 ]
    )
    f .write ("\n")

    
    f .write ("Text.")
    end =_flatten_snapshot ('snapshot_end')
    end_rows =[]
    for item in end :
        soc =float (item .get ('soc',0.0 ))
        needed_soc =float (item .get ('needed_soc',0.0 ))
        target_soc =float (item .get ('target_soc',None ))
        if target_soc is None :
            raise ValueError (f"EV {item.get('id', 'unknown')}: target_soc not found in log data. Cannot proceed without target SoC information.")
        end_rows .append ([
        str (item .get ('id',0 )),
        f"{soc:.2f}",
        f"{needed_soc:.2f}",
        f"{target_soc:.2f}",
        f"{float(item.get('remaining_time', 0.0)):.2f}",
        str (item .get ('switch_count',0 )),
        ])
    end_rows =_pad_rows (end_rows ,total_slots )
    _write_table_aligned (
    f ,
    ["EV#","SOC(%)","Text.","Text.","Text.","Text."],
    end_rows ,
    [4 ,8 ,10 ,10 ,8 ,8 ]
    )

    
    after_ids =set (_ids_from_snapshot (after ))
    end_ids =set (_ids_from_snapshot (end ))
    departed_cnt =len ([i for i in after_ids if i not in end_ids and i !=0 ])
    f .write ("Text.")

    f .write ("\n"+"-"*60 +"\n")
    f .close ()

    
    _last_end_snapshot_by_ep_global [episode ]=end 


