

from __future__ import annotations 

import random 
import math 
import numpy as np 
import torch 
from environment.ev_info_loader import load_accurate_ev_info ,load_arrival_probabilities 
from Config import (
EV_CAPACITY ,EPISODE_STEPS ,TOL_NARROW_METRICS ,
SOC_WIDE ,
MAX_EV_PER_STATION ,
EV_SOC_ARRIVAL_DISTRIBUTION_PATH ,
EV_PROFILE_DATA_PATH ,
PER_STATION_ARRIVAL_PROFILE_PATHS ,
INITIAL_EVS_PER_STATION ,
NUM_STATIONS ,NUM_EVS ,DEVICE ,GLOBAL_BALANCE_REWARD ,
MAX_EV_POWER_KW ,POWER_TO_ENERGY ,TIME_STEP_MINUTES ,
USE_STATION_TOTAL_POWER_LIMIT ,STATION_MAX_TOTAL_POWER_KW ,
LOCAL_DEFICIT_SHAPING_COEF ,
LOCAL_DEFICIT_SIGMA_POS ,LOCAL_DEFICIT_SIGMA_NEG ,
LOCAL_DEFICIT_EXP_CLIP ,LOCAL_DEFICIT_BUFFER ,
LOCAL_STATION_LIMIT_PENALTY ,
LOCAL_SWITCH_PENALTY ,
USE_SWITCHING_CONSTRAINTS ,
)
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_USE_STATION_POWER ,
LOCAL_DEMAND_STEPS ,

LOCAL_USE_STEP ,
GLOBAL_DEMAND_STEPS ,
)

device =DEVICE 

class EVEnv :
    "Documentation."

    
    
    

    def __init__ (
    self ,
    num_stations :int =NUM_STATIONS ,
    num_evs :int =NUM_EVS ,
    episode_steps :int =EPISODE_STEPS ,
    ):
        self .num_stations =num_stations 
        self .num_evs =num_evs 
        self .episode_steps =episode_steps 
        self .arrival_prob =0.3 

        self .accurate_mode =True 
        self .accurate_ev_info =None 
        self .arrival_profiles_by_station =None 
        if self .accurate_mode :
            self .accurate_ev_info =load_accurate_ev_info (
            "",
            EV_SOC_ARRIVAL_DISTRIBUTION_PATH ,
            EV_PROFILE_DATA_PATH ,
            self .episode_steps ,
            self .arrival_prob ,
            )

            
            paths =list (PER_STATION_ARRIVAL_PROFILE_PATHS or [])
            if len (paths )<=0 :
                raise ValueError ("Error: invalid runtime state.")

                
                
                
            if len (paths )<int (self .num_stations ):
                if int (self .num_stations )%len (paths )!=0 :
                    raise ValueError (
                    "Text."
                    "Text."
                    f" num_stations={self.num_stations}, paths={len(paths)}"
                    )
                group_size =int (self .num_stations )//len (paths )
                expanded_paths =[paths [st //group_size ]for st in range (int (self .num_stations ))]
                paths =expanded_paths 
            elif len (paths )>int (self .num_stations ):
                raise ValueError (
                "Text."
                f" num_stations={self.num_stations}, paths={len(paths)}"
                )

            self .arrival_profiles_by_station =[
            load_arrival_probabilities (p ,self .episode_steps ,self .arrival_prob )for p in paths 
            ]
            if self .arrival_profiles_by_station is None :
                raise ValueError (
                "Accurate mode requires per-station arrival profiles. "
                "Provide PER_STATION_ARRIVAL_PROFILE_PATHS."
                )
        self .arrival_soc_log =[]
        self .arrival_needed_log =[]
        self .arrival_dwell_log =[]
        self .arrivals_this_step =0 
        self .arrivals_by_station =[0 ]*self .num_stations 

        
        self .ev_capacity =EV_CAPACITY 
        self .tol_narrow_metrics =TOL_NARROW_METRICS 
        self .balance_reward =GLOBAL_BALANCE_REWARD 
        self .soc_wide =SOC_WIDE 

        self .max_ev_per_station =MAX_EV_PER_STATION 
        self .use_station_total_power_limit =bool (USE_STATION_TOTAL_POWER_LIMIT )
        self .station_total_power_limit_kw =float (STATION_MAX_TOTAL_POWER_KW )
        self .local_discharge_penalty_coef =0.0 
        self .local_station_limit_penalty_coef =float (LOCAL_STATION_LIMIT_PENALTY )
        self .local_use_switch_features =bool (USE_SWITCHING_CONSTRAINTS )
        self .station_power_limit_kw =torch .full (
        (num_stations ,),
        self .station_total_power_limit_kw ,
        dtype =torch .float32 ,
        device =device ,
        )

        
        self .ev_ids =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )
        self .soc =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .prev_soc =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .target =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .depart =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )
        self .ev_mask =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .bool ,device =device )
        
        self .initial_remaining =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        
        self .arrival_step =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        
        self .switch_count =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )
        self .last_non_zero_state =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )

        
        self .stations_evs ={i :[]for i in range (self .num_stations )}

        
        self .metrics ={
        'total_steps':0 ,
        'surplus_within_narrow':0 ,
        'shortage_within_narrow':0 ,
        'surplus_steps':0 ,
        'shortage_steps':0 ,
        'station_limit_hits':0 ,
        'station_limit_steps':0 ,
        'station_charge_limit_hits':0 ,
        'station_discharge_limit_hits':0 ,
        'station_limit_penalty_total':0.0 ,
        'departing_evs':0 ,
        'departing_evs_soc_met':0 ,
        'total_switches_departed':0 ,
        'total_switches_current':0 
        }



        
        self .net_demand_series =None 
        self .current_net_demand =0.0 

        
        self .step_count =0 
        self .used_ev_ids =set ()
        self .last_station_powers =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )

        
        

        
        self .record_snapshots =False 

        
        self .reset_metrics ()

        
        
        

    def reset (self ,net_demand_series :np .ndarray |None =None ):
    
        self .step_count =0 

        
        if net_demand_series is None :
        
        
        
            try :
                from environment.readcsv import load_multiple_demand_files ,get_random_demand_episode 
                all_demand_data =load_multiple_demand_files (train_split =25 )
                data_pool =all_demand_data .get ('train')or all_demand_data .get ('test')or []
                if not data_pool :
                    raise ValueError ("CSV demand data list is empty.")
                net_demand_series =get_random_demand_episode (data_pool ,self .episode_steps )
            except Exception as e :
                raise ValueError (
                "Text."
                "Text."
                )from e 

                
        self .net_demand_series =torch .tensor (net_demand_series ,dtype =torch .float32 ,device =device )
        self .current_net_demand =float (self .net_demand_series [0 ].item ())

        
        self .ev_ids .zero_ ()
        self .soc .zero_ ()
        self .prev_soc .zero_ ()
        self .target .zero_ ()
        self .depart .zero_ ()
        self .ev_mask .fill_ (False )
        self .initial_remaining .zero_ ()
        self .arrival_step .zero_ ()
        self .switch_count .zero_ ()
        self .last_non_zero_state .zero_ ()
        self .last_station_powers .zero_ ()

        
        self .stations_evs ={i :[]for i in range (self .num_stations )}
        self .arrival_soc_log =[]
        self .arrival_needed_log =[]
        self .arrival_dwell_log =[]
        self .arrivals_this_step =0 
        self .arrivals_by_station =[0 ]*self .num_stations 

        
        random_ids =list (range (self .num_evs ))
        random .shuffle (random_ids )
        random_id_idx =0 
        used_ids =set ()

        
        for st in range (self .num_stations ):
            for i in range (INITIAL_EVS_PER_STATION ):
                if random_id_idx >=len (random_ids ):
                    break 
                init_soc ,target_soc ,dwell_steps ,needed_soc ,profile_ev_id =self ._sample_ev_profile ()
                dep =self .step_count +dwell_steps -1 
                profile_remaining =max (dep -self .step_count ,0 )

                if (
                profile_ev_id is not None 
                and 0 <=profile_ev_id <self .num_evs 
                and profile_ev_id not in used_ids 
                ):
                    assigned_id =profile_ev_id 
                else :
                    while random_id_idx <len (random_ids )and random_ids [random_id_idx ]in used_ids :
                        random_id_idx +=1 
                    if random_id_idx >=len (random_ids ):
                        break 
                    assigned_id =random_ids [random_id_idx ]
                    random_id_idx +=1 

                used_ids .add (assigned_id )
                self ._record_arrival (init_soc ,needed_soc ,dwell_steps )
                self ._set_ev_slot (
                st ,i ,assigned_id ,init_soc ,target_soc ,dep ,
                profile_remaining =profile_remaining 
                )

            if random_id_idx >=len (random_ids ):
                break 

                
        self .used_ev_ids =used_ids


        self .reset_metrics ()

        return self ._get_obs ()

        
        
        

    def _get_obs (self )->np .ndarray :

        all_obs_tensors =[]

        for st in range (self .num_stations ):
        
            active_evs =torch .nonzero (self .ev_mask [st ],as_tuple =False ).squeeze (-1 )

            if len (active_evs )>0 :
            
                sorted_active_evs =self ._sort_active_evs (st ,active_evs )
                
                
                k =len (sorted_active_evs )
                safe_soc =torch .clamp (self .soc [st ,sorted_active_evs ],min =1e-3 )
                ev_socs =safe_soc 
                ev_deps =self .depart [st ,sorted_active_evs ].float ()
                ev_times =(ev_deps -self .step_count )
                ev_needs =self .target [st ,sorted_active_evs ]-ev_socs

                presence_k =torch .ones (k ,device =device )
                if self .local_use_switch_features :
                    ev_switches =self .switch_count [st ,sorted_active_evs ].float ()
                    ev_dirs =self .last_non_zero_state [st ,sorted_active_evs ].float ()
                    ev_matrix =torch .stack (
                    [presence_k ,ev_socs ,ev_times ,ev_needs ,ev_switches ,ev_dirs ],
                    dim =1 ,
                    )
                else :
                    ev_matrix =torch .stack ([presence_k ,ev_socs ,ev_times ,ev_needs ],dim =1 )

                    
                padded =torch .zeros (self .max_ev_per_station ,EV_FEAT_DIM ,device =device )
                padded [:k ]=ev_matrix 
                vec =padded .flatten ()
            else :
            
                vec =torch .zeros (self .max_ev_per_station *EV_FEAT_DIM ,device =device )

                
            tail_features =[]

            
            if LOCAL_USE_STATION_POWER :
                try :
                    if isinstance (self .last_station_powers ,torch .Tensor )and self .last_station_powers .numel ()>=self .num_stations :
                        station_power =self .last_station_powers [st :st +1 ]
                    else :
                        station_power =torch .zeros (1 ,dtype =torch .float32 ,device =device )
                except Exception :
                    station_power =torch .zeros (1 ,dtype =torch .float32 ,device =device )
                tail_features .append (station_power )

                
            if LOCAL_DEMAND_STEPS >0 and self .net_demand_series is not None :
                L_local =int (LOCAL_DEMAND_STEPS )
                start_idx =self .step_count -1 
                
                indices =torch .arange (start_idx ,start_idx +L_local ,dtype =torch .long ,device =device )
                valid_mask =(indices >=0 )&(indices <len (self .net_demand_series ))
                ag_tensor =torch .zeros (L_local ,dtype =torch .float32 ,device =device )
                if valid_mask .any ():
                    ag_tensor [valid_mask ]=self ._clip_demand_for_obs (self .net_demand_series [indices [valid_mask ]])
                tail_features .append (ag_tensor )


                
            if LOCAL_USE_STEP :
                current_step =torch .tensor ([float (self .step_count )],device =device )
                tail_features .append (current_step )

            if tail_features :
                tail_vec =torch .cat (tail_features )
                station_obs =torch .cat ([vec ,tail_vec ])
            else :
                station_obs =vec 

            all_obs_tensors .append (station_obs .unsqueeze (0 ))


            
        return torch .cat (all_obs_tensors ,dim =0 ).to (device =device ,dtype =torch .float32 )

    def _clip_demand_for_obs (self ,demand_values :torch .Tensor )->torch .Tensor :
        active_evs_total =int (self .ev_mask .sum ().item ())
        if active_evs_total <=0 :
            return torch .zeros_like (demand_values ,dtype =torch .float32 ,device =device )
        max_total_power =float (active_evs_total )*float (MAX_EV_POWER_KW )
        return torch .clamp (demand_values ,-max_total_power ,max_total_power )

    def _record_arrival (self ,init_soc :float ,needed_soc :float ,dwell_steps :int ):
        if needed_soc <0 :
            needed_soc =0.0 
        self .arrival_soc_log .append (float (init_soc ))
        self .arrival_needed_log .append (float (needed_soc ))
        dwell_hours =float (max (dwell_steps ,0 ))*TIME_STEP_MINUTES /60.0 
        self .arrival_dwell_log .append (dwell_hours )

    def _sample_ev_profile (self )->tuple [float ,float ,int ,float ,int |None ]:
        def _clip_target_by_physical_limit (init_soc_val :float ,target_soc_val :float ,dwell_steps_val :int )->float :
        
        
            max_reachable_soc =(
            init_soc_val 
            +float (dwell_steps_val )*float (MAX_EV_POWER_KW )*float (POWER_TO_ENERGY )*0.8 
            )
            return float (min (max (target_soc_val ,init_soc_val ),max_reachable_soc ,80.0 ))

            
        if not (self .accurate_mode and self .accurate_ev_info ):
            init_soc =random .uniform (20.0 ,60.0 )
            dwell_steps =random .randint (1 ,max (1 ,int (self .episode_steps )))
            raw_target =init_soc +random .uniform (5.0 ,25.0 )
            target_soc =_clip_target_by_physical_limit (init_soc ,raw_target ,dwell_steps )
            actual_needed =max (target_soc -init_soc ,0.0 )
            return init_soc ,target_soc ,dwell_steps ,actual_needed ,None 

        info =self .accurate_ev_info 

        profile_ev_id :int |None =None 
        if info .profile_ev_ids .size >0 :
            prof_idx =random .randrange (len (info .profile_ev_ids ))
            profile_ev_id =int (info .profile_ev_ids [prof_idx ])
            dwell_steps =int (round (float (info .profile_dwell_units [prof_idx ])))
            needed_soc =float (info .profile_needed_soc [prof_idx ])
        else :
            dwell_raw =float (np .random .choice (info .dwell_values ,p =info .dwell_pmf ))
            dwell_steps =int (round (dwell_raw *60.0 /max (TIME_STEP_MINUTES ,1 )))
            needed_soc =float (np .random .choice (info .needed_soc_values ,p =info .needed_soc_pmf ))*2.0 

        needed_soc =max (needed_soc ,0.0 )

        rand =random .random ()
        soc_idx =int (np .searchsorted (info .soc_cdf ,rand ,side ="right"))
        if soc_idx >=len (info .soc_values ):
            soc_idx =len (info .soc_values )-1 
        init_soc =float (info .soc_values [soc_idx ])
        init_soc =max (0.0 ,min (init_soc ,80.0 ))

        
        dwell_steps =int (dwell_steps )
        if dwell_steps <=0 :
            dwell_steps =1 
        dwell_steps =max (1 ,dwell_steps )

        raw_target =init_soc +needed_soc 
        target_soc =_clip_target_by_physical_limit (init_soc ,raw_target ,dwell_steps )

        actual_needed =max (target_soc -init_soc ,0.0 )

        return init_soc ,target_soc ,dwell_steps ,actual_needed ,profile_ev_id 


    def _set_ev_slot (
    self ,
    station :int ,
    slot_idx :int ,
    ev_id :int ,
    init_soc :float ,
    target_soc :float ,
    dep :int ,
    profile_remaining :float |None =None ,
    ):
    
    
        if profile_remaining is None :
            profile_remaining =max (dep -self .step_count ,0 )
        dep =max (self .step_count ,int (dep ))

        self .ev_ids [station ,slot_idx ]=ev_id 
        self .soc [station ,slot_idx ]=init_soc 
        self .prev_soc [station ,slot_idx ]=init_soc 
        self .target [station ,slot_idx ]=target_soc 
        self .depart [station ,slot_idx ]=dep 
        self .ev_mask [station ,slot_idx ]=True 
        self .initial_remaining [station ,slot_idx ]=float (profile_remaining )
        self .arrival_step [station ,slot_idx ]=float (self .step_count )
        self .switch_count [station ,slot_idx ]=0 
        self .last_non_zero_state [station ,slot_idx ]=0 

        
        self .stations_evs [station ]=[ev for ev in self .stations_evs [station ]if ev .get ('id')!=ev_id ]
        ev =dict (id =ev_id ,station =station ,depart =dep ,soc =init_soc ,target =target_soc ,prev_soc =init_soc )
        self .stations_evs [station ].append (ev )

    def _apply_station_power_limit (
    self ,
    station :int ,
    effective_power_kw :torch .Tensor ,
    )->tuple [torch .Tensor ,bool ,bool ]:
        """Apply a symmetric site-level import/export cap to one station."""
        if (not self .use_station_total_power_limit )or effective_power_kw .numel ()==0 :
            return effective_power_kw ,False ,False 

        limit_kw =self .station_power_limit_kw [station ]
        charge_mask =effective_power_kw >0 
        discharge_mask =effective_power_kw <0 

        total_charge_kw =effective_power_kw [charge_mask ].sum ()
        total_discharge_kw =(-effective_power_kw [discharge_mask ]).sum ()

        charge_limited =bool ((total_charge_kw >(limit_kw +1e-6 )).item ())
        discharge_limited =bool ((total_discharge_kw >(limit_kw +1e-6 )).item ())

        if charge_limited :
            charge_scale =limit_kw /torch .clamp (total_charge_kw ,min =1e-6 )
            effective_power_kw =torch .where (
            charge_mask ,
            effective_power_kw *charge_scale ,
            effective_power_kw ,
            )

        if discharge_limited :
            discharge_scale =limit_kw /torch .clamp (total_discharge_kw ,min =1e-6 )
            effective_power_kw =torch .where (
            discharge_mask ,
            effective_power_kw *discharge_scale ,
            effective_power_kw ,
            )

        return effective_power_kw ,charge_limited ,discharge_limited 

    def _handle_arrivals (self ):
        "Documentation."
        total_arrivals =0 
        arrivals_by_station =[0 ]*self .num_stations 
        for st in range (self .num_stations ):
            avail_slots =(~self .ev_mask [st ]).sum ().item ()
            for _ in range (avail_slots ):
                prof =self .arrival_profiles_by_station [st ]
                step_index =min (max (self .step_count -1 ,0 ),len (prof )-1 )
                arrival_threshold =float (prof [step_index ])
                if random .random ()<arrival_threshold :
                    if self ._spawn_ev (st ):
                        total_arrivals +=1 
                        arrivals_by_station [st ]+=1 
        self .arrivals_this_step =total_arrivals 
        self .arrivals_by_station =arrivals_by_station 

    def step (self ,actions ):
    
        self .begin_step ()
        observation ,local_rewards ,global_reward ,done ,info =self .apply_action (actions )
        return observation ,local_rewards ,global_reward ,done ,info 

    def begin_step (self ):
    
    
        self .step_count +=1 

        
        
        demand_idx =self .step_count -1 

        if 0 <=demand_idx <len (self .net_demand_series ):
            raw_net_demand =float (self .net_demand_series [demand_idx ].item ())
        else :
            raw_net_demand =0.0 

        self .arrivals_this_step =0 
        self .arrivals_by_station =[0 ]*self .num_stations 
        
        self ._handle_arrivals ()

        
        active_evs_total =int (self .ev_mask .sum ().item ())
        if active_evs_total >0 :
            max_total_power =active_evs_total *MAX_EV_POWER_KW 
            
            clipped_net_demand =max (-max_total_power ,min (raw_net_demand ,max_total_power ))
        else :
        
            clipped_net_demand =0.0 

        self .current_net_demand =clipped_net_demand 

        
        self ._snapshot_pre ={}
        if self .record_snapshots :
            for st in range (self .num_stations ):
                details =[]
                active_evs =torch .nonzero (self .ev_mask [st ],as_tuple =False ).squeeze (-1 )
                if len (active_evs )>0 :
                    sorted_evs =self ._sort_active_evs (st ,active_evs )
                    ids =self .ev_ids [st ,sorted_evs ]
                    socs =self .soc [st ,sorted_evs ]
                    
                    remains =(self .depart [st ,sorted_evs ]-self .step_count ).float ()
                    
                    needs =self .target [st ,sorted_evs ]-socs 
                    for i in range (len (sorted_evs )):
                        details .append ({
                        'id':int (ids [i ].item ()),
                        'soc':float (socs [i ].item ()),
                        'remaining_time':float (remains [i ].item ()),
                        'needed_soc':float (needs [i ].item ()),
                        'target_soc':float (self .target [st ,sorted_evs ][i ].item ()),
                        'switch_count':int (self .switch_count [st ,sorted_evs ][i ].item ()),
                        })
                self ._snapshot_pre [st ]=details 
        return self ._get_obs ()

    def apply_action (self ,actions ):
    
        is_final_step =(self .step_count >=self .episode_steps )

        
        
        local_rewards_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        progress_shaping_rewards_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        soc_violations_tensor =torch .zeros (self .num_stations ,dtype =torch .int32 ,device =device )
        departing_ev_stats =[0 ,0 ]
        current_request =self .current_net_demand 

        
        
        station_powers_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        discharge_penalties_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        switch_penalties_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        station_limit_penalties_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        station_charge_limit_hits_tensor =torch .zeros (self .num_stations ,dtype =torch .bool ,device =device )
        station_discharge_limit_hits_tensor =torch .zeros (self .num_stations ,dtype =torch .bool ,device =device )
        total_violations =0 

        
        actual_ev_power_kw_tensor =torch .zeros ((self .num_stations ,self .max_ev_per_station ),device =device )
        snapshot_after ={}
        for st in range (self .num_stations ):
        
            active_evs =torch .nonzero (self .ev_mask [st ],as_tuple =False ).squeeze (-1 )

            if len (active_evs )>0 :
            
                sorted_active_evs =self ._sort_active_evs (st ,active_evs )
                
                
                
                actions_for_station =actions [st ,:len (sorted_active_evs )]
                
                scaled_power_kw =actions_for_station *MAX_EV_POWER_KW 
                

                prev_socs =self .soc [st ,sorted_active_evs ].clone ()

                
                if not is_final_step :
                
                    proposed_delta_kwh =scaled_power_kw *POWER_TO_ENERGY 

                    
                    
                    proposed_delta_kwh [torch .abs (proposed_delta_kwh )<1e-7 ]=0.0 

                    current_state =torch .zeros_like (proposed_delta_kwh ,dtype =torch .int32 )
                    current_state [proposed_delta_kwh >0 ]=1 
                    current_state [proposed_delta_kwh <0 ]=-1 

                    
                    last_states =self .last_non_zero_state [st ,sorted_active_evs ]
                    switched =(current_state !=0 )&(last_states !=0 )&(current_state !=last_states )

                    
                    self .switch_count [st ,sorted_active_evs ]+=switched .int ()
                    
                    non_zero_mask =current_state !=0 
                    if non_zero_mask .any ():
                        updated_last_states =last_states .clone ()
                        updated_last_states [non_zero_mask ]=current_state [non_zero_mask ]
                        self .last_non_zero_state [st ,sorted_active_evs ]=updated_last_states 

                        
                    if switched .any ():
                        self .metrics ['total_switches_current']+=int (switched .sum ().item ())

                        
                    if switched .any ()and USE_SWITCHING_CONSTRAINTS :
                        st_switch_penalty =switched .float ().sum ()*LOCAL_SWITCH_PENALTY 
                        local_rewards_tensor [st ]-=st_switch_penalty 
                        switch_penalties_tensor [st ]=st_switch_penalty 

                        
                    new_socs =prev_socs +proposed_delta_kwh 
                    violations =((new_socs <0.0 )|(new_socs >self .ev_capacity )).sum ()
                    soc_violations_tensor [st ]=violations .int ()
                    total_violations +=violations .item ()
                    
                    new_socs =torch .clamp (new_socs ,0.0 ,self .ev_capacity )

                    
                    actual_delta_kwh =new_socs -prev_socs 
                    effective_power_kw =actual_delta_kwh /POWER_TO_ENERGY 
                    charge_excess_kw =torch .clamp (
                    effective_power_kw [effective_power_kw >0 ].sum ()-self .station_power_limit_kw [st ],
                    min =0.0 ,
                    )
                    discharge_excess_kw =torch .clamp (
                    (-effective_power_kw [effective_power_kw <0 ]).sum ()-self .station_power_limit_kw [st ],
                    min =0.0 ,
                    )
                    station_limit_penalty =self .local_station_limit_penalty_coef *(
                    charge_excess_kw +discharge_excess_kw 
                    )
                    local_rewards_tensor [st ]-=station_limit_penalty 
                    station_limit_penalties_tensor [st ]=station_limit_penalty 
                    effective_power_kw ,charge_limited ,discharge_limited =self ._apply_station_power_limit (
                    st ,effective_power_kw 
                    )
                    actual_delta_kwh =effective_power_kw *POWER_TO_ENERGY 
                    new_socs =prev_socs +actual_delta_kwh 
                    self .prev_soc [st ,sorted_active_evs ]=prev_socs 
                    self .soc [st ,sorted_active_evs ]=new_socs 

                    discharge_energy_kwh =torch .clamp (-actual_delta_kwh ,min =0.0 ).sum ()
                    discharge_penalty =self .local_discharge_penalty_coef *discharge_energy_kwh 
                    local_rewards_tensor [st ]-=discharge_penalty 
                    discharge_penalties_tensor [st ]=discharge_penalty 

                    if charge_limited :
                        station_charge_limit_hits_tensor [st ]=True 
                    if discharge_limited :
                        station_discharge_limit_hits_tensor [st ]=True 

                    for i ,ev_idx in enumerate (sorted_active_evs ):
                        ev_id =int (self .ev_ids [st ,ev_idx ].item ())
                        for ev_dict in self .stations_evs [st ]:
                            if ev_dict ['id']==ev_id :
                                ev_dict ['soc']=float (new_socs [i ].item ())
                                break 

                                
                    target =self .target [st ,sorted_active_evs ]
                    deficit_before =torch .clamp (target -prev_socs +LOCAL_DEFICIT_BUFFER ,min =0.0 )
                    deficit_after =torch .clamp (target -new_socs +LOCAL_DEFICIT_BUFFER ,min =0.0 )
                    delta =deficit_before -deficit_after 

                    
                    active =(deficit_before >0 )&(deficit_after >0 )

                    
                    pos_mag =torch .exp (deficit_after /LOCAL_DEFICIT_SIGMA_POS ).clamp (max =LOCAL_DEFICIT_EXP_CLIP )
                    neg_mag =torch .exp (deficit_after /LOCAL_DEFICIT_SIGMA_NEG ).clamp (max =LOCAL_DEFICIT_EXP_CLIP )

                    r_shaping =torch .zeros_like (deficit_after )
                    r_shaping [(delta >0 )&active ]=pos_mag [(delta >0 )&active ]
                    r_shaping [(delta <0 )&active ]=-neg_mag [(delta <0 )&active ]

                    
                    r_shaping =r_shaping *torch .abs (delta )

                    
                    
                    elapsed =float (self .step_count )-self .arrival_step [st ,sorted_active_evs ].float ()
                    rem =(self .initial_remaining [st ,sorted_active_evs ].float ()-elapsed ).clamp (min =1.0 )
                    r_shaping =r_shaping /rem 

                    shaping_sum =LOCAL_DEFICIT_SHAPING_COEF *r_shaping .sum ()
                    local_rewards_tensor [st ]+=shaping_sum 
                    progress_shaping_rewards_tensor [st ]=shaping_sum 

                    effective_power_kw =actual_delta_kwh /POWER_TO_ENERGY 
                else :
                
                    new_socs =prev_socs 
                    actual_delta_kwh =torch .zeros_like (new_socs )
                    effective_power_kw =torch .zeros_like (new_socs )

                station_powers_tensor [st ]=effective_power_kw .sum ()

                
                
                
                critic_power =effective_power_kw .clone ()

                
                actual_ev_power_kw_tensor [st ,:len (sorted_active_evs )]=critic_power 

                
                if self .record_snapshots :
                
                    remains =(self .depart [st ,sorted_active_evs ]-self .step_count ).float ()
                    
                    needs_before =self .target [st ,sorted_active_evs ]-prev_socs 
                    ids =self .ev_ids [st ,sorted_active_evs ]
                    details_after =[]
                    for i in range (len (sorted_active_evs )):
                        details_after .append ({
                        'id':int (ids [i ].item ()),
                        'remaining_time':float (remains [i ].item ()),
                        'needed_soc':float (needs_before [i ].item ()),
                        'prev_soc':float (prev_socs [i ].item ()),
                        'action_scaled':float (scaled_power_kw [i ].item ()),
                        'new_soc':float (new_socs [i ].item ()),
                        'delta_soc':float (actual_delta_kwh [i ].item ()),
                        'critic_input':float (critic_power [i ].item ()),
                        'target_soc':float (self .target [st ,sorted_active_evs ][i ].item ()),
                        'switch_count':int (self .switch_count [st ,sorted_active_evs ][i ].item ()),
                        })
                    snapshot_after [st ]=details_after 
                else :
                    snapshot_after [st ]=[]
            else :
                snapshot_after [st ]=[]

                
        total_ev_transport =station_powers_tensor .sum ().item ()

        
        self .last_station_powers =station_powers_tensor .clone ()

        
        obs_after_action =self ._get_obs ()

        
        
        deviation =abs (current_request -total_ev_transport )
        if not is_final_step :
            self .metrics ['station_charge_limit_hits']+=int (station_charge_limit_hits_tensor .sum ().item ())
            self .metrics ['station_discharge_limit_hits']+=int (station_discharge_limit_hits_tensor .sum ().item ())
            self .metrics ['station_limit_hits']+=int (
            (station_charge_limit_hits_tensor |station_discharge_limit_hits_tensor ).sum ().item ()
            )
            self .metrics ['station_limit_steps']+=int (
            (station_charge_limit_hits_tensor |station_discharge_limit_hits_tensor ).any ().item ()
            )
            self .metrics ['station_limit_penalty_total']+=float (station_limit_penalties_tensor .sum ().item ())
            self .metrics ['total_steps']+=1 
            if current_request >0 :
                self .metrics ['surplus_steps']+=1 
            elif current_request <0 :
                self .metrics ['shortage_steps']+=1 

                
            if current_request >0 :
            
                if deviation <=self .tol_narrow_metrics :
                    self .metrics ['surplus_within_narrow']+=1 
            elif current_request <0 :
                if deviation <=self .tol_narrow_metrics :
                    self .metrics ['shortage_within_narrow']+=1 
            else :
                if abs (total_ev_transport )<=self .tol_narrow_metrics :
                    self .metrics ['zero_request_within_narrow']+=1 
                self .metrics ['zero_request_steps']+=1 


                
        deviation =abs (current_request -total_ev_transport )
        global_reward =self ._calculate_balance_reward (deviation )if not is_final_step else 0.0 
        
        balance_reward_only =float (global_reward )

        
        
        all_departing_data ={
        'station_ids':[],
        'slot_indices':[],
        'ev_ids':[],
        'final_socs':[],
        'target_socs':[]
        }

        
        for st in range (self .num_stations ):
        
        
        
            if not is_final_step :
                departing_mask =(self .depart [st ]==self .step_count )&self .ev_mask [st ]
            else :
                departing_mask =torch .zeros_like (self .ev_mask [st ],dtype =torch .bool )

            departing_indices =torch .nonzero (departing_mask ,as_tuple =False ).squeeze (-1 )

            if len (departing_indices )>0 :
            
                all_departing_data ['station_ids'].extend ([st ]*len (departing_indices ))
                all_departing_data ['slot_indices'].extend (departing_indices .tolist ())
                all_departing_data ['ev_ids'].append (self .ev_ids [st ,departing_indices ])
                all_departing_data ['final_socs'].append (self .soc [st ,departing_indices ])
                all_departing_data ['target_socs'].append (self .target [st ,departing_indices ])

                
                
                self .metrics ['total_switches_departed']+=int (self .switch_count [st ,departing_indices ].sum ().item ())

                
                self .ev_mask [st ,departing_indices ]=False 
                self .ev_ids [st ,departing_indices ]=0 
                self .soc [st ,departing_indices ]=0.0 
                self .prev_soc [st ,departing_indices ]=0.0 
                self .target [st ,departing_indices ]=0.0 
                self .depart [st ,departing_indices ]=0 
                self .initial_remaining [st ,departing_indices ]=0.0 
                self .arrival_step [st ,departing_indices ]=0.0 
                self .switch_count [st ,departing_indices ]=0 
                self .last_non_zero_state [st ,departing_indices ]=0 

                
        departure_reward_sum_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        if all_departing_data ['ev_ids']and not is_final_step :
        
            departing_ev_ids =torch .cat (all_departing_data ['ev_ids'])
            departing_final_socs =torch .cat (all_departing_data ['final_socs'])
            departing_target_socs =torch .cat (all_departing_data ['target_socs'])

            
            soc_achieved =departing_final_socs >=departing_target_socs 
            soc_achieved_metric =departing_final_socs >=departing_target_socs 
            soc_diff =departing_target_socs -departing_final_socs 

            
            soc_deficit =torch .clamp (soc_diff ,min =0.0 )
            self .metrics ['total_soc_deficit']+=float (soc_deficit .sum ().item ())

            
            
            from Config import LOCAL_R_SOC_HIT 
            soc_reward_value =LOCAL_R_SOC_HIT 

            
            
            
            deviation_tensor =soc_diff 

            
            reward_mask_achieved =deviation_tensor <=0 
            reward_mask_linear =(deviation_tensor >0 )&(deviation_tensor <=self .soc_wide )
            reward_mask_penalty =deviation_tensor >self .soc_wide 

            departure_rewards =torch .zeros_like (departing_final_socs )
            departure_rewards [reward_mask_achieved ]=soc_reward_value 
            departure_rewards [reward_mask_linear ]=soc_reward_value *(1 -deviation_tensor [reward_mask_linear ]/self .soc_wide )
            excess_deviation =deviation_tensor [reward_mask_penalty ]-self .soc_wide 
            decay_rate =soc_reward_value /self .soc_wide *2 
            departure_rewards [reward_mask_penalty ]=-decay_rate *excess_deviation 

            
            from Config import SOC_HIT_BONUS 
            if SOC_HIT_BONUS >0 :
                departure_rewards =departure_rewards +SOC_HIT_BONUS *reward_mask_achieved .float ()

                
            station_ids_cpu =all_departing_data ['station_ids']
            ev_ids_cpu =departing_ev_ids .cpu ().tolist ()
            final_socs_cpu =departing_final_socs .cpu ().tolist ()
            target_socs_cpu =departing_target_socs .cpu ().tolist ()
            soc_achieved_cpu =soc_achieved .cpu ().tolist ()
            soc_achieved_metric_cpu =soc_achieved_metric .cpu ().tolist ()
            departure_rewards_cpu =departure_rewards .cpu ().tolist ()

            
            departing_ev_stats [0 ]=len (ev_ids_cpu )
            departing_ev_stats [1 ]=sum (soc_achieved_metric_cpu )

            
            departure_reward_sum_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
            station_ids_tensor =torch .tensor (station_ids_cpu ,dtype =torch .long ,device =device )
            for st_idx in range (self .num_stations ):
                mask =station_ids_tensor ==st_idx 
                if mask .any ():
                    departure_reward_sum_tensor [st_idx ]=departure_rewards [mask ].sum ()
                    local_rewards_tensor [st_idx ]+=departure_reward_sum_tensor [st_idx ]

                    
            departure_reward_sum =departure_reward_sum_tensor .cpu ().tolist ()

            
            for i ,(st ,ev_id ,final_soc ,target_soc )in enumerate (zip (
            station_ids_cpu ,ev_ids_cpu ,final_socs_cpu ,target_socs_cpu 
            )):

            
                self .used_ev_ids .discard (ev_id )
                
                ev_dict_idx =next ((j for j ,ev in enumerate (self .stations_evs [st ])if ev ['id']==ev_id ),-1 )
                if ev_dict_idx >=0 :
                    self .stations_evs [st ].pop (ev_dict_idx )

                    
        self .metrics ['departing_evs']+=departing_ev_stats [0 ]
        self .metrics ['departing_evs_soc_met']+=departing_ev_stats [1 ]

        
        snapshot_end ={}
        if self .record_snapshots :
            for st in range (self .num_stations ):
                details_end =[]
                active_evs =torch .nonzero (self .ev_mask [st ],as_tuple =False ).squeeze (-1 )
                if len (active_evs )>0 :
                    sorted_evs =self ._sort_active_evs (st ,active_evs )
                    ids =self .ev_ids [st ,sorted_evs ]
                    socs =self .soc [st ,sorted_evs ]
                    
                    remains =(self .depart [st ,sorted_evs ]-self .step_count ).float ()
                    
                    needs =self .target [st ,sorted_evs ]-socs 
                    for i in range (len (sorted_evs )):
                        details_end .append ({
                        'id':int (ids [i ].item ()),
                        'soc':float (socs [i ].item ()),
                        'needed_soc':float (needs [i ].item ()),
                        'remaining_time':float (remains [i ].item ()),
                        'target_soc':float (self .target [st ,sorted_evs ][i ].item ()),
                        'switch_count':int (self .switch_count [st ,sorted_evs ][i ].item ()),
                        })
                snapshot_end [st ]=details_end 

                
                
        done =[self .step_count >=self .episode_steps ]*self .num_stations 

        
        
        signed_deviation =float (total_ev_transport -current_request )

        
        _active_ev_counts =self .ev_mask .sum (dim =1 ).cpu ().tolist ()
        _station_powers_list =station_powers_tensor .cpu ().tolist ()
        _soc_violations_list =soc_violations_tensor .cpu ().tolist ()
        _local_rewards_list =local_rewards_tensor .cpu ().tolist ()
        _progress_shaping_list =progress_shaping_rewards_tensor .cpu ().tolist ()
        _discharge_penalties_list =discharge_penalties_tensor .cpu ().tolist ()
        _switch_penalties_list =switch_penalties_tensor .cpu ().tolist ()
        _departure_reward_list =departure_reward_sum_tensor .cpu ().tolist ()
        _station_charge_limit_hits =station_charge_limit_hits_tensor .cpu ().tolist ()
        _station_discharge_limit_hits =station_discharge_limit_hits_tensor .cpu ().tolist ()
        _station_power_limits_kw =self .station_power_limit_kw .cpu ().tolist ()
        _station_limit_penalties_list =station_limit_penalties_tensor .cpu ().tolist ()

        info ={
        'net_demand':current_request ,
        'station_powers':_station_powers_list ,
        'station_power_limits_kw':_station_power_limits_kw ,
        'station_limit_penalties':_station_limit_penalties_list ,
        'station_charge_limited':_station_charge_limit_hits ,
        'station_discharge_limited':_station_discharge_limit_hits ,
        'total_ev_transport':total_ev_transport ,
        'soc_violations':_soc_violations_list ,
        'active_evs':{i :_active_ev_counts [i ]for i in range (self .num_stations )},
        'actual_ev_power_kw':actual_ev_power_kw_tensor ,
        'snapshot_pre':getattr (self ,'_snapshot_pre',{}),
        'snapshot_after':snapshot_after ,
        'snapshot_end':snapshot_end ,
        'arrivals_this_step':int (self .arrivals_this_step ),
        'arrivals_by_station':[int (x )for x in self .arrivals_by_station ],
        'step_count':int (self .step_count ),
        'local_rewards':_local_rewards_list ,
        'global_reward':float (global_reward ),
        'reward_breakdown':{
        'global':{
        'balance_reward':float (balance_reward_only ),
        'global_total':float (global_reward ),
        'deviation':float (signed_deviation ),
        'abs_deviation':float (deviation ),
        'net_demand':float (current_request ),
        'total_ev_transport':float (total_ev_transport ),
        },
        'per_station':[
        {
        'station_power':_station_powers_list [st ],
        'progress_shaping':_progress_shaping_list [st ],
        'discharge_penalty':_discharge_penalties_list [st ],
        'switch_penalty':_switch_penalties_list [st ],
        'station_limit_penalty':_station_limit_penalties_list [st ],
        'departure_reward':_departure_reward_list [st ],
        'local_total':_local_rewards_list [st ],
        }
        for st in range (self .num_stations )
        ]
        },
        }

        
        try :
            L =int (GLOBAL_DEMAND_STEPS )
        except Exception :
            L =0 
            
        start_idx =self .step_count -1 
        indices =torch .arange (start_idx ,start_idx +L ,dtype =torch .long ,device =device )
        valid_mask =(indices >=0 )&(indices <len (self .net_demand_series ))
        ag_tensor =torch .zeros (L ,dtype =torch .float32 ,device =device )
        if valid_mask .any ():
            ag_tensor [valid_mask ]=self ._clip_demand_for_obs (self .net_demand_series [indices [valid_mask ]])
        info ['ag_lookahead']=ag_tensor .cpu ().tolist ()

        
        if all_departing_data ['ev_ids']:
            station_ids_cpu =all_departing_data ['station_ids']
            slot_indices_cpu =all_departing_data ['slot_indices']
            
            if 'ev_ids_cpu'in locals ():
                info ['departed_evs']=[
                {'id':int (eid ),'station':int (st ),'slot':int (si )}
                for st ,si ,eid in zip (station_ids_cpu ,slot_indices_cpu ,ev_ids_cpu )
                ]
            else :
            
                info ['departed_evs']=[
                {'station':int (st ),'slot':int (si )}
                for st ,si in zip (station_ids_cpu ,slot_indices_cpu )
                ]
        else :
            info ['departed_evs']=[]

            
        observation =self ._get_obs ()

        

        
        local_rewards =local_rewards_tensor .cpu ().tolist ()
        return observation ,local_rewards ,global_reward ,done ,info 

        
        
        

    def _spawn_ev (self ,station :int ):
        "Documentation."
        
        empty_slots =torch .nonzero (~self .ev_mask [station ],as_tuple =False ).squeeze (-1 )
        if len (empty_slots )==0 :
            return False 

            
        slot_idx =int (empty_slots [random .randint (0 ,len (empty_slots )-1 )])

        init_soc ,target_soc ,dwell_steps ,needed_soc ,profile_ev_id =self ._sample_ev_profile ()

        if (
        profile_ev_id is not None 
        and 0 <=profile_ev_id <self .num_evs 
        and profile_ev_id not in self .used_ev_ids 
        ):
            ev_id =profile_ev_id 
        else :
            valid_ids =list (set (range (self .num_evs ))-self .used_ev_ids )
            if not valid_ids :
            
                return False 
            ev_id =random .choice (valid_ids )

        self .used_ev_ids .add (ev_id )

        dep =self .step_count +dwell_steps -1 
        profile_remaining =max (dep -self .step_count ,0 )

        self ._record_arrival (init_soc ,needed_soc ,dwell_steps )
        self ._set_ev_slot (
        station ,slot_idx ,ev_id ,init_soc ,target_soc ,dep ,
        profile_remaining =profile_remaining 
        )
        return True 


    def _calculate_balance_reward (self ,deviation :float )->float :
        "Documentation."
        balance_reward_value =getattr (self ,'balance_reward',GLOBAL_BALANCE_REWARD )
        D =max (float (self .tol_narrow_metrics ),1e-6 )
        d =float (deviation )
        eps =0.1 

        if d <=D :
            return balance_reward_value *(1.0 -eps *(d /D )**2 )
        else :
            return balance_reward_value *(1.0 -eps )-(2.0 *balance_reward_value /D )*(d -D )


    def _calculate_soc_reward (self ,deviation :float )->float :
        "Documentation."
        
        from Config import LOCAL_R_SOC_HIT ,SOC_HIT_BONUS 
        soc_reward_value =LOCAL_R_SOC_HIT 

        if deviation <=0 :
        
            return soc_reward_value +SOC_HIT_BONUS 

        elif deviation <=self .soc_wide :
        
            rate =1 -deviation /self .soc_wide 
            return rate *soc_reward_value 

        else :
        
        
            excess_deviation =deviation -self .soc_wide 
            
            
            decay_rate =soc_reward_value /self .soc_wide *2 
            return -decay_rate *excess_deviation 

            

    def get_metrics (self ):
        "Documentation."
        metrics ={}

        
        if self .metrics ['departing_evs']>0 :
            metrics ['soc_miss_rate']=100.0 -(self .metrics ['departing_evs_soc_met']/self .metrics ['departing_evs']*100.0 )
            
            metrics ['avg_switches']=self .metrics ['total_switches_departed']/self .metrics ['departing_evs']
            
            metrics ['avg_soc_deficit']=self .metrics ['total_soc_deficit']/self .metrics ['departing_evs']
        else :
            metrics ['soc_miss_rate']=0.0 
            metrics ['avg_switches']=0.0 
            metrics ['avg_soc_deficit']=0.0 

            
        if self .metrics ['surplus_steps']>0 :
            metrics ['surplus_absorption_rate']=self .metrics ['surplus_within_narrow']/self .metrics ['surplus_steps']*100.0 
        else :
            metrics ['surplus_absorption_rate']=0.0 

            
        if self .metrics ['shortage_steps']>0 :
            metrics ['supply_cooperation_rate']=self .metrics ['shortage_within_narrow']/self .metrics ['shortage_steps']*100.0 
        else :
            metrics ['supply_cooperation_rate']=0.0 

            
        if self .metrics ['zero_request_steps']>0 :
            metrics ['zero_request_maintenance_rate']=self .metrics ['zero_request_within_narrow']/self .metrics ['zero_request_steps']*100.0 
        else :
            metrics ['zero_request_maintenance_rate']=0.0 

            
        metrics ['total_switches']=self .metrics ['total_switches_current']
        metrics ['surplus_steps']=self .metrics ['surplus_steps']
        metrics ['surplus_within_narrow']=self .metrics ['surplus_within_narrow']
        metrics ['shortage_steps']=self .metrics ['shortage_steps']
        metrics ['shortage_within_narrow']=self .metrics ['shortage_within_narrow']
        metrics ['zero_request_steps']=self .metrics ['zero_request_steps']
        metrics ['zero_request_within_narrow']=self .metrics ['zero_request_within_narrow']
        metrics ['total_steps']=self .metrics ['total_steps']
        metrics ['station_limit_hits']=self .metrics ['station_limit_hits']
        metrics ['station_limit_steps']=self .metrics ['station_limit_steps']
        metrics ['station_charge_limit_hits']=self .metrics ['station_charge_limit_hits']
        metrics ['station_discharge_limit_hits']=self .metrics ['station_discharge_limit_hits']
        metrics ['station_limit_penalty_total']=self .metrics ['station_limit_penalty_total']
        metrics ['departing_evs']=self .metrics ['departing_evs']
        metrics ['departing_evs_soc_met']=self .metrics ['departing_evs_soc_met']

        return metrics 

    def reset_metrics (self ):
        "Documentation."
        self .metrics ={
        'total_steps':0 ,
        'surplus_within_narrow':0 ,
        'shortage_within_narrow':0 ,
        'surplus_steps':0 ,
        'shortage_steps':0 ,
        'zero_request_within_narrow':0 ,
        'zero_request_steps':0 ,
        'station_limit_hits':0 ,
        'station_limit_steps':0 ,
        'station_charge_limit_hits':0 ,
        'station_discharge_limit_hits':0 ,
        'station_limit_penalty_total':0.0 ,
        'departing_evs':0 ,
        'departing_evs_soc_met':0 ,
        'total_switches_departed':0 ,
        'total_switches_current':0 ,
        'total_soc_deficit':0.0 
        }

        
    def _sort_active_evs (self ,station :int ,active_evs :torch .Tensor )->torch .Tensor :
        "Documentation."
        if len (active_evs )==0 :
            return active_evs 

            
        depart_times =self .depart [station ,active_evs ]
        sorted_indices =torch .argsort (depart_times )
        return active_evs [sorted_indices ]

