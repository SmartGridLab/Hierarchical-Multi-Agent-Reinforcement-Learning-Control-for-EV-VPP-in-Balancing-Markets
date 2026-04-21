

"""
可視化・グラフ描画ユーティリティモジュール。

学習曲線・性能メトリクス・EV SoC推移・到着数など各種グラフを生成しPNG/CSVで保存する。
TensorBoardへのQ値・勾配ノルム記録クラス (GradientLossVisualizer) も提供する。
"""

import os
import shutil
import csv
import warnings
import numpy as np
import matplotlib
matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
import matplotlib as mpl 


warnings .filterwarnings ("ignore",category =UserWarning ,module ="matplotlib")
warnings .filterwarnings ("ignore",message ="Glyph .* missing from font")
import logging 
logging .getLogger ('matplotlib.font_manager').setLevel (logging .ERROR )
logging .getLogger ('matplotlib.ticker').setLevel (logging .ERROR )

from torch .utils .tensorboard import SummaryWriter 
from Config import (
TOL_NARROW_METRICS ,
POWER_TO_ENERGY ,
INITIAL_EVS_PER_STATION ,
NUM_STATIONS ,
)


mpl .rcParams ['font.family']='sans-serif'
mpl .rcParams ['font.sans-serif']=['Arial','Helvetica','Liberation Sans','FreeSans','sans-serif']
mpl .rcParams ['font.size']=30 
mpl .rcParams ['axes.titlesize']=36 
mpl .rcParams ['axes.labelsize']=30 
mpl .rcParams ['xtick.labelsize']=24 
mpl .rcParams ['ytick.labelsize']=24 
mpl .rcParams ['legend.fontsize']=24 
mpl .rcParams ['lines.linewidth']=3.0 
mpl .rcParams ['axes.linewidth']=3.0 
mpl .rcParams ['grid.linewidth']=1.5 
mpl .rcParams ['path.simplify']=True 
mpl .rcParams ['path.simplify_threshold']=0.5 
mpl .rcParams ['agg.path.chunksize']=10000 





def plot_daily_rewards (local_rewards ,
global_rewards ,
results_dir ,
episode_num =None ,
performance_metrics =None ,
title_prefix :str ="",
skip_png :bool =False ,
x_values =None ):
    "Documentation."

    base_results_dir =os .path .dirname (results_dir )if "TEST"in os .path .basename (results_dir )else results_dir
    os .makedirs (base_results_dir ,exist_ok =True )
    os .makedirs (results_dir ,exist_ok =True )


    rewards_filename ="episode_rewards_all.png"
    if title_prefix and ("test" in title_prefix .lower ()):
        rewards_filename ="test_episode_rewards_all.png"


    n_local =len (local_rewards )if local_rewards else 0
    n_global =len (global_rewards )if global_rewards else 0
    if x_values is not None :
        x_ep =list (x_values )
    else :
        x_ep =list (range (1 ,max (n_local ,n_global ,1 )+1 ))

    if not skip_png :

        fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(14 ,12 ))


        if local_rewards and n_local >0 :
            ax1 .plot (x_ep [:n_local ],local_rewards ,label ="Local Reward",linewidth =6 ,color ='green',alpha =0.8 )


            win =20
            if n_local >=win :
                local_mov =[sum (local_rewards [i :i +win ])/win for i in range (n_local -win +1 )]
                ax1 .plot (x_ep [win -1 :n_local ],local_mov ,
                color ='green',label =f"Local MA ({win}ep)",
                linewidth =3 ,linestyle ='--',alpha =0.6 )

        ax1 .axhline (0 ,color ="k",alpha =.3 ,linewidth =3 )
        ax1 .set_xlabel ("Episode",fontsize =24 )
        ax1 .set_ylabel ("Local Reward (Station Avg)",fontsize =24 )

        ax1 .grid (alpha =.3 ,linewidth =3 )
        ax1 .legend (fontsize =18 )


        if global_rewards and n_global >0 :
            ax2 .plot (x_ep [:n_global ],global_rewards ,label ="Global Reward",linewidth =6 ,color ='orange',alpha =0.8 )


            win =20
            if n_global >=win :
                global_mov =[sum (global_rewards [i :i +win ])/win for i in range (n_global -win +1 )]
                ax2 .plot (x_ep [win -1 :n_global ],global_mov ,
                color ='orange',label =f"Global MA ({win}ep)",
                linewidth =3 ,linestyle ='--',alpha =0.6 )

        ax2 .axhline (0 ,color ="k",alpha =.3 ,linewidth =3 )
        ax2 .set_xlabel ("Episode",fontsize =24 )
        ax2 .set_ylabel ("Global Reward (Total)",fontsize =24 )

        ax2 .grid (alpha =.3 ,linewidth =3 )
        ax2 .legend (fontsize =18 )

        fig .tight_layout ()

        fig .savefig (os .path .join (base_results_dir ,rewards_filename ),dpi =140 ,bbox_inches ='tight')
        plt .close (fig )

    csv_filename =rewards_filename .replace ('.png','.csv')
    csv_path =os .path .join (base_results_dir ,csv_filename )
    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        writer .writerow (['Episode','Local_Reward','Global_Reward'])
        for i in range (n_local ):
            writer .writerow ([x_ep [i ],local_rewards [i ],global_rewards [i ]])

            


            
            
            
def plot_performance_metrics (performance_metrics ,results_dir ,title_prefix :str ="",x_values =None ,skip_png :bool =False ):
    "Documentation."
    if not performance_metrics or len (performance_metrics .get ('soc_miss_count',[]))==0 :
        return 

        
    base_results_dir =os .path .dirname (results_dir )if "TEST"in os .path .basename (results_dir )else results_dir 
    os .makedirs (base_results_dir ,exist_ok =True )

    
    soc_miss_rates =performance_metrics .get ('soc_miss_count',[])
    avg_switches_list =performance_metrics .get ('avg_switches',[])

    
    surplus_steps_list =performance_metrics .get ('surplus_steps',[])
    surplus_within_list =performance_metrics .get ('surplus_within_narrow',[])
    shortage_steps_list =performance_metrics .get ('shortage_steps',[])
    shortage_within_list =performance_metrics .get ('shortage_within_narrow',[])

    
    soc_hit_rates =[100 -rate for rate in soc_miss_rates ]

    
    dispatch_tracking_rates =[]
    for i in range (len (soc_hit_rates )):
        s_steps =surplus_steps_list [i ]if i <len (surplus_steps_list )else 0 
        s_within =surplus_within_list [i ]if i <len (surplus_within_list )else 0 
        sh_steps =shortage_steps_list [i ]if i <len (shortage_steps_list )else 0 
        sh_within =shortage_within_list [i ]if i <len (shortage_within_list )else 0 

        denom =s_steps +sh_steps 
        numer =s_within +sh_within 
        rate =(numer /denom *100.0 )if denom >0 else 0.0 
        dispatch_tracking_rates .append (rate )

        
    if x_values is not None :
        x_plot =np .asarray (x_values )
    else :
        episodes =np .arange (1 ,len (soc_hit_rates )+1 )
        
        x_plot =episodes 

        
    metrics_filename ="train_performance_metrics.png"
    if title_prefix and ("test" in title_prefix .lower ()):
        metrics_filename ="test_performance_metrics.png"

    if not skip_png :
    
        plt .rcParams .update ({
        "font.size":24 ,
        "axes.labelsize":26 ,
        "axes.titlesize":28 ,
        "legend.fontsize":20 ,
        "xtick.labelsize":20 ,
        "ytick.labelsize":20 ,
        })

        soc_color ="tab:blue"
        compliance_color ="tab:red"

        fig ,ax1 =plt .subplots (figsize =(12 ,7 ))

        
        soc_line ,=ax1 .plot (x_plot ,soc_hit_rates ,linestyle ="-",linewidth =5 ,
        label ="Target SoC Satisfaction Rate",color =soc_color )
        ax1 .set_xlabel ("Episode")
        ax1 .set_ylabel ("Target SoC Satisfaction Rate (%)",color =soc_color )
        ax1 .tick_params (axis ="y",labelcolor =soc_color )
        ax1 .set_ylim (0 ,105 )
        ax1 .spines ["left"].set_color (soc_color )

        
        ax2 =ax1 .twinx ()
        dt_line ,=ax2 .plot (x_plot ,dispatch_tracking_rates ,linestyle ="-",linewidth =5 ,
        label ="Dispatch Tracking Rate",color =compliance_color )
        ax2 .set_ylabel ("Dispatch Tracking Rate (%)",color =compliance_color )
        ax2 .tick_params (axis ="y",labelcolor =compliance_color )
        ax2 .set_ylim (0 ,105 )
        ax2 .spines ["right"].set_color (compliance_color )

        
        lines =[soc_line ,dt_line ]
        labels =["Target SoC Satisfaction Rate","Dispatch Tracking Rate"]

        if len (avg_switches_list )>0 :
            ax3 =ax1 .twinx ()
            
            ax3 .spines ["right"].set_position (("axes",1.15 ))
            
            switch_line ,=ax3 .plot (x_plot ,avg_switches_list ,marker ='o',markersize =15 ,linestyle ='None',
            label ="Avg Switches",color ="green")
            ax3 .set_ylabel ("Avg Switches",color ="green")
            ax3 .tick_params (axis ="y",labelcolor ="green")
            
            max_sw =max (max (avg_switches_list ),10 )
            ax3 .set_ylim (0 ,max_sw +2 )
            ax3 .spines ["right"].set_color ("green")
            lines .append (switch_line )
            labels .append ("Avg Switches")

            
            
            
        ax1 .grid (True ,linewidth =0.7 ,alpha =0.7 )

        
        if len (x_plot )>10 :
            step =max (1 ,len (x_plot )//10 )
            tick_idx =np .arange (0 ,len (x_plot ),step )
            ax1 .set_xticks (x_plot [tick_idx ])
        else :
            ax1 .set_xticks (x_plot )

        fig .tight_layout ()

        
        fig .savefig (os .path .join (base_results_dir ,metrics_filename ),dpi =140 ,bbox_inches ='tight')
        plt .close (fig )

        
        leg_fig ,leg_ax =plt .subplots (figsize =(8 ,3 ))
        leg_ax .axis ("off")
        leg_ax .legend (lines ,labels ,loc ="center",frameon =True )
        leg_fig .tight_layout ()
        legend_filename =metrics_filename .replace (".png","_legend.png")
        leg_fig .savefig (os .path .join (base_results_dir ,legend_filename ),dpi =140 )
        plt .close (leg_fig )

        
    csv_filename =metrics_filename .replace ('.png','.csv')
    csv_path =os .path .join (base_results_dir ,csv_filename )

    
    departing_evs =performance_metrics .get ('departing_evs',[])
    departing_evs_soc_met =performance_metrics .get ('departing_evs_soc_met',[])
    avg_soc_deficit_list =performance_metrics .get ('avg_soc_deficit',[])
    station_limit_hits_list =performance_metrics .get ('station_limit_hits',[])
    station_limit_steps_list =performance_metrics .get ('station_limit_steps',[])
    station_charge_limit_hits_list =performance_metrics .get ('station_charge_limit_hits',[])
    station_discharge_limit_hits_list =performance_metrics .get ('station_discharge_limit_hits',[])
    station_limit_penalty_total_list =performance_metrics .get ('station_limit_penalty_total',[])
    station_limit_penalty_per_step_list =performance_metrics .get ('station_limit_penalty_per_step',[])
    station_limit_penalty_per_hit_list =performance_metrics .get ('station_limit_penalty_per_hit',[])
    has_switch_cols =len (avg_switches_list )>0 
    has_stlimit_cols =(
    len (station_limit_hits_list )>0 or
    len (station_limit_steps_list )>0 or
    len (station_charge_limit_hits_list )>0 or
    len (station_discharge_limit_hits_list )>0 or
    len (station_limit_penalty_total_list )>0 or
    len (station_limit_penalty_per_step_list )>0 or
    len (station_limit_penalty_per_hit_list )>0
    )

    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        header =[
        'Episode',
        'SoC_Hit_Rate_%',
        'Dispatch_Tracking_Rate_%',
        'Avg_SoC_Deficit_kWh',
        ]
        if has_switch_cols :
            header .append ('Avg_Switches')
        if has_stlimit_cols :
            header .extend ([
            'Station_Limit_Hits',
            'Station_Limit_Steps',
            'Station_Charge_Limit_Hits',
            'Station_Discharge_Limit_Hits',
            'Station_Limit_Penalty_Total',
            'Station_Limit_Penalty_Per_Step',
            'Station_Limit_Penalty_Per_Hit',
            ])
        header .extend ([
        'SoC_Departing_EVs',
        'SoC_Hit_EVs',
        'Surplus_Steps',
        'Surplus_Steps_Within_Narrow',
        'Shortage_Steps',
        'Shortage_Steps_Within_Narrow',
        ])
        writer .writerow (header )

        def _safe_get (seq ,idx ):
            return seq [idx ]if (isinstance (seq ,(list ,tuple ,np .ndarray ))and idx <len (seq ))else ""

        for i in range (len (soc_hit_rates )):
            row =[
            i +1 ,
            soc_hit_rates [i ],
            dispatch_tracking_rates [i ],
            _safe_get (avg_soc_deficit_list ,i ),
            ]
            if has_switch_cols :
                row .append (_safe_get (avg_switches_list ,i ))
            if has_stlimit_cols :
                row .extend ([
                _safe_get (station_limit_hits_list ,i ),
                _safe_get (station_limit_steps_list ,i ),
                _safe_get (station_charge_limit_hits_list ,i ),
                _safe_get (station_discharge_limit_hits_list ,i ),
                _safe_get (station_limit_penalty_total_list ,i ),
                _safe_get (station_limit_penalty_per_step_list ,i ),
                _safe_get (station_limit_penalty_per_hit_list ,i ),
                ])
            row .extend ([
            _safe_get (departing_evs ,i ),
            _safe_get (departing_evs_soc_met ,i ),
            _safe_get (surplus_steps_list ,i ),
            _safe_get (surplus_within_list ,i ),
            _safe_get (shortage_steps_list ,i ),
            _safe_get (shortage_within_list ,i ),
            ])
            writer .writerow (row )


            
            
            
def plot_station_cooperation_full (all_episode_data :dict ,
results_dir :str ,
random_window :bool =False ,
title_prefix :str ="")->None :
    "Documentation."
    import random 

    
    title_prefix_en =title_prefix

    
    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )>10 :
        episode_keys =episode_keys [-10 :]

    for ep_key in episode_keys :
        ep =all_episode_data [ep_key ]
        total_steps =len (ep ["ag_requests"])

        
        start =0 
        end =total_steps 
        rng =np .arange (end -start )

        
        num_stations =0 
        while f"actual_ev{num_stations+1}"in ep :
            num_stations +=1 
        if num_stations ==0 :
            raise ValueError ("Visualization requires actual_ev* series. Found none in episode data.")

            
        stations =[]
        for i in range (1 ,num_stations +1 ):
            key =f"actual_ev{i}"
            if key not in ep :
                raise KeyError (f"Missing required series '{key}' in episode data")
            arr =np .asarray (ep [key ][start :end ],dtype =float )
            stations .append (arr )

        ag_req =np .asarray (ep ["ag_requests"][start :end ],dtype =float )

        
        if "total_ev_transport"not in ep :
            raise KeyError ("Visualization requires 'total_ev_transport' series. Found none in episode data.")
        total_station_power =np .asarray (ep ["total_ev_transport"][start :end ],dtype =float )

        

        
        tol_narrow =TOL_NARROW_METRICS 

        
        
        narrow_success_count =np .sum (np .abs (ag_req -total_station_power )<=tol_narrow )
        
        narrow_success_rate =narrow_success_count /len (rng )*100.0 

        
        fig ,ax =plt .subplots (figsize =(20 ,12 ))
        bottoms =np .zeros_like (rng ,dtype =float )
        width =.9 

        labels =[f"Station {i}"for i in range (1 ,num_stations +1 )]

        
        colors =plt .cm .tab20 (np .linspace (0 ,1 ,num_stations ))
        if num_stations >20 :
        
            cmap1 =plt .cm .tab20 (np .linspace (0 ,1 ,20 ))
            cmap2 =plt .cm .tab20b (np .linspace (0 ,1 ,20 ))
            cmap3 =plt .cm .tab20c (np .linspace (0 ,1 ,20 ))
            cmap4 =plt .cm .Set3 (np .linspace (0 ,1 ,20 ))
            cmap5 =plt .cm .Pastel1 (np .linspace (0 ,1 ,20 ))
            colors =np .vstack ([cmap1 ,cmap2 ,cmap3 ,cmap4 ,cmap5 ])[:num_stations ]

            
        pos_bottoms =np .zeros_like (rng ,dtype =float )
        neg_bottoms =np .zeros_like (rng ,dtype =float )
        for s ,lbl ,col in zip (stations ,labels ,colors ):
            s =np .asarray (s ,dtype =float )
            pos =np .clip (s ,0 ,None )
            neg =np .clip (s ,None ,0 )
            pos_drawn =np .any (pos !=0 )
            neg_drawn =np .any (neg !=0 )

            
            if pos_drawn :
                ax .bar (rng ,pos ,width ,bottom =pos_bottoms ,label =lbl ,color =col ,alpha =.7 )

                
            if neg_drawn :
                label_for_neg =lbl if not pos_drawn else "_nolegend_"
                ax .bar (rng ,neg ,width ,bottom =neg_bottoms ,label =label_for_neg ,color =col ,alpha =.7 )

                
            if not pos_drawn and not neg_drawn :
                ax .plot ([],[],color =col ,label =lbl ,linestyle ='-')

            pos_bottoms +=pos 
            neg_bottoms +=neg 


            
            
            
        ax .step (rng ,ag_req ,"k-",lw =6 ,where ='mid',label ="Grid Request",alpha =0.8 )

        
        ax .plot (rng ,total_station_power ,"m-",lw =9 ,label ="Total Station Power")

        
        
        for i ,step_idx in enumerate (rng ):
            step_req =ag_req [i ]
            step_upper =step_req +tol_narrow 
            step_lower =step_req -tol_narrow 

            
            ax .fill_between ([step_idx -0.4 ,step_idx +0.4 ],
            [step_lower ,step_lower ],
            [step_upper ,step_upper ],
            color ='gray',alpha =0.15 ,edgecolor ='none')

            
        import matplotlib .patches as mpatches 
        narrow_patch =mpatches .Patch (color ='gray',alpha =0.15 ,label ="Tolerance band")

        ax .axhline (0 ,color ="k",alpha =.3 ,linewidth =3 )
        ax .set_xticks (rng )
        
        ax .set_xticklabels ([start +i +1 for i in rng ])
        ax .set_xlabel ("Step")
        ax .set_ylabel ("Power ")
        
        
        
        ax .grid (alpha =.3 ,linewidth =3 )

        
        if num_stations <=10 :
        
            handles ,labels =ax .get_legend_handles_labels ()
            
            handles .append (narrow_patch )
            labels .append ("Tolerance band")
            ax .legend (handles =handles ,labels =labels ,loc ="upper left")
        else :
        
            handles ,labels =ax .get_legend_handles_labels ()
            
            non_station_items =[i for i ,lbl in enumerate (labels )if not lbl .startswith ("Station")]
            selected_items =non_station_items +list (range (min (3 ,num_stations )))

            
            selected_handles =[handles [i ]for i in selected_items ]
            selected_labels =[labels [i ]for i in selected_items ]
            selected_handles .append (narrow_patch )
            selected_labels .append ("Tolerance band")

            ax .legend (handles =selected_handles ,labels =selected_labels ,loc ="upper left")

            
        fig .tight_layout ()

        
        
        
        base_fname =f"zz_station_cooperation_full_episode_{ep_key}"+(
        "_random_window"if random_window else ""
        )
        fname =base_fname 
        if title_prefix :
            fname =f"{title_prefix_en.lower().replace(' ', '_')}_{base_fname}"
        fig .savefig (os .path .join (results_dir ,f"{fname}.png"))
        plt .close (fig )

        
        base_csv_fname =f"zz_station_cooperation_full_episode_{ep_key}"
        csv_fname =base_csv_fname 
        if title_prefix :
            csv_fname =f"{title_prefix_en.lower().replace(' ', '_')}_{base_csv_fname}"
        csv_filename =f"{csv_fname}.csv"
        csv_path =os .path .join (results_dir ,csv_filename )

        
        all_ag_req =np .asarray (ep ["ag_requests"],dtype =float )
        all_stations =[]
        for i in range (1 ,num_stations +1 ):
            key =f"actual_ev{i}"
            all_stations .append (np .asarray (ep [key ],dtype =float ))
        all_total_station_power =np .asarray (ep ["total_ev_transport"],dtype =float )

        with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
            writer =csv .writer (f )
            
            header =['Step','Grid_Request']+[f'Station_{i}'for i in range (1 ,num_stations +1 )]+['Total_Station_Power']
            writer .writerow (header )
            
            for i in range (total_steps ):
                row =[i +1 ,all_ag_req [i ]]
                for s in all_stations :
                    row .append (s [i ])
                row .append (all_total_station_power [i ])
                writer .writerow (row )


                
                
                
def plot_ev_detailed_soc (all_episode_data :dict ,
results_dir :str ,
display_steps :int =48 ,
random_window :bool =False ,
title_prefix :str ="")->None :
    "Documentation."
    import random 

    
    title_prefix_en =title_prefix

    
    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )>2 :
        episode_keys =episode_keys [-2 :]

    for ep_key in episode_keys :
        ep =all_episode_data [ep_key ]
        total_steps =len (ep ["ag_requests"])

        soc_data =ep .get ("soc_data",{})
        if not soc_data :
            continue 

            
        long_stay_evs =find_long_stay_evs (soc_data ,min_stay =10 ,max_evs =3 )

        for idx ,long_stay_ev in enumerate (long_stay_evs ):
        
            long_stay_fname =f"test_results_ev_soc_long_stay_{idx+1}_episode_{ep_key}"
            if title_prefix :
                prefix =title_prefix_en .lower ().replace (' ','_')
                if not long_stay_fname .startswith (f"{prefix}_"):
                    long_stay_fname =f"{prefix}_{long_stay_fname}"

            if long_stay_ev :
                station_id ,ev_id ,ev_data =long_stay_ev 

                
                csv_filename =f"{long_stay_fname}.csv"
                csv_path =os .path .join (results_dir ,csv_filename )
                with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
                    writer =csv .writer (f )
                    writer .writerow (['Step','EV_ID','Station','SoC_%','Target_%','Departure_Step'])
                    ts =np .asarray (ev_data ["times"])
                    soc_values =np .asarray (ev_data ["soc"])
                    
                    for j ,step in enumerate (ts ):
                        if j <len (soc_values ):
                            writer .writerow ([step ,ev_id ,station_id ,soc_values [j ],ev_data ['target'],ev_data .get ('depart','N/A')])

                            
                fig_long ,ax_long =plt .subplots (figsize =(14 ,8 ))
                soc_color ="tab:blue"
                dispatch_color ="tab:red"

                
                start =0 
                end =total_steps 
                ag_req =np .asarray (ep ["ag_requests"][start :end ],dtype =float )

                
                for i in range (len (ag_req )):
                    x =start +i 
                    if ag_req [i ]>0 :
                        ax_long .axvspan (x -0.5 ,x +0.5 ,color ='lightgreen',alpha =0.2 )
                    elif ag_req [i ]<0 :
                        ax_long .axvspan (x -0.5 ,x +0.5 ,color ='lightpink',alpha =0.2 )

                ts =np .asarray (ev_data ["times"],dtype =float )
                soc_values =np .asarray (ev_data ["soc"],dtype =float )
                mask =(ts >=start )&(ts <end )
                ts_plot =ts [mask ]
                soc_plot =soc_values [mask ]

                
                departure_time =ev_data .get ("depart_step",ev_data .get ("depart",None ))
                target_soc =float (ev_data .get ("target",np .nan ))
                target_achieved =True 
                depart_soc =None 
                if departure_time is not None :
                    dep_idx =np .where (ts ==float (departure_time ))[0 ]
                    if dep_idx .size >0 :
                        depart_soc =float (soc_values [int (dep_idx [0 ])])
                    elif soc_plot .size >0 :
                        depart_soc =float (soc_plot [-1 ])
                    if np .isfinite (target_soc )and depart_soc is not None :
                        target_achieved =(depart_soc >=target_soc )

                line_style ="-"if target_achieved else "--"
                ax_long .plot (
                ts_plot ,soc_plot ,
                color =soc_color ,lw =6 ,ls =line_style ,
                label =f"EV {ev_id} (Target {target_soc:.0f}%)"
                )

                
                if np .isfinite (target_soc ):
                    ax_long .axhline (target_soc ,color ="green",ls ="--",alpha =0.6 ,lw =4.5 ,label ="Target SoC")

                    
                if departure_time is not None and start <=float (departure_time )<end :
                    ax_long .axvline (float (departure_time ),color =soc_color ,ls ="--",alpha =0.8 ,lw =4.5 )
                    if depart_soc is not None :
                        ax_long .plot ([float (departure_time )],[depart_soc ],'o',color =soc_color ,markersize =8 *3 )
                        if np .isfinite (target_soc ):
                            ax_long .plot ([float (departure_time )],[target_soc ],'s',color ="green",markersize =6 *3 ,alpha =0.85 )
                            label_txt ="Goal met!"if depart_soc >=target_soc else "Goal not met"
                            label_col ="green"if depart_soc >=target_soc else "red"
                            ax_long .text (float (departure_time ),depart_soc +5.0 ,label_txt ,color =label_col ,ha ="center",fontsize =9 *3 )

                            
                ax_long2 =ax_long .twinx ()
                t_range =np .arange (start ,end )
                if ag_req .size >0 :
                    y_pos =np .where (ag_req >=0.0 ,ag_req ,np .nan )
                    y_neg =np .where (ag_req <0.0 ,ag_req ,np .nan )
                    ax_long2 .plot (t_range ,y_pos ,color =dispatch_color ,lw =3.2 ,alpha =0.95 ,label ="Dispatch (+)")
                    ax_long2 .plot (t_range ,y_neg ,color ="tab:blue",lw =3.2 ,alpha =0.95 ,label ="Dispatch (-)")
                    ax_long2 .fill_between (t_range ,0 ,ag_req ,where =(ag_req >0 ),color =dispatch_color ,alpha =0.15 )
                    ax_long2 .fill_between (t_range ,0 ,ag_req ,where =(ag_req <0 ),color ="tab:blue",alpha =0.15 )
                ax_long2 .set_ylabel ("Dispatch signal [kW]",fontsize =22 ,color =dispatch_color )
                ax_long2 .tick_params (axis ='y',labelcolor =dispatch_color ,labelsize =16 )
                ax_long2 .axhline (0 ,color =dispatch_color ,ls ="--",lw =1.2 ,alpha =0.45 )

                ax_long .set_title (
                f"Long Stay EV #{idx+1} - {station_id} / EV {ev_id}",
                fontsize =30 
                )
                ax_long .set_xlabel ("Time Step",fontsize =22 )
                ax_long .set_ylabel ("SoC [%]",fontsize =22 ,color =soc_color )
                ax_long .tick_params (axis ='both',labelsize =16 )
                ax_long .tick_params (axis ='y',labelcolor =soc_color )
                ax_long .set_xlim (start ,max (end -1 ,start +1 ))
                ax_long .set_ylim (0 ,105 )
                ax_long .grid (alpha =0.3 ,linewidth =2 )

                h1 ,l1 =ax_long .get_legend_handles_labels ()
                h2 ,l2 =ax_long2 .get_legend_handles_labels ()
                if h1 or h2 :
                    ax_long .legend (h1 +h2 ,l1 +l2 ,loc ="upper left",fontsize =14 )

                fig_long .tight_layout ()
                out_png =os .path .join (results_dir ,f"{long_stay_fname}.png")
                fig_long .savefig (out_png ,dpi =180 ,bbox_inches ='tight')
                plt .close (fig_long )

                
                
        max_stay_duration =0 
        if soc_data :
            for station_evs in soc_data .values ():
                for ev in station_evs .values ():
                    if "times"in ev and len (ev ["times"])>0 :
                        ev_start =min (ev ["times"])
                        ev_end =max (ev ["times"])
                        stay_duration =ev_end -ev_start 
                        max_stay_duration =max (max_stay_duration ,stay_duration )

                        
        fig_random ,ax_random =plt .subplots (figsize =(14 ,8 ))

        
        soc_color ="tab:blue"
        dispatch_color ="tab:red"
        ax_random2 =None 

        
        if soc_data :
            station_key =random .choice (list (soc_data .keys ()))
            evs =soc_data [station_key ]

            
            start =0 
            end =total_steps 

            
            valid_evs ={}
            for ev_id ,ev in evs .items ():
                ts =np .asarray (ev ["times"])
                
                if len (ts )>0 :
                    ev_start =min (ts )
                    ev_end =max (ts )
                    
                    if ev_start <=end and ev_end >=start :
                        valid_evs [ev_id ]=ev 

                        
            if not valid_evs :
                for st_key ,st_evs in soc_data .items ():
                    if st_evs :
                        station_key =st_key 
                        evs =st_evs 
                        
                        first_ev =list (st_evs .values ())[0 ]
                        first_ev_times =np .asarray (first_ev ["times"])
                        if len (first_ev_times )>0 :
                            start =max (0 ,int (first_ev_times [0 ]))
                            end =min (total_steps ,start +display_steps )
                            
                            valid_evs ={}
                            for ev_id ,ev in evs .items ():
                                ts =np .asarray (ev ["times"])
                                if len (ts )>0 :
                                    ev_start =min (ts )
                                    ev_end =max (ts )
                                    if ev_start <=end and ev_end >=start :
                                        valid_evs [ev_id ]=ev 
                            if valid_evs :
                                break 

                                
            ag_req =np .asarray (ep ["ag_requests"][start :end ],dtype =float )

            
            for i in range (len (ag_req )):
                if ag_req [i ]>0 :
                    ax_random .axvspan (start +i -0.5 ,start +i +0.5 ,color ='lightgreen',alpha =0.2 )
                elif ag_req [i ]<0 :
                    ax_random .axvspan (start +i -0.5 ,start +i +0.5 ,color ='lightpink',alpha =0.2 )

                    
            if valid_evs :
            
                ev_ids_to_show =list (valid_evs .keys ())
                if len (ev_ids_to_show )>5 :
                    random .shuffle (ev_ids_to_show )
                    ev_ids_to_show =ev_ids_to_show [:5 ]

                    
                colors =plt .cm .tab10 (np .linspace (0 ,1 ,len (ev_ids_to_show )))

                
                for i ,ev_id in enumerate (ev_ids_to_show ):
                    ev =valid_evs [ev_id ]
                    ts =np .asarray (ev ["times"])
                    soc_vals =np .asarray (ev ["soc"])
                    color =colors [i ]

                    
                    target_achieved =True 
                    if "soc"in ev and "depart"in ev and "target"in ev :
                        depart_idx =[j for j ,t in enumerate (ev ["times"])if t ==ev ["depart"]]
                        if depart_idx and ev ["soc"][depart_idx [0 ]]<ev ["target"]:
                            target_achieved =False 

                    line_style ="--"if not target_achieved else "-"
                    line_color =color 

                    
                    if len (ts )>0 and len (soc_vals )>0 :
                        ax_random .plot (ts ,soc_vals ,
                        label =f"EV {ev_id} (Target {ev['target']:.0f}%)",
                        color =line_color ,lw =6 ,ls =line_style )

                        
                    depart_time_rand =ev .get ("depart_step",ev .get ("depart",None ))
                    if depart_time_rand is not None :
                        ax_random .axvline (depart_time_rand ,color =color ,ls ="--",alpha =0.7 ,lw =4.5 )
                        
                        depart_idx =[j for j ,t in enumerate (ev ["times"])if t ==depart_time_rand ]
                        
                        depart_soc =None 
                        if "final_soc"in ev :
                        
                            depart_soc =ev ["final_soc"]
                            
                            ax_random .plot ([depart_time_rand ],[depart_soc ],'o',color =color ,markersize =8 *3 )
                            
                            if "target_soc"in ev :
                                ax_random .axhline (ev ["target_soc"],color ="green",ls ="--",alpha =0.3 ,lw =4.5 )
                                ax_random .plot ([depart_time_rand ],[ev ["target_soc"]],'s',color ="green",markersize =6 *3 ,alpha =0.7 )
                                
                                if depart_soc >=ev ["target_soc"]:
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal met!",color ="green",ha ="center",fontsize =9 *3 )
                                else :
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal not met",color ="red",ha ="center",fontsize =9 *3 )
                        elif depart_idx :
                        
                            depart_soc =ev ["soc"][depart_idx [0 ]]
                            ax_random .plot ([depart_time_rand ],[depart_soc ],'o',color =color ,markersize =6 *3 )
                            
                            if "target"in ev :
                                target_soc =ev ["target"]
                                ax_random .axhline (target_soc ,color ="green",ls ="--",alpha =0.3 ,lw =4.5 )
                                ax_random .plot ([depart_time_rand ],[target_soc ],'s',color ="green",markersize =6 *3 ,alpha =0.7 )
                                
                                if depart_soc >=target_soc :
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal met!",color ="green",ha ="center",fontsize =9 *3 )
                                else :
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal not met",color ="red",ha ="center",fontsize =9 *3 )
                                    
                        if depart_soc is not None and not any (ts ==depart_time_rand ):
                        
                            plot_ts =np .append (ts ,[depart_time_rand ])
                            plot_soc =np .append (soc_vals ,[depart_soc ])
                            
                            sort_idx =np .argsort (plot_ts )
                            ax_random .plot (plot_ts [sort_idx ],plot_soc [sort_idx ],color =line_color ,lw =6 ,ls =line_style ,alpha =0.7 )

                            
                ax_random2 =ax_random .twinx ()
                time_range =np .arange (start ,end )

                
                ax_random2 .plot (time_range ,ag_req ,"-",lw =3 ,label ="Dispatch signal value",color =dispatch_color )
                ax_random2 .set_ylabel ("Dispatch signal value [kW]",color =dispatch_color ,fontsize =30 )
                ax_random2 .tick_params (axis ="y",labelcolor =dispatch_color ,labelsize =24 )
                ax_random2 .spines ["right"].set_color (dispatch_color )

                
                ax_random .spines ["left"].set_color (soc_color )
                ax_random .spines ["bottom"].set_color ("black")
                ax_random .tick_params (axis ="x",colors ="black")
                ax_random .tick_params (axis ="y",labelcolor =soc_color )

                
                
                ax_random .set_xlim (start ,end -1 )
            else :
                ax_random .text (0.5 ,0.5 ,"No EVs connected in this time window",
                ha ='center',va ='center',transform =ax_random .transAxes ,fontsize =24 )
        else :
            ax_random .text (0.5 ,0.5 ,"No station data available",
            ha ='center',va ='center',transform =ax_random .transAxes ,fontsize =24 )

        ax_random .set_ylabel ("SoC [%]",color =soc_color )
        ax_random .set_ylim (0 ,105 )
        ax_random .grid (alpha =.3 ,linewidth =3 )
        
        lines1 ,labels1 =ax_random .get_legend_handles_labels ()
        if ax_random2 is not None :
            lines2 ,labels2 =ax_random2 .get_legend_handles_labels ()
            ax_random .legend (lines1 +lines2 ,labels1 +labels2 ,loc ="best")
        else :
            ax_random .legend (loc ="best")

            
        fig_random .tight_layout ()

        
        random_window_fname =f"ev_soc_random_window_episode_{ep_key}"
        if title_prefix :
            random_window_fname =f"{title_prefix_en.lower().replace(' ', '_')}_{random_window_fname}"
        fig_random .savefig (os .path .join (results_dir ,f"{random_window_fname}.png"))
        plt .close (fig_random )

        
        if soc_data and valid_evs :
            csv_filename =f"{random_window_fname}.csv"
            csv_path =os .path .join (results_dir ,csv_filename )
            with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
                writer =csv .writer (f )
                writer .writerow (['Step','EV_ID','Station','SoC_%','Target_%'])
                for ev_id in (ev_ids_to_show if 'ev_ids_to_show'in locals ()and ev_ids_to_show else valid_evs .keys ()):
                    ev =valid_evs [ev_id ]
                    ts =np .asarray (ev ["times"])
                    soc_vals =np .asarray (ev ["soc"])
                    
                    for j ,step in enumerate (ts ):
                        if j <len (soc_vals ):
                            writer .writerow ([step ,ev_id ,station_key ,soc_vals [j ],ev ['target']])

            
            
        "Info."

        
        
def find_long_stay_evs (soc_data ,min_stay =10 ,max_evs =3 ):
    "Documentation."
    long_stay_evs =[]
    for station_id ,evs in sorted (soc_data .items ()):
        for ev_id ,ev in evs .items ():
            if "times"in ev and "depart"in ev :
                arrival_time =min (ev ["times"])
                departure_time =ev ["depart"]
                stay_duration =departure_time -arrival_time 
                if stay_duration >=min_stay :
                    long_stay_evs .append ((station_id ,ev_id ,ev ))

                    
    return long_stay_evs [:max_evs ]if len (long_stay_evs )>max_evs else long_stay_evs 

    
def find_short_stay_evs (soc_data ,min_stay =2 ,max_stay =5 ,max_evs =5 ):
    "Documentation."
    short_stay_evs =[]
    for station_id ,evs in sorted (soc_data .items ()):
        for ev_id ,ev in evs .items ():
            if "times"in ev and "depart"in ev :
                arrival_time =min (ev ["times"])
                departure_time =ev ["depart"]
                stay_duration =departure_time -arrival_time 
                if min_stay <=stay_duration <=max_stay :
                    short_stay_evs .append ((station_id ,ev_id ,ev ))

                    
    import random 
    random .shuffle (short_stay_evs )
    return short_stay_evs [:max_evs ]if len (short_stay_evs )>max_evs else short_stay_evs 

    
from torch .utils .tensorboard import SummaryWriter 
import os 


def create_tensorboard_writer (log_dir ="temp/timing",comment =None ,purge_step =None ,max_queue =10 ,flush_secs =120 ,filename_suffix =''):
    "Documentation."
    
    os .makedirs (log_dir ,exist_ok =True )

    
    return SummaryWriter (
    log_dir =log_dir ,
    comment =comment ,
    purge_step =purge_step ,
    max_queue =max_queue ,
    flush_secs =flush_secs ,
    filename_suffix =filename_suffix 
    )




    
    
    
class GradientLossVisualizer :
    "Documentation."

    def __init__ (self ,num_stations ,tb_writer =None ):
        "Documentation."
        self .num_stations =num_stations 
        self .tb_writer =tb_writer 
        self .mode =None 
        self .reset_episode_data ()

    def _set_mode (self ,mode ):
        if self .mode is None :
            self .mode =mode 
        elif self .mode !=mode :
            self .mode =mode 

    def update_q_values (self ,q_values_per_agent ,q_mean ,q_global ):
        "Documentation."
        self ._set_mode ("distributed")
        for i ,q_val in enumerate (q_values_per_agent ):
            if i <len (self .local_q_agents_sums ):
                self .local_q_agents_sums [i ]+=q_val 
        self .local_q_mean_sum +=q_mean 
        self .global_q_sum +=q_global 
        self .q_step_count +=1 

    def update_central_q_value (self ,q_value ):
        "Documentation."
        self ._set_mode ("centralized_joint")
        self .central_q_sum +=float (q_value )
        self .q_step_count +=1 

    def update_gradients (self ,agent ):
        "Documentation."
        is_central =getattr (agent ,"visualizer_layout","")=="centralized_joint"
        if is_central :
            self ._set_mode ("centralized_joint")
            self .central_critic_grad_sum_before_clip +=float (
            getattr (agent ,"last_central_critic_grad_norm_before_clip",0.0 )
            )
            self .central_actor_grad_sum_before_clip +=float (
            getattr (agent ,"last_central_actor_grad_norm_before_clip",0.0 )
            )
        else :
            self ._set_mode ("distributed")
            self .global_critic_grad_sum_before_clip +=getattr (agent ,'last_global_critic_grad_norm_before_clip',0.0 )
            self .actor_source_local_grad_sum_before_clip +=float (
            getattr (agent ,"last_actor_source_local_grad_norm_before_clip",0.0 )
            )
            self .actor_source_global_grad_sum_before_clip +=float (
            getattr (agent ,"last_actor_source_global_grad_norm_before_clip",0.0 )
            )
            self .actor_source_global_ratio_sum +=float (
            getattr (agent ,"last_actor_source_global_ratio",0.0 )
            )
            self .actor_source_cos_sum +=float (getattr (agent ,"last_actor_source_cos",0.0 ))
            self .actor_source_cos_valid_fraction_sum +=float (
            getattr (agent ,"last_actor_source_cos_valid_fraction",0.0 )
            )

            if hasattr (agent ,'actor_source_local_norms_before_clip')and len (agent .actor_source_local_norms_before_clip )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_local_norms_before_clip ):
                    if i <len (self .actor_source_local_norm_sums_before_clip ):
                        self .actor_source_local_norm_sums_before_clip [i ]+=float (v )

            if hasattr (agent ,'actor_source_global_norms_before_clip')and len (agent .actor_source_global_norms_before_clip )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_global_norms_before_clip ):
                    if i <len (self .actor_source_global_norm_sums_before_clip ):
                        self .actor_source_global_norm_sums_before_clip [i ]+=float (v )

            if hasattr (agent ,'actor_source_global_ratio')and len (agent .actor_source_global_ratio )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_global_ratio ):
                    if i <len (self .actor_source_global_ratio_sums ):
                        self .actor_source_global_ratio_sums [i ]+=float (v )

            if hasattr (agent ,'actor_source_cos')and len (agent .actor_source_cos )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_cos ):
                    if i <len (self .actor_source_cos_sums ):
                        self .actor_source_cos_sums [i ]+=float (v )

            if hasattr (agent ,'actor_source_cos_valid')and len (agent .actor_source_cos_valid )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_cos_valid ):
                    if i <len (self .actor_source_cos_valid_sums ):
                        self .actor_source_cos_valid_sums [i ]+=int (v )

            if hasattr (agent ,'critic_norms_before_clip')and len (agent .critic_norms_before_clip )==self .num_stations :
                for i ,critic_norm_before in enumerate (agent .critic_norms_before_clip ):
                    if i <len (self .critic_norms_sums_before_clip ):
                        self .critic_norms_sums_before_clip [i ]+=critic_norm_before 

            if hasattr (agent ,'actor_norms_before_clip')and len (agent .actor_norms_before_clip )==self .num_stations :
                for i ,actor_norm_before in enumerate (agent .actor_norms_before_clip ):
                    if i <len (self .actor_norms_sums_before_clip ):
                        self .actor_norms_sums_before_clip [i ]+=actor_norm_before 

            self .local_critic_grad_sum +=getattr (agent ,'last_local_critic_grad_norm',0.0 )
            self .global_critic_grad_sum +=getattr (agent ,'last_global_critic_grad_norm',0.0 )

        self .grad_step_count +=1 

    def update_losses (self ,agent ):
        "Documentation."
        is_central =getattr (agent ,"visualizer_layout","")=="centralized_joint"
        if is_central :
            self ._set_mode ("centralized_joint")
            self .central_critic_loss_sum +=float (getattr (agent ,"last_central_critic_loss",0.0 ))
            self .central_actor_loss_sum +=float (getattr (agent ,"last_central_actor_loss",0.0 ))
        else :
            self ._set_mode ("distributed")
            if hasattr (agent ,'critic_losses')and len (agent .critic_losses )==self .num_stations :
                for i ,loss in enumerate (agent .critic_losses ):
                    if i <len (self .critic_loss_sums ):
                        self .critic_loss_sums [i ]+=loss 

            self .global_critic_loss_sum +=getattr (agent ,'last_global_critic_loss',0.0 )

            if hasattr (agent ,'actor_losses')and len (agent .actor_losses )==self .num_stations :
                for i ,loss in enumerate (agent .actor_losses ):
                    if i <len (self .actor_loss_sums ):
                        self .actor_loss_sums [i ]+=loss 

        self .loss_step_count +=1 

    def update_clipping (self ,agent ):
        "Documentation."
        is_central =getattr (agent ,"visualizer_layout","")=="centralized_joint"
        if is_central :
            self ._set_mode ("centralized_joint")
            self .central_critic_clip_sum +=int (getattr (agent ,"last_central_critic_clip_count",0 ))
            self .central_actor_clip_sum +=int (getattr (agent ,"last_central_actor_clip_count",0 ))
        else :
            self ._set_mode ("distributed")
            self .global_critic_clip_sum +=getattr (agent ,'last_global_critic_clip_count',0 )

            if hasattr (agent ,'local_critic_clip_counts')and len (agent .local_critic_clip_counts )==self .num_stations :
                for i ,clip_count in enumerate (agent .local_critic_clip_counts ):
                    if i <len (self .local_critic_clip_sums ):
                        self .local_critic_clip_sums [i ]+=clip_count 

            if hasattr (agent ,'actor_clip_counts')and len (agent .actor_clip_counts )==self .num_stations :
                for i ,clip_count in enumerate (agent .actor_clip_counts ):
                    if i <len (self .actor_clip_sums ):
                        self .actor_clip_sums [i ]+=clip_count 

        self .clip_step_count +=1 

    def record_to_tensorboard (self ,episode ):
        "Documentation."
        if not self .tb_writer :
            return 

        w =self .tb_writer 

        
        if self .mode =="centralized_joint":
            if self .q_step_count >0 :
                w .add_scalar ("Q/central_mean",self .central_q_sum /self .q_step_count ,episode )
            if self .grad_step_count >0 :
                w .add_scalar ("Gradient/central_critic_raw",
                self .central_critic_grad_sum_before_clip /self .grad_step_count ,episode )
                w .add_scalar ("Gradient/central_actor_raw",
                self .central_actor_grad_sum_before_clip /self .grad_step_count ,episode )
            if self .loss_step_count >0 :
                w .add_scalar ("Loss/central_critic",
                self .central_critic_loss_sum /self .loss_step_count ,episode )
                w .add_scalar ("Loss/central_actor",
                self .central_actor_loss_sum /self .loss_step_count ,episode )
            if self .clip_step_count >0 :
                w .add_scalar ("Clipping/central_critic",
                self .central_critic_clip_sum /self .clip_step_count ,episode )
                w .add_scalar ("Clipping/central_actor",
                self .central_actor_clip_sum /self .clip_step_count ,episode )
            return 

            
            
        if self .q_step_count >0 :
            n =self .q_step_count 
            for i ,s in enumerate (self .local_q_agents_sums ):
                w .add_scalar (f"Q/local_agent{i+1}",s /n ,episode )
            w .add_scalar ("Q/local_mean",self .local_q_mean_sum /n ,episode )
            w .add_scalar ("Q/global",self .global_q_sum /n ,episode )

            
        if self .grad_step_count >0 :
            n =self .grad_step_count 
            for i ,v in enumerate (self .critic_norms_sums_before_clip ):
                w .add_scalar (f"Gradient/local_critic_raw_agent{i+1}",v /n ,episode )
            w .add_scalar ("Gradient/global_critic_raw",
            self .global_critic_grad_sum_before_clip /n ,episode )
            for i ,v in enumerate (self .actor_norms_sums_before_clip ):
                w .add_scalar (f"Gradient/actor_raw_agent{i+1}",v /n ,episode )
            for i ,v in enumerate (self .actor_source_local_norm_sums_before_clip ):
                w .add_scalar (f"Gradient/actor_source_local_raw_agent{i+1}",v /n ,episode )
            for i ,v in enumerate (self .actor_source_global_norm_sums_before_clip ):
                w .add_scalar (f"Gradient/actor_source_global_raw_agent{i+1}",v /n ,episode )
            for i ,v in enumerate (self .actor_source_global_ratio_sums ):
                w .add_scalar (f"Gradient/actor_source_global_ratio_agent{i+1}",v /n ,episode )
            for i ,v in enumerate (self .actor_source_cos_sums ):
                w .add_scalar (f"Gradient/actor_source_cos_agent{i+1}",v /n ,episode )
            for i ,v in enumerate (self .actor_source_cos_valid_sums ):
                w .add_scalar (f"Gradient/actor_source_cos_valid_fraction_agent{i+1}",v /n ,episode )
            w .add_scalar ("Gradient/actor_source_local_raw_mean",
            self .actor_source_local_grad_sum_before_clip /n ,episode )
            w .add_scalar ("Gradient/actor_source_global_raw_mean",
            self .actor_source_global_grad_sum_before_clip /n ,episode )
            w .add_scalar ("Gradient/actor_source_global_ratio_mean",
            self .actor_source_global_ratio_sum /n ,episode )
            w .add_scalar ("Gradient/actor_source_cos_mean",
            self .actor_source_cos_sum /n ,episode )
            w .add_scalar ("Gradient/actor_source_cos_valid_fraction_mean",
            self .actor_source_cos_valid_fraction_sum /n ,episode )
            
            w .add_scalar ("Gradient/local_critic_mean_after_clip",
            self .local_critic_grad_sum /n ,episode )
            w .add_scalar ("Gradient/global_critic_after_clip",
            self .global_critic_grad_sum /n ,episode )

            
        if self .loss_step_count >0 :
            n =self .loss_step_count 
            for i ,v in enumerate (self .critic_loss_sums ):
                w .add_scalar (f"Loss/local_critic_agent{i+1}",v /n ,episode )
            w .add_scalar ("Loss/global_critic",self .global_critic_loss_sum /n ,episode )
            for i ,v in enumerate (self .actor_loss_sums ):
                w .add_scalar (f"Loss/actor_agent{i+1}",v /n ,episode )
            w .add_scalar ("Loss/local_critic_mean",
            sum (self .critic_loss_sums )/len (self .critic_loss_sums )/n 
            if self .critic_loss_sums else 0.0 ,episode )

            
        if self .clip_step_count >0 :
            n =self .clip_step_count 
            for i ,v in enumerate (self .local_critic_clip_sums ):
                w .add_scalar (f"Clipping/local_critic_agent{i+1}",v /n ,episode )
            w .add_scalar ("Clipping/global_critic",self .global_critic_clip_sum /n ,episode )
            for i ,v in enumerate (self .actor_clip_sums ):
                w .add_scalar (f"Clipping/actor_agent{i+1}",v /n ,episode )

    def reset_episode_data (self ):
        "Documentation."
        self .mode =None 

        self .local_q_agents_sums =[0.0 ]*self .num_stations 
        self .local_q_mean_sum =0.0 
        self .global_q_sum =0.0 
        self .central_q_sum =0.0 
        self .q_step_count =0 

        self .global_critic_grad_sum_before_clip =0.0 
        self .critic_norms_sums_before_clip =[0.0 ]*self .num_stations 
        self .actor_norms_sums_before_clip =[0.0 ]*self .num_stations 
        self .actor_source_local_norm_sums_before_clip =[0.0 ]*self .num_stations 
        self .actor_source_global_norm_sums_before_clip =[0.0 ]*self .num_stations 
        self .actor_source_global_ratio_sums =[0.0 ]*self .num_stations 
        self .actor_source_cos_sums =[0.0 ]*self .num_stations 
        self .actor_source_cos_valid_sums =[0 ]*self .num_stations 
        self .actor_source_local_grad_sum_before_clip =0.0 
        self .actor_source_global_grad_sum_before_clip =0.0 
        self .actor_source_global_ratio_sum =0.0 
        self .actor_source_cos_sum =0.0 
        self .actor_source_cos_valid_fraction_sum =0.0 
        self .central_critic_grad_sum_before_clip =0.0 
        self .central_actor_grad_sum_before_clip =0.0 
        self .grad_step_count =0 

        self .critic_loss_sums =[0.0 ]*self .num_stations 
        self .global_critic_loss_sum =0.0 
        self .actor_loss_sums =[0.0 ]*self .num_stations 
        self .central_critic_loss_sum =0.0 
        self .central_actor_loss_sum =0.0 
        self .loss_step_count =0 

        self .local_critic_clip_sums =[0 ]*self .num_stations 
        self .global_critic_clip_sum =0 
        self .actor_clip_sums =[0 ]*self .num_stations 
        self .central_critic_clip_sum =0 
        self .central_actor_clip_sum =0 
        self .clip_step_count =0 

        self .local_critic_grad_sum =0.0 
        self .global_critic_grad_sum =0.0 


        
        
        
def plot_arrival_counts (all_episode_data :dict ,
results_dir :str ,
title_prefix :str ="")->None :
    "Documentation."
    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )==0 :
        return 

    ep_key =episode_keys [-1 ]
    ep =all_episode_data [ep_key ]

    if 'arrivals_per_step'not in ep or len (ep ['arrivals_per_step'])==0 :
        return 

    raw_arrivals =ep ['arrivals_per_step']

    
    is_vector =any (isinstance (v ,(list ,tuple ,np .ndarray ))for v in raw_arrivals )
    if is_vector :
    
        rows =[]
        max_s =0 
        for v in raw_arrivals :
            if isinstance (v ,(list ,tuple ,np .ndarray )):
                arr =np .asarray (v ,dtype =float ).reshape (-1 )
            else :
                arr =np .asarray ([float (v )],dtype =float )
            max_s =max (max_s ,int (arr .size ))
            rows .append (arr )
        T =len (rows )
        S =max_s 
        arrivals_by_station =np .zeros ((T ,S ),dtype =float )
        for t ,arr in enumerate (rows ):
            n =min (S ,int (arr .size ))
            if n >0 :
                arrivals_by_station [t ,:n ]=arr [:n ]

                
        if arrivals_by_station .shape [0 ]>0 and INITIAL_EVS_PER_STATION >0 :
            arrivals_by_station [0 ,:]=np .maximum (
            arrivals_by_station [0 ,:]-float (INITIAL_EVS_PER_STATION ),0.0 
            )

        arrivals_total =arrivals_by_station .sum (axis =1 )
        steps =np .arange (1 ,arrivals_total .shape [0 ]+1 )
    else :
        arrivals_list =[]
        for value in raw_arrivals :
            try :
                arrivals_list .append (float (value ))
            except (TypeError ,ValueError ):
                arrivals_list .append (0.0 )
        arrivals_total =np .asarray (arrivals_list ,dtype =float )
        if arrivals_total .size >0 :
            initial_seed_total =INITIAL_EVS_PER_STATION *NUM_STATIONS 
            if initial_seed_total >0 :
                arrivals_total [0 ]=max (arrivals_total [0 ]-initial_seed_total ,0.0 )
        steps =np .arange (1 ,len (arrivals_total )+1 )

    title_prefix_en =title_prefix

    if is_vector :
    
        try :
            from pathlib import Path 
            from Config import (
            PER_STATION_ARRIVAL_PROFILE_PATHS ,
            )
        except Exception :
            PER_STATION_ARRIVAL_PROFILE_PATHS =[]

        S =int (arrivals_by_station .shape [1 ])
        if len (PER_STATION_ARRIVAL_PROFILE_PATHS )<S :
            print ("[plot_arrival_counts] per-station arrival profiles are required; skipping grouped plot.")
            return 
        profile_paths =list (PER_STATION_ARRIVAL_PROFILE_PATHS [:S ])

        
        groups ={}
        for st ,pth in enumerate (profile_paths ):
            groups .setdefault (str (pth ),[]).append (st )

            
        group_items =[]
        for pth ,sts in groups .items ():
            group_total =float (arrivals_by_station [:,sts ].sum ())if sts else 0.0 
            rep =int (sts [0 ])if sts else 0 
            group_items .append ((group_total ,pth ,rep ,sts ))
        group_items .sort (reverse =True ,key =lambda x :x [0 ])
        selected =group_items [:5 ]

        fig ,axes =plt .subplots (5 ,1 ,figsize =(20 ,18 ),sharex =True )
        for i in range (5 ):
            ax =axes [i ]
            if i <len (selected ):
                _ ,pth ,rep ,sts =selected [i ]
                y =arrivals_by_station [:,rep ]
                label_name =Path (pth ).stem if pth else "arrival_profile"
                ax .bar (steps ,y ,color ='skyblue',alpha =0.55 )
                ax .plot (steps ,y ,color ='navy',linewidth =2 )
                ax .set_title (f"Station {rep} (rep of {len(sts)} stations) | {label_name}",fontsize =14 )
                ax .grid (alpha =0.3 ,linewidth =1.0 )
                ax .set_ylabel ('Arrivals',fontsize =12 )
            else :
                ax .axis ('off')

        axes [-1 ].set_xlabel ('Step',fontsize =14 )
        fig .suptitle (f"{title_prefix_en} Arrivals per Step (by station, Episode {ep_key})".strip (),fontsize =18 )
        fig .tight_layout ()
    else :
    
        fig ,ax =plt .subplots (figsize =(20 ,10 ))
        ax .bar (steps ,arrivals_total ,color ='skyblue',alpha =0.6 ,label ='New Arrivals (bar)')
        ax .plot (steps ,arrivals_total ,color ='navy',linewidth =3 ,marker ='o',markersize =6 ,label ='New Arrivals (line)')

        ax .set_xlabel ('Step',fontsize =26 )
        ax .set_ylabel ('Newly Arrived EVs [count]',fontsize =26 )
        ax .grid (alpha =0.3 ,linewidth =1.5 )
        ax .legend (fontsize =18 )
        fig .tight_layout ()

    fname =f"arrive_EV_per_step_episode_{ep_key}.png"
    if title_prefix :
        prefix_clean =title_prefix .lower ().replace (' ','_')
        fname =f"{prefix_clean}_{fname}"

    fig .savefig (os .path .join (results_dir ,fname ),dpi =140 ,bbox_inches ='tight')
    plt .close (fig )
    

    
    csv_filename =fname .replace ('.png','.csv')
    csv_path =os .path .join (results_dir ,csv_filename )
    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        writer .writerow (['Step','New_Arrivals'])
        for i ,value in enumerate (arrivals_total ):
            writer .writerow ([i +1 ,int (value )])

            
    if is_vector :
        by_station_csv =csv_filename .replace ('.csv','_by_station.csv')
        by_station_path =os .path .join (results_dir ,by_station_csv )
        with open (by_station_path ,'w',newline ='',encoding ='utf-8')as f :
            writer =csv .writer (f )
            header =['Step']+[f'Station_{i}'for i in range (int (arrivals_by_station .shape [1 ]))]+['Total']
            writer .writerow (header )
            for t in range (int (arrivals_by_station .shape [0 ])):
                row =[t +1 ]+[int (x )for x in arrivals_by_station [t ,:]]+[int (arrivals_total [t ])]
                writer .writerow (row )
                


                
                
                
def plot_power_mismatch_analysis (all_episode_data :dict ,
results_dir :str ,
title_prefix :str ="")->None :
    "Documentation."
    
    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )==0 :
        return 

    ep_key =episode_keys [-1 ]
    ep =all_episode_data [ep_key ]

    if 'power_mismatch'not in ep or len (ep ['power_mismatch'])==0 :
        return 

    mismatches =np .array (ep ['power_mismatch'])
    steps =np .arange (1 ,len (mismatches )+1 )

    
    max_over_idx =np .argmax (mismatches )
    max_under_idx =np .argmin (mismatches )
    max_over_value =mismatches [max_over_idx ]
    max_under_value =mismatches [max_under_idx ]

    
    energy_mismatches =mismatches *POWER_TO_ENERGY 
    cumulative_energy =np .cumsum (energy_mismatches )

    
    max_cumulative_idx =np .argmax (np .abs (cumulative_energy ))
    max_cumulative_value =cumulative_energy [max_cumulative_idx ]

    
    title_prefix_en =title_prefix

    
    fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(20 ,16 ))

    
    colors =['red'if m >0 else 'blue'for m in mismatches ]
    ax1 .bar (steps ,mismatches ,color =colors ,alpha =0.6 ,width =0.8 )

    
    ax1 .bar (max_over_idx ,max_over_value ,color ='darkred',alpha =0.9 ,width =0.8 )
    ax1 .bar (max_under_idx ,max_under_value ,color ='darkblue',alpha =0.9 ,width =0.8 )

    
    ax1 .text (max_over_idx ,max_over_value ,
    f'Max Shortage\nStep {max_over_idx+1}\n{max_over_value:.1f} kW',
    ha ='center',va ='bottom',fontsize =24 ,color ='darkred',fontweight ='bold')
    ax1 .text (max_under_idx ,max_under_value ,
    f'Max Excess\nStep {max_under_idx+1}\n{max_under_value:.1f} kW',
    ha ='center',va ='top',fontsize =24 ,color ='darkblue',fontweight ='bold')

    ax1 .axhline (0 ,color ='black',linewidth =2 ,alpha =0.5 )
    ax1 .set_xlabel ('Step',fontsize =30 )
    ax1 .set_ylabel ('Mismatch [kW]\n(Request - Actual)',fontsize =30 )
    
    
    ax1 .grid (alpha =0.3 ,linewidth =2 )

    
    ax2 .plot (steps ,cumulative_energy ,linewidth =4 ,color ='purple',alpha =0.8 )
    ax2 .fill_between (steps ,0 ,cumulative_energy ,alpha =0.3 ,color ='purple')

    
    ax2 .plot (max_cumulative_idx ,max_cumulative_value ,'o',
    markersize =20 ,color ='darkred',zorder =5 )
    ax2 .text (max_cumulative_idx ,max_cumulative_value ,
    f'Max Cumulative\nStep {max_cumulative_idx+1}\n{max_cumulative_value:.2f} kWh',
    ha ='center',va ='bottom'if max_cumulative_value >0 else 'top',
    fontsize =24 ,color ='darkred',fontweight ='bold')

    ax2 .axhline (0 ,color ='black',linewidth =2 ,alpha =0.5 )
    ax2 .set_xlabel ('Step',fontsize =30 )
    ax2 .set_ylabel ('Cumulative Energy Mismatch [kWh]',fontsize =30 )
    
    
    ax2 .grid (alpha =0.3 ,linewidth =2 )

    fig .tight_layout ()

    
    fname =f"power_mismatch_analysis_episode_{ep_key}.png"
    if title_prefix :
        prefix_clean =title_prefix .lower ().replace (' ','_')
        fname =f"{prefix_clean}_{fname}"

    fig .savefig (os .path .join (results_dir ,fname ),dpi =140 ,bbox_inches ='tight')
    plt .close (fig )
    

    
    csv_filename =fname .replace ('.png','.csv')
    csv_path =os .path .join (results_dir ,csv_filename )
    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        writer .writerow (['Step','Mismatch_kW','Cumulative_Energy_kWh'])
        for i in range (len (steps )):
            writer .writerow ([i +1 ,mismatches [i ],cumulative_energy [i ]])
            


def plot_reward_breakdown (all_episode_data :dict ,
results_dir :str ,
title_prefix :str ="")->None :
    "Documentation."
    
    title_prefix_en =title_prefix

    
    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )>2 :
        episode_keys =episode_keys [-2 :]

    for ep_key in episode_keys :
        ep =all_episode_data [ep_key ]

        
        if 'rewards_global_balance'not in ep :
            continue 

        steps =np .arange (1 ,len (ep ['rewards_global_balance'])+1 )
        local_shaping =np .asarray (ep ['rewards_local_shaping'],dtype =float )
        local_departure =np .asarray (ep ['rewards_local_departure'],dtype =float )
        local_discharge_penalty =np .asarray (ep ['rewards_local_discharge_penalty'],dtype =float )
        local_station_limit_penalty =np .asarray (
        ep .get ('rewards_local_station_limit_penalty',np .zeros (len (steps ))),
        dtype =float ,
        )
        global_balance =np .asarray (ep ['rewards_global_balance'],dtype =float )

        fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(20 ,16 ),sharex =True )

        
        ax1 .plot (steps ,local_shaping ,label ='Local Shaping',color ='cyan',lw =3 ,alpha =0.8 )
        ax1 .plot (steps ,local_departure ,label ='Local Departure',color ='orange',lw =4 ,marker ='o',markersize =4 ,ls ='None')
        ax1 .plot (steps ,local_discharge_penalty ,label ='Local Discharge Penalty',color ='red',lw =2 ,alpha =0.6 )
        ax1 .plot (steps ,local_station_limit_penalty ,label ='Local Station Limit Penalty',color ='magenta',lw =3 ,alpha =0.9 )
        ax1 .fill_between (
        steps ,
        0 ,
        local_station_limit_penalty ,
        color ='magenta',
        alpha =0.18 ,
        linewidth =0 ,
        )

        
        local_total =local_shaping +local_departure +local_discharge_penalty +local_station_limit_penalty 
        ax1 .plot (steps ,local_total ,label ='Local Total (Sum)',color ='green',lw =2 ,ls ='--',alpha =0.5 )

        
        all_local_vals =np .concatenate ([
        local_shaping ,
        local_departure ,
        local_discharge_penalty ,
        local_station_limit_penalty ,
        local_total ,
        ])
        if len (all_local_vals )>0 :
            v_min ,v_max =np .min (all_local_vals ),np .max (all_local_vals )
            
            if v_min ==v_max :
                ax1 .set_ylim (v_min -1.0 ,v_max +1.0 )
            else :
                margin =(v_max -v_min )*0.1 
                ax1 .set_ylim (v_min -margin ,v_max +margin )

        ax1 .set_ylabel ('Local Reward Components',fontsize =26 )
        
        ax1 .grid (alpha =0.3 ,lw =1.5 )
        ax1 .legend (loc ='upper left',fontsize =18 )
        ax1 .axhline (0 ,color ='black',lw =2 ,alpha =0.5 )

        
        ax2 .plot (steps ,global_balance ,label ='Global Balance',color ='blue',lw =4 )

        
        if len (global_balance )>0 :
            gv_min ,gv_max =np .min (global_balance ),np .max (global_balance )
            if gv_min ==gv_max :
                ax2 .set_ylim (gv_min -1.0 ,gv_max +1.0 )
            else :
                margin =(gv_max -gv_min )*0.1 
                ax2 .set_ylim (gv_min -margin ,gv_max +margin )

        ax2 .set_xlabel ('Step',fontsize =26 )
        ax2 .set_ylabel ('Global Reward Components',fontsize =26 )
        
        ax2 .grid (alpha =0.3 ,lw =1.5 )
        ax2 .legend (loc ='upper left',fontsize =18 )
        ax2 .axhline (0 ,color ='black',lw =2 ,alpha =0.5 )

        
        

        fig .tight_layout (rect =[0 ,0.03 ,1 ,0.95 ])

        
        fname =f"reward_breakdown_episode_{ep_key}.png"
        if title_prefix :
            prefix_clean =title_prefix_en .lower ().replace (' ','_')
            fname =f"{prefix_clean}_{fname}"

        fig .savefig (os .path .join (results_dir ,fname ),dpi =140 ,bbox_inches ='tight')
        plt .close (fig )
        

        
        csv_filename =fname .replace ('.png','.csv')
        csv_path =os .path .join (results_dir ,csv_filename )
        with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
            writer =csv .writer (f )
            writer .writerow ([
            'Step',
            'Global_Balance',
            'Local_Shaping',
            'Local_Departure',
            'Local_Discharge_Penalty',
            'Local_Station_Limit_Penalty',
            ])
            for i in range (len (steps )):
                writer .writerow ([
                steps [i ],
                global_balance [i ],
                local_shaping [i ],
                local_departure [i ],
                local_discharge_penalty [i ],
                local_station_limit_penalty [i ],
                ])
                


                
                
                
def snapshot_code_to_archive (model_dir :str ,project_root :str =None )->str :
    "Documentation."
    root =project_root or os .getcwd ()
    snapshot_dir =os .path .join (model_dir ,"code_snapshot")
    os .makedirs (snapshot_dir ,exist_ok =True )

    
    dirs_to_copy =["training","environment","tools","data"]
    for d in dirs_to_copy :
        src =os .path .join (root ,d )
        dst =os .path .join (snapshot_dir ,d )
        if os .path .isdir (src ):
        
            def _ignore (dirpath ,names ):
                ignored =set ()
                base =os .path .basename (dirpath )
                if base in {"__pycache__",".git",".idea",".vscode",".venv",".pytest_cache",".mypy_cache"}:
                    ignored .update (names )
                return ignored 
            shutil .copytree (src ,dst ,dirs_exist_ok =True ,ignore =_ignore )

            
    root_files =[
    fn for fn in os .listdir (root )
    if os .path .isfile (os .path .join (root ,fn ))
    ]
    allow_names ={"requirements.txt","README.md","README.MD",".env"}
    for fn in root_files :
        if fn .endswith (".py")or fn in allow_names :
            src =os .path .join (root ,fn )
            dst =os .path .join (snapshot_dir ,fn )
            try :
                shutil .copy2 (src ,dst )
            except Exception :
            
                pass 

                
                

    return snapshot_dir 


    
    
    
def _plot_ev_soc_with_dispatch_from_csv_fallback (
ev_soc_csv_path ,
out_dir =None ,
start_step =None ,
end_step =None ,
dispatch_csv_path =None ,
):
    """Internal fallback plotter when Graphs.EV_SOC_Dispatch is unavailable."""
    from pathlib import Path 

    ev_soc_csv_path =Path (ev_soc_csv_path )
    out_dir =Path (out_dir )if out_dir else ev_soc_csv_path .parent 
    dispatch_csv_path =Path (dispatch_csv_path )if dispatch_csv_path else None 
    out_dir .mkdir (parents =True ,exist_ok =True )

    
    ev_rows ={}
    with open (ev_soc_csv_path ,"r",encoding ="utf-8",newline ="")as f :
        reader =csv .DictReader (f )
        for row in reader :
            try :
                step =int (float (row .get ("Step",0 )))
            except Exception :
                continue 
            ev_id =str (row .get ("EV_ID","EV"))
            try :
                soc =float (row .get ("SoC_%",np .nan ))
            except Exception :
                soc =np .nan 
            try :
                target =float (row .get ("Target_%",np .nan ))
            except Exception :
                target =np .nan 
            dep_raw =row .get ("Departure_Step","")
            try :
                departure =int (float (dep_raw ))if str (dep_raw ).strip ()not in {"","N/A","None","nan"}else None 
            except Exception :
                departure =None 

            d =ev_rows .setdefault (ev_id ,{"step":[],"soc":[],"target":target ,"departure":departure })
            d ["step"].append (step )
            d ["soc"].append (soc )
            if np .isfinite (target ):
                d ["target"]=target 
            if departure is not None :
                d ["departure"]=departure 

    if not ev_rows :
        raise ValueError (f"No EV SoC rows found in {ev_soc_csv_path}")

    all_steps =sorted ({s for d in ev_rows .values ()for s in d ["step"]})
    if not all_steps :
        raise ValueError (f"No valid step values found in {ev_soc_csv_path}")

    lo =all_steps [0 ]if start_step is None else int (start_step )
    hi =all_steps [-1 ]if end_step is None else int (end_step )
    if hi <lo :
        lo ,hi =hi ,lo 

    fig ,ax1 =plt .subplots (figsize =(14 ,8 ))
    target_label_shown =False 
    for ev_id in sorted (ev_rows .keys (),key =lambda x :(len (x ),x )):
        d =ev_rows [ev_id ]
        s =np .asarray (d ["step"],dtype =int )
        y =np .asarray (d ["soc"],dtype =float )
        m =(s >=lo )&(s <=hi )&np .isfinite (y )
        if not np .any (m ):
            continue 
        line ,=ax1 .plot (s [m ],y [m ],label =f"EV {ev_id}",alpha =0.95 ,lw =2.8 )
        if d .get ("departure")is not None and lo <=d ["departure"]<=hi :
            ax1 .axvline (d ["departure"],color ="gray",ls ="--",alpha =0.25 ,lw =1.2 )
        if np .isfinite (d .get ("target",np .nan )):
            seg_lo =int (np .min (s [m ]))
            seg_hi =int (np .max (s [m ]))
            ax1 .hlines (
            y =float (d ["target"]),
            xmin =seg_lo ,
            xmax =seg_hi ,
            colors =line .get_color (),
            linestyles ="--",
            lw =2.2 ,
            alpha =0.85 ,
            label ="Target SoC"if not target_label_shown else None ,
            )
            target_label_shown =True 

    ax1 .set_xlim (lo ,hi )
    ax1 .set_ylim (0 ,100 )
    ax1 .set_xlabel ("Step")
    ax1 .set_ylabel ("SoC (%)")
    ax1 .grid (True ,alpha =0.3 )

    
    if dispatch_csv_path and dispatch_csv_path .exists ():
        disp_x ,disp_y =[],[]
        with open (dispatch_csv_path ,"r",encoding ="utf-8",newline ="")as f :
            reader =csv .DictReader (f )
            fields =[c for c in (reader .fieldnames or [])if c ]
            step_col =None 
            for c in fields :
                if c .lower ()=="step":
                    step_col =c 
                    break 
            if step_col is None and fields :
                step_col =fields [0 ]

            val_candidates =[
            "demand_adjustment",
            "ag_request",
            "dispatch",
            "dispatch_signal",
            "request",
            "value",
            ]
            val_col =None 
            lower_to_orig ={c .lower ():c for c in fields }
            for name in val_candidates :
                if name in lower_to_orig :
                    val_col =lower_to_orig [name ]
                    break 
            if val_col is None :
                for c in fields :
                    if c !=step_col :
                        val_col =c 
                        break 

            for row in reader :
                try :
                    sx =int (float (row .get (step_col ,np .nan )))
                    sy =float (row .get (val_col ,np .nan ))
                except Exception :
                    continue 
                if np .isfinite (sx )and np .isfinite (sy )and lo <=sx <=hi :
                    disp_x .append (sx )
                    disp_y .append (sy )

        if disp_x :
            ax2 =ax1 .twinx ()
            disp_x =np .asarray (disp_x ,dtype =int )
            disp_y =np .asarray (disp_y ,dtype =float )
            y_pos =np .where (disp_y >=0.0 ,disp_y ,np .nan )
            y_neg =np .where (disp_y <0.0 ,disp_y ,np .nan )
            ax2 .plot (disp_x ,y_pos ,color ="tab:red",lw =3.2 ,alpha =0.9 ,label ="Dispatch (+)")
            ax2 .plot (disp_x ,y_neg ,color ="tab:blue",lw =3.2 ,alpha =0.9 ,label ="Dispatch (-)")
            pos =disp_y >0 
            neg =disp_y <0 
            if np .any (pos ):
                ax2 .fill_between (disp_x ,0 ,disp_y ,where =pos ,color ="tab:red",alpha =0.18 )
            if np .any (neg ):
                ax2 .fill_between (disp_x ,0 ,disp_y ,where =neg ,color ="tab:blue",alpha =0.18 )
            ax2 .set_ylabel ("Dispatch signal")
            h2 ,l2 =ax2 .get_legend_handles_labels ()
            if h2 :
                ax2 .legend (loc ="upper right")

    handles ,labels =ax1 .get_legend_handles_labels ()
    if handles :
        ax1 .legend (loc ="best",ncol =2 )

    graph_path =out_dir /f"{ev_soc_csv_path.stem}_graph.png"
    fig .tight_layout ()
    fig .savefig (graph_path ,dpi =140 )
    plt .close (fig )

    legend_path =out_dir /f"{ev_soc_csv_path.stem}_legend.png"
    return graph_path ,legend_path 


def plot_ev_soc_with_dispatch_from_csv (
ev_soc_csv_path :str ,
out_dir :str =None ,
start_step :int =None ,
end_step :int =None ,
dispatch_csv_path :str =None ,
):
    """
    Plot EV SoC and Dispatch signal value from CSV files.

    First tries Graphs.EV_SOC_Dispatch, then falls back to an internal implementation.
    """
    from pathlib import Path 

    ev_soc_csv_path =Path (ev_soc_csv_path )
    out_dir =Path (out_dir )if out_dir else None 
    dispatch_csv_path =Path (dispatch_csv_path )if dispatch_csv_path else None 


    return _plot_ev_soc_with_dispatch_from_csv_fallback (
        ev_soc_csv_path =ev_soc_csv_path ,
        out_dir =out_dir ,
        start_step =start_step ,
        end_step =end_step ,
        dispatch_csv_path =dispatch_csv_path ,
        )


        
        
        

import socket as _socket 
import urllib .request as _urllib_request 
import urllib .error as _urllib_error 


def check_port_available (port ):
    "Documentation."
    sock =_socket .socket (_socket .AF_INET ,_socket .SOCK_STREAM )
    try :
        sock .bind (('localhost',port ))
        sock .close ()
        return True 
    except OSError :
        return False 


def find_available_port (start_port =6006 ,max_attempts =10 ):
    "Documentation."
    for p in range (start_port ,start_port +max_attempts ):
        if check_port_available (p ):
            return p 
    return None 


def check_tensorboard_running (port ,max_wait =30 ):
    "Documentation."
    import time as _time 
    url =f"http://localhost:{port}"
    for _ in range (max_wait ):
        try :
            response =_urllib_request .urlopen (url ,timeout =1 )
            if response .getcode ()==200 :
                return True 
        except (_urllib_error .URLError ,_socket .timeout ,ConnectionRefusedError ):
            _time .sleep (1 )
    return False 


def runtensorboard_main ():
    """TensorBoard launcher CLI entry point (formerly main() in runtensorboard.py)."""
    import sys 
    import subprocess 
    import webbrowser 

    
    if len (sys .argv )>1 :
        logdir =sys .argv [1 ].strip ()
    else :
        logdir =input ("Text.").strip ()
        if not logdir :
            sys .exit (1 )

    port =6006 
    if not check_port_available (port ):
        new_port =find_available_port (port +1 )
        if new_port :
            port =new_port 
        else :
            pass

    cmd =[sys .executable ,"-m","tensorboard.main","--logdir",logdir ,"--port",str (port )]
    process =subprocess .Popen (
    cmd ,
    stdout =subprocess .PIPE ,
    stderr =subprocess .PIPE ,
    creationflags =subprocess .CREATE_NO_WINDOW if sys .platform =='win32'else 0 ,
    )


    if check_tensorboard_running (port ,max_wait =30 ):
        try :
            if sys .platform =='win32':
                chrome_paths =[
                'C:/Program Files/Google/Chrome/Application/chrome.exe',
                'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
                os .path .expanduser ('~/AppData/Local/Google/Chrome/Application/chrome.exe'),
                ]
                chrome_opened =False 
                for chrome_path in chrome_paths :
                    if os .path .exists (chrome_path ):
                        subprocess .Popen ([chrome_path ,f"http://localhost:{port}"])
                        chrome_opened =True 
                        break 
                if not chrome_opened :
                    webbrowser .open (f"http://localhost:{port}")
            else :
                webbrowser .get ('google-chrome').open (f"http://localhost:{port}")
        except Exception as e :
            webbrowser .open (f"http://localhost:{port}")
    else :
        process .terminate ()
        sys .exit (1 )

    try :
        process .wait ()
    except KeyboardInterrupt :
        process .terminate ()
        process .wait ()
