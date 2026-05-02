"""
replay_buffer.py - experience replay buffer.

This is a circular replay buffer for off-policy algorithms such as MADDPG.
All core tensors are stored on the configured device for fast minibatch
sampling.

Design notes:
- Lazy initialization: shapes are inferred on the first `cache()` call.
- Temporary CPU staging: data can be accumulated before GPU tensors exist.
- Circular overwrite: `ptr % buf_size` overwrites the oldest entries.
"""
import torch
import numpy as np

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


class ReplayBuffer :
    def __init__ (self ,cap =int (1e5 )):
        """Initialize the replay buffer with a fixed capacity."""
        self .buf_size =cap

        self .s_dim =None
        self .a_dim =None
        self .n_agents =None
        self .max_evs =None

        self .ptr =0
        self .size =0

        self .s =None
        self .s2 =None
        self .a =None

        self .r_local =None
        self .r_global =None
        self .d =None
        self .actual_station_powers =None
        self .estimated_memory_gb =0.0

        self .temp_buf =[]

    def cache (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers =None ,actual_ev_soc_changes =None ):
        """
        Add one transition to the replay buffer.

        On the first call, the buffer infers tensor shapes, allocates device
        tensors, and flushes any temporarily staged entries.
        """
        if self .s is None :
            self .n_agents =len (s )
            self .s_dim =s [0 ].shape [0 ]if isinstance (s [0 ],np .ndarray )else len (s [0 ])

            if isinstance (a ,np .ndarray ):
                self .max_evs =a .shape [2 ]if a .ndim ==3 else a .shape [1 ]
            else :
                try :
                    if isinstance (a ,tuple )or isinstance (a ,list ):
                        if len (a )>0 and hasattr (a [0 ],'shape')and len (a [0 ].shape )>0 :
                            self .max_evs =a [0 ].shape [1 ]
                        else :
                            self .max_evs =5
                    elif isinstance (a ,torch .Tensor ):
                        self .max_evs =a .shape [1 ]if len (a .shape )>1 else 5
                    else :
                        self .max_evs =5
                except (IndexError ,AttributeError ):
                    self .max_evs =5

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
            self .actual_ev_soc_changes =torch .zeros ((self .buf_size ,self .n_agents ,self .max_evs ),
            dtype =torch .float32 ,device =device )

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
            self .temp_buf =[]

        if self .s is not None :
            self ._cache_to_tensor (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes )
        else :
            if actual_ev_soc_changes is not None :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes ))
            elif actual_station_powers is not None :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ))
            else :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ))

    def _cache_to_tensor (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers =None ,actual_ev_soc_changes =None ):
        """Write one transition into the current buffer slot."""
        idx =self .ptr

        if isinstance (s ,torch .Tensor ):
            if s .device ==device :
                self .s [idx ]=s .detach ()
            else :
                self .s [idx ]=s .detach ().to (device )
        else :
            self .s [idx ]=torch .tensor (s ,dtype =torch .float32 ,device =device )

        if isinstance (s2 ,torch .Tensor ):
            if s2 .device ==device :
                self .s2 [idx ]=s2 .detach ()
            else :
                self .s2 [idx ]=s2 .detach ().to (device )
        else :
            self .s2 [idx ]=torch .tensor (s2 ,dtype =torch .float32 ,device =device )

        if isinstance (a ,torch .Tensor ):
            if a .device ==device :
                self .a [idx ]=a .detach ()
            else :
                self .a [idx ]=a .detach ().to (device )
        else :
            self .a [idx ]=torch .tensor (a ,dtype =torch .float32 ,device =device )

        if isinstance (r_local ,torch .Tensor ):
            if r_local .device ==device :
                self .r_local [idx ]=r_local .detach ()
            else :
                self .r_local [idx ]=r_local .detach ().to (device )
        else :
            self .r_local [idx ]=torch .tensor (r_local ,dtype =torch .float32 ,device =device )

        if isinstance (r_global ,torch .Tensor ):
            if r_global .device ==device :
                self .r_global [idx ]=r_global .detach ().flatten ()[:1 ]
            else :
                self .r_global [idx ]=r_global .detach ().to (device ).flatten ()[:1 ]
        else :
            self .r_global [idx ]=torch .tensor ([r_global ],dtype =torch .float32 ,device =device )

        if isinstance (d ,torch .Tensor ):
            if d .device ==device :
                self .d [idx ]=d .detach ()
            else :
                self .d [idx ]=d .detach ().to (device )
        else :
            self .d [idx ]=torch .tensor (d ,dtype =torch .float32 ,device =device )

        if actual_station_powers is not None :
            if isinstance (actual_station_powers ,torch .Tensor ):
                if actual_station_powers .device ==device :
                    self .actual_station_powers [idx ]=actual_station_powers .detach ()
                else :
                    self .actual_station_powers [idx ]=actual_station_powers .detach ().to (device )
            else :
                self .actual_station_powers [idx ]=torch .tensor (actual_station_powers ,dtype =torch .float32 ,device =device )
        else :
            raise ValueError ("actual_station_powers must be provided. Environment should provide actual station powers after applying SoC constraints.")

        if actual_ev_soc_changes is not None :
            if isinstance (actual_ev_soc_changes ,torch .Tensor ):
                if actual_ev_soc_changes .device ==device :
                    self .actual_ev_soc_changes [idx ]=actual_ev_soc_changes .detach ()
                else :
                    self .actual_ev_soc_changes [idx ]=actual_ev_soc_changes .detach ().to (device )
            else :
                self .actual_ev_soc_changes [idx ]=torch .tensor (actual_ev_soc_changes ,dtype =torch .float32 ,device =device )
        else :
            self .actual_ev_soc_changes [idx ]=self .a [idx ].clone ()

        self .ptr =(self .ptr +1 )%self .buf_size
        self .size =min (self .size +1 ,self .buf_size )

    def sample (self ,batch ):
        """Sample a random minibatch directly from device tensors."""
        if self .s is None :
            raise RuntimeError ("ReplayBuffer is not initialized. cache() must be called to initialize GPU buffers before sampling.")

        idxs =torch .randint (0 ,self .size ,(batch ,),device =device )
        return self ._gather_batch (idxs )

    def _gather_batch (self ,idxs ):
        """Gather one batch given pre-selected indices (1-step semantics)."""
        s_batch =self .s [idxs ]
        s2_batch =self .s2 [idxs ]
        a_batch =self .a [idxs ]
        r_local_batch =self .r_local [idxs ]
        r_global_batch =self .r_global [idxs ]
        d_batch =self .d [idxs ]
        actual_station_powers_batch =self .actual_station_powers [idxs ]
        actual_ev_soc_changes_batch =self .actual_ev_soc_changes [idxs ]
        return tuple (t .to (device ,non_blocking =True )for t in (
        s_batch ,s2_batch ,a_batch ,r_local_batch ,r_global_batch ,d_batch ,
        actual_station_powers_batch ,actual_ev_soc_changes_batch ))

    def sample_with_nstep_global (self ,batch ,n_step ,gamma ):
        """
        Sample a minibatch and additionally compute n-step global rewards.

        Returns the standard 1-step batch plus three extra tensors for the
        global critic target:
            r_global_n  : sum_{k=0..k_eff-1} gamma^k * r_global[idx+k]
            s2_n        : s2[idx + k_eff - 1]                  (state after k_eff steps)
            d_n         : done flag at idx + k_eff - 1         (per-agent)
            n_eff       : effective n used per sample          (1..n_step)
        where k_eff = min(n_step, k of first done in window) so that bootstrap
        cleanly stops at episode boundaries. The local 1-step quantities
        (s2, r_local, d) are returned unchanged so the caller can use them
        for the local critic.

        Buffer wrap-around is avoided by restricting starting indices to
        positions whose chronological window of length n_step does not cross
        the write pointer.
        """
        if self .s is None :
            raise RuntimeError ("ReplayBuffer is not initialized.")
        n =max (1 ,int (n_step ))

        if self .size <self .buf_size :
            # Buffer not yet wrapped: chronological order matches index order.
            max_start =max (1 ,self .size -n +1 )
            idxs =torch .randint (0 ,max_start ,(batch ,),device =device )
        else :
            # Buffer wrapped. Forbidden start = last n-1 chronological positions
            # whose lookahead window would cross the write pointer (oldest).
            # Allowed offsets from `ptr`: [0, buf_size - n + 1).
            offsets =torch .randint (0 ,self .buf_size -n +1 ,(batch ,),device =device )
            idxs =(offsets +self .ptr )%self .buf_size

        # Standard 1-step batch (used for local critic and as base for global)
        std_batch =self ._gather_batch (idxs )
        s_b ,s2_b ,a_b ,r_local_b ,r_global_b ,d_b ,asp_b ,aevsc_b =std_batch

        # Compute n-step global return with done-truncation
        B =idxs .size (0 )
        r_n =torch .zeros ((B ,1 ),dtype =torch .float32 ,device =device )
        # Track if we've already passed an episode boundary (then stop accumulating)
        stopped =torch .zeros ((B ,1 ),dtype =torch .float32 ,device =device )
        n_eff =torch .ones ((B ,),dtype =torch .long ,device =device )
        s2_n =s2_b .clone ()
        d_n =d_b .clone ()
        discount =1.0

        for k in range (n ):
            cur_idx =(idxs +k )%self .buf_size
            r_k =self .r_global [cur_idx ]
            d_k_per_agent =self .d [cur_idx ]
            d_k_any =d_k_per_agent .max (dim =1 ,keepdim =True )[0 ]

            # Add this step's reward only if not yet stopped
            active =1.0 -stopped
            r_n =r_n +discount *r_k *active

            # Update state-after-window and done-flag-of-window for samples still active
            active_b =(active >0.5 ).squeeze (-1 )  # bool [B]
            if active_b .any ():
                s2_n [active_b ]=self .s2 [cur_idx [active_b ]]
                d_n [active_b ]=d_k_per_agent [active_b ]
                n_eff [active_b ]=k +1

            # Update stopped flag for next iteration
            stopped =torch .maximum (stopped ,d_k_any *active )
            discount *=gamma

        return (
        s_b ,s2_b ,a_b ,r_local_b ,r_global_b ,d_b ,asp_b ,aevsc_b ,
        # extra n-step augmentation for the global critic
        r_n ,s2_n ,d_n ,n_eff ,
        )
