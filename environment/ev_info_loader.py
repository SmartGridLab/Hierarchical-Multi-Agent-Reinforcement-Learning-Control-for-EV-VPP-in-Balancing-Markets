"""
ev_info_loader.py
=================
Loaders for EV arrival and profile information.

- `load_accurate_ev_info`: load EV profile data derived from real datasets
- `load_arrival_probabilities`: load time-dependent arrival probabilities

Inputs:
- Per-station arrival profile CSVs are read by `load_arrival_probabilities()`.
  Column 0 is the time-step key and column 1 is a non-negative arrival-intensity
  weight for that step.
- The arrival-SoC distribution CSV is read by `load_accurate_ev_info()`.
  Column 0 is an SoC support value and column 2 is its distribution weight.
- The EV profile CSV is read by `load_accurate_ev_info()`.
  Required columns are `ev_id`, `connection_minutes_5min`, and
  `required_soc_percent`.

Outputs:
- `load_arrival_probabilities()` returns a length-`episode_steps` NumPy array of
  Bernoulli arrival probabilities in `[0, 1]` for one station. The time pattern
  follows the CSV weights, while the average probability is controlled by
  `baseline_prob`.
- `load_accurate_ev_info()` returns profile arrays used by EVEnv when sampling
  an arriving EV's initial SoC, dwell time, required SoC, and original profile ID.

Sampling model:
- Station arrival profiles define when arrivals are likely to occur.
- The profile CSV defines paired dwell duration and required SoC, preserving the
  empirical relation between stay length and energy need.
- The arrival-SoC CSV defines the marginal initial-SoC distribution. It is
  converted to a CDF because EVEnv samples initial SoC by inverse transform.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass (frozen =True )
class AccurateEVInfo :
    """
    Immutable profile bundle consumed by EVEnv.

    `soc_values` and `soc_cdf` define the initial-SoC sampler. The profile arrays
    share the same row index, so a single random index picks a consistent EV ID,
    dwell duration, and required SoC from the source profile CSV.
    """
    soc_values :np .ndarray
    soc_cdf :np .ndarray
    profile_ev_ids :np .ndarray
    profile_dwell_units :np .ndarray
    profile_needed_soc :np .ndarray


def _read_numeric_column (path :str ,value_index :int )->Tuple [np .ndarray ,np .ndarray ]:
    """
    Read a CSV and return column 0 as integer keys and `value_index` as values.

    Rows that cannot be parsed numerically are ignored. This keeps header rows,
    blank lines, and annotation rows from affecting the empirical distributions.
    """
    file_path =Path (path )
    keys =[]
    vals =[]
    with file_path .open (newline ="",encoding ="utf-8-sig")as f :
        reader =csv .reader (f )
        header_skipped =False
        for row in reader :
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
    """Resize a non-empty array to `target_len` by tiling or trimming."""
    if len (arr )==target_len :
        return arr
    reps =int (np .ceil (target_len /len (arr )))
    tiled =np .tile (arr ,reps )[:target_len ]
    return tiled .astype (np .float64 )


def _load_arrival_probabilities (path :str ,episode_steps :int ,baseline_prob :float )->np .ndarray :
    """
    Load arrival-probability weights and normalize them to the desired baseline.

    If `w_t` is the non-negative CSV weight at step t, the returned probability
    is `clip(w_t * baseline_prob * episode_steps / sum(w), 0, 1)`. Thus the CSV
    controls relative timing, and `baseline_prob` controls average arrival rate.
    """
    _ ,weights =_read_numeric_column (path ,1 )
    if len (weights )==0 :
        raise ValueError (f"Arrival profile has no usable rows: {path}")
    weights =np .clip (weights ,a_min =0.0 ,a_max =None )
    weights =_ensure_length (weights ,episode_steps )
    total =float (weights .sum ())
    if total <=0 :
        raise ValueError (f"Arrival profile has zero total weight: {path}")
    scale =baseline_prob *episode_steps /total
    probs =np .clip (weights *scale ,0.0 ,1.0 )
    return probs


def load_arrival_probabilities (path :str ,episode_steps :int ,baseline_prob :float )->np .ndarray :
    """Load per-step arrival probabilities for one station."""
    return _load_arrival_probabilities (path ,episode_steps ,baseline_prob )


def _load_soc_cdf (path :str )->Tuple [np .ndarray ,np .ndarray ]:
    """
    Load the arrival-SoC distribution CSV and convert it into a CDF.

    The returned arrays implement inverse-transform sampling: EVEnv draws
    `u ~ Uniform(0, 1)` and selects `soc_values[searchsorted(cdf, u)]`.
    """
    soc_values ,percents =_read_numeric_column (path ,2 )
    if len (percents )==0 :
        raise ValueError (f"SoC arrival distribution has no usable rows: {path}")
    percents =np .clip (percents ,a_min =0.0 ,a_max =None )
    if float (percents .sum ())<=0 :
        raise ValueError (f"SoC arrival distribution has zero total weight: {path}")
    cdf =np .cumsum (percents )
    cdf /=cdf [-1 ]
    return soc_values ,cdf


def _load_ev_profile_csv (path :str )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """
    Load EV profile CSV columns: EV ID, dwell duration, and required SoC.

    Rows with malformed values are skipped. The file itself must still contain
    at least one usable EV profile row.
    """
    ev_ids =[]
    dwell_units =[]
    needed_soc =[]
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
    if not ev_ids :
        raise ValueError (f"EV profile CSV has no usable rows: {path}")
    return (
    np .asarray (ev_ids ,dtype =np .int64 ),
    np .asarray (dwell_units ,dtype =np .float64 ),
    np .asarray (needed_soc ,dtype =np .float64 ),
    )


def load_accurate_ev_info (
soc_arrival_path :str ,
profile_csv_path :str ,
)->AccurateEVInfo :
    """
    Load all EV-profile inputs and return them as an `AccurateEVInfo` object.

    Arrival probabilities are intentionally loaded separately per station by
    EVEnv because each station now has its own required arrival profile CSV.
    """
    soc_values ,soc_cdf =_load_soc_cdf (soc_arrival_path )
    profile_ev_ids ,profile_dwell_units ,profile_needed_soc =_load_ev_profile_csv (profile_csv_path )

    return AccurateEVInfo (
    soc_values =soc_values ,
    soc_cdf =soc_cdf ,
    profile_ev_ids =profile_ev_ids ,
    profile_dwell_units =profile_dwell_units ,
    profile_needed_soc =profile_needed_soc ,
    )
