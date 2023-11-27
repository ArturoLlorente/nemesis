import sys
sys.path.insert(0, '../olympus')
sys.path.insert(0, '../hyperion')
from multiprocessing import Pool
import jax
jax.config.update('jax_platform_name', 'cpu')
import pickle

def events_wrapped(indexes_all):    
    
    
    import pickle
    import os
    import functools
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.8"
    import sys
    sys.path.insert(0, '../olympus')
    sys.path.insert(0, '../hyperion')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from itertools import product
    import awkward as ak
    import pandas as pd
    from olympus.event_generation.photon_propagation.norm_flow_photons import make_generate_norm_flow_photons, make_nflow_photon_likelihood
    from olympus.event_generation.photon_propagation.utils import sources_to_model_input
    from nemesis.event_generation.detector import Detector, make_line, make_triang
    from olympus.event_generation.event_generation import (
        generate_cascade,
        generate_cascades,
        simulate_noise,
        generate_realistic_track,
        generate_realistic_tracks,
        generate_realistic_tracks_test,
        generate_realistic_starting_tracks,)
    from olympus.event_generation.lightyield import make_pointlike_cascade_source, make_realistic_cascade_source
    from olympus.event_generation.utils import sph_to_cart_jnp, proposal_setup

    #from olympus.plotting import plot_event
    from hyperion.medium import medium_collections
    from hyperion.constants import Constants
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    from jax import random
    from jax import numpy as jnp
    import json
    
    
    
    event_type = indexes_all[0]
    v_x = indexes_all[1]
    seed_start = indexes_all[2]

    path_to_config = "../hyperion/data/pone_config_optimistic.json"
    config = json.load(open(path_to_config))["photon_propagation"]
    ref_ix_f, sca_a_f, sca_l_f, _ = medium_collections[config["medium"]]

    def c_medium_f(wl):
        """Speed of light in medium for wl (nm)."""
        return Constants.BaseConstants.c_vac / ref_ix_f(wl)

    rng = np.random.RandomState(31338)
    oms_per_line = 20
    dist_z = 50 # m
    dark_noise_rate = 16 * 1e-5  # 1/ns
    side_len = 100 # m
    pmts_per_module = 16
    pmt_cath_area_r = 75E-3 / 2 # m
    module_radius = 0.21 # m

    efficiency = pmts_per_module * (pmt_cath_area_r)**2 * np.pi / (4*np.pi*module_radius**2)
    det = make_triang(side_len, oms_per_line, dist_z, dark_noise_rate, rng, efficiency=efficiency, v_x=v_x, buoy_weight=30)

    gen_ph = make_generate_norm_flow_photons(
        "../hyperion/data/photon_arrival_time_nflow_params.pickle",
        "../hyperion/data/photon_arrival_time_counts_params.pickle",
        c_medium=c_medium_f(700) / 1E9

    )
    
    if event_type == 'starting_tracks':
        prop = proposal_setup()
        events = generate_realistic_starting_tracks(
            det,
            cylinder_height=det._outer_cylinder[1] + 100,
            cylinder_radius=det._outer_cylinder[0] + 50,
            log_emin=3.5,
            log_emax=6.5,
            nsamples=100,
            seed=seed_start,
            pprop_func=gen_ph,
            proposal_prop=prop

        )
        
    elif event_type == 'tracks':
    
        prop = proposal_setup()
        events = generate_realistic_tracks(
            det,
            cylinder_height=det._outer_cylinder[1] + 100,
            cylinder_radius=det._outer_cylinder[0] + 50,
            log_emin=3.5,
            log_emax=6.5,
            nsamples=100,
            seed=seed_start,
            pprop_func=gen_ph,
            proposal_prop=prop
        )
        
    elif event_type == 'cascades':
    
        events = generate_cascades(
            det,
            cylinder_height=det._outer_cylinder[1] + 1100,
            cylinder_radius=det._outer_cylinder[0] + 50,
            log_emin=3.5,
            log_emax=6.5,
            particle_id=11,
            nsamples=100,
            seed=seed_start,
            converter_func=functools.partial(make_realistic_cascade_source, moliere_rand=True, resolution=0.2),
            pprop_func=gen_ph,
            noise_function=None

        )
    else:
        raise ValueError('Not a valid element type')
        
        
    return events