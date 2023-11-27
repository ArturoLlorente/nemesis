from multiprocessing import Pool
import pickle
import numpy as np
import os
from tqdm.auto import tqdm

def load_test_events_multi(v_x):
    path_to_events = "/dss/pone/pone_events"
    det, cascades, cascade_records = pickle.load(open(os.path.join(path_to_events, f"cascades1500_vx{v_x}.pickle"), "rb"))
    det, tracks, track_records = pickle.load(open(os.path.join(path_to_events, f"tracks1500_vx{v_x}.pickle"), "rb"))
    det, stracks, strack_records = pickle.load(open(os.path.join(path_to_events, f"starting_tracks1500_vx{v_x}.pickle"), "rb"))
    
    all_ev = [det, cascades, cascade_records, tracks, track_records, stracks, strack_records]
    
    return all_ev
    
    
def load_multi_test():
    cascades_test, cascade_records_test = [], []
    tracks_test, track_records_test = [], []
    stracks_test, strack_records_test = [], []
    det_test = []
    v_x_all = []

    for it in range(1,21):
        v_x_all.append(np.round(it*0.005, 3))

    for i in range(2):
        v_x = v_x_all[(i*10):((i+1)*10)]
        if __name__ == '__main__':
            with Pool(10) as p:
                all_ev = p.map(load_test_events_multi, v_x)
        print(f"iteration {i} done")


        for v in range(len(all_ev)):
            det_test.append(all_ev[v][0])
            cascades_test += all_ev[v][1]
            cascade_records_test += all_ev[v][2]
            tracks_test += all_ev[v][3]
            track_records_test += all_ev[v][4]    
            stracks_test += all_ev[v][5]
            strack_records_test += all_ev[v][6]
            
    return det_test, cascades_test, cascade_records_test, tracks_test, track_records_test, stracks_test, strack_records_test

def load_test_events(v_x):
    path_to_events = "/dss/pone/pone_events"
    det, cascades, cascade_records = pickle.load(open(os.path.join(path_to_events, f"cascades1500_vx{v_x}.pickle"), "rb"))
    det, tracks, track_records = pickle.load(open(os.path.join(path_to_events, f"tracks1500_vx{v_x}.pickle"), "rb"))
    det, stracks, strack_records = pickle.load(open(os.path.join(path_to_events, f"starting_tracks1500_vx{v_x}.pickle"), "rb"))
    
    return det, cascades, cascade_records, tracks, track_records, stracks, strack_records
    
def load_test():
    cascades_test, cascade_records_test = [], []
    tracks_test, track_records_test = [], []
    stracks_test, strack_records_test = [], []
    det_test = []
    v_x_all = []
    for it in range(1,21):
        v_x_all.append(np.round(it*0.005, 3))

    pbar = tqdm(total=len(v_x_all))
    for v_x in v_x_all:
        det, cascades, cascade_records, tracks, track_records, stracks, strack_records = load_test_events(v_x)
        det_test.append(det)
        cascades_test += cascades
        cascade_records_test += cascade_records
        tracks_test += tracks
        track_records_test += track_records
        stracks_test += stracks
        strack_records_test += strack_records
        pbar.update()    
    
    return det_test, cascades_test, cascade_records_test, tracks_test, track_records_test, stracks_test, strack_records_test
