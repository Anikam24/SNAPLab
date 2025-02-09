import os
import random
import h5py
import numpy as np
import time

from utils import find_node_velocity, get_stats, fill_missing, graph_vels, nan_vals

defaultdir = '/gpfs/radev/pi/saxena/aj764'
rootdir = f'{defaultdir}/PairedTestingSessions/'

vid_subdirs = []
for subdir, dirs, files in os.walk(rootdir):
    if subdir.endswith("Videos"):
        vid_subdirs.append(subdir)
vid_subdirs = sorted(vid_subdirs)

single_vids = {}
multi_vids = {}
for vids in vid_subdirs:
    files = os.listdir(vids)
    cut_vids = vids[28:]
    single_vids[cut_vids] = []
    multi_vids[cut_vids] = []
    for file in files:
        if file.endswith('.mp4') and int(file[:2]) >= 4:
            KL_count = file.count('KL')
            EB_count = file.count('EB')
            HF_count = file.count('HF')
            if KL_count + EB_count + HF_count == 2:
                multi_vids[cut_vids].append(file)
            else:
                single_vids[cut_vids].append(file)

single_len_tot = 0
for key, value in single_vids.items():
    single_len_tot += len(value)
print(f'There are {single_len_tot} single instance videos')

multi_len_tot = 0
for key, value in multi_vids.items():
    multi_len_tot += len(value)
print(f'There are {multi_len_tot} multi instance videos')

CHECK = False

start_time = time.time()

total_intial_nan = 0
total_after_out_nan = 0
total_final_nan = 0

for i, session in enumerate(single_vids.keys()): 
    video_list = single_vids[session]
    analysis_path = defaultdir + '/' + session[:-6] + 'Tracking/h5/'
    
    for video in video_list:
        # open analysis file
        analysis_file = analysis_path + video[:-3] + 'predictions.h5'
        with h5py.File(analysis_file,'r+') as f:
            locations = f["tracks"][:].T 

            # find nan values
            intial = nan_vals(locations)

            # just to check you haven't already done this vid or it isn't empty
            if intial != 0 and video != '091924_Cam2_TrNum15_IS_KL005Y.mp4':
                # take out positional outliers
                all_vels = {}
                for node in range(locations.shape[1]):
                    # find the velocities
                    all_vels[node] = find_node_velocity(locations[:, node, :])
                
                    # get values need to find outliers
                    mean, std, low, high = get_stats(all_vels[node])
                
                    # if you want to check that these values looks good
                    graph_vels(all_vels[node], CHECK)
                
                    # replace outliers in locations with nan
                    nan_index = [i for i in range(len(all_vels[node])) if (all_vels[node][i] > high or all_vels[node][i] < low)]
                    for index in nan_index:
                        locations[index + 1, node, 0], locations[index + 1, node, 0] = np.nan, np.nan
                
                    # if you want to check that new locations look good
                    test_vels = find_node_velocity(locations[:, node, :])
                    graph_vels(test_vels, check=CHECK, old_low=low, old_high=high)
    
                # find nan values again
                after_out = nan_vals(locations)
    
                # fill in missing locations
                new_locations = fill_missing(locations)
                f["tracks"][:] = new_locations.T
    
                # finds nan values for final time
                after_fill = nan_vals(new_locations)
                
                total_intial_nan += intial
                total_after_out_nan += after_out
                total_final_nan += after_fill
    
                # if you want to check the nan/fill values for a each video
                if True:
                    print(f'video name: {video}')
                    print(f'intial nan: {round(intial, 2)} %, after out nan: {round(after_out, 2)} %, final nan: {round(after_fill, 2)} %')
                
    
print('totals:')
print(f'intial nan: {round(total_intial_nan / single_len_tot, 2)} %, after out nan: {round(total_after_out_nan / single_len_tot, 2)} %, final nan: {round(total_final_nan / single_len_tot, 2)} %')
print(f'time elapse: {time.time() - start_time}')


