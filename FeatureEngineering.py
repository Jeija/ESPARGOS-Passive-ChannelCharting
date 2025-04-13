#!/usr/bin/env python3

# Pre-compute features for Neural Networks in NumPy
# Used for both supervised baseline and Channel Charting

import multiprocessing as mp
from tqdm.auto import tqdm
import espargos_0007
import numpy as np
import CRAP

PROCESSES = min(8, mp.cpu_count() // 2)

TAP_START = espargos_0007.SUBCARRIER_COUNT // 2 - 4
TAP_STOP = espargos_0007.SUBCARRIER_COUNT // 2 + 8
cov_matrix_shape = espargos_0007.ROW_COUNT * espargos_0007.COL_COUNT
FEATURE_SHAPE = (espargos_0007.TX_COUNT, espargos_0007.ARRAY_COUNT, TAP_STOP - TAP_START, cov_matrix_shape, cov_matrix_shape)

def precompute_features(all_datasets):
    todo_queue = mp.Queue()
    output_queue = mp.Queue()
    
    def feature_engineering_worker(todo_queue, output_queue):
        while True:
            dataset_idx, cluster_idx, tx_idx = todo_queue.get()
    
            if dataset_idx == -1:
                output_queue.put((-1, -1, -1, -1))
                break
    
            csi_fdomain = all_datasets[dataset_idx]["clusters"][cluster_idx]["csi_freq_domain"][tx_idx]
            csi_fdomain_noclutter = CRAP.remove_clutter(csi_fdomain, all_datasets[dataset_idx]["clutter_acquisitions"][tx_idx])
            
            csi_tdomain = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(csi_fdomain_noclutter, axes = -1), axis = -1), axes = -1)[...,TAP_START:TAP_STOP]
            csi_tdomain_flat = np.reshape(csi_tdomain, (csi_tdomain.shape[0], espargos_0007.ARRAY_COUNT, cov_matrix_shape, TAP_STOP - TAP_START))
            output_queue.put((dataset_idx, cluster_idx, tx_idx, np.einsum("davt,dawt->atvw", np.conj(csi_tdomain_flat), csi_tdomain_flat)))
    
    total_tasks = 0
    for dataset_idx, dataset in enumerate(all_datasets):
        dataset["cluster_features"] = np.zeros((len(dataset["clusters"]),) + FEATURE_SHAPE, dtype = np.complex64)
        for cluster_idx, cluster in enumerate(dataset["clusters"]):
            for tx_idx in range(len(cluster["csi_freq_domain"])):
                total_tasks = total_tasks + 1
                todo_queue.put((dataset_idx, cluster_idx, tx_idx))
    
    print(f"Pre-computing training features for {total_tasks} datapoints in total")
    with tqdm(total = total_tasks) as pbar:
        for i in range(PROCESSES):
            todo_queue.put((-1, -1, -1))
            p = mp.Process(target = feature_engineering_worker, args = (todo_queue, output_queue))
            p.start()
        
        finished_processes = 0
        while finished_processes != PROCESSES:
            dataset_idx, cluster_idx, tx_idx, res = output_queue.get()
    
            if dataset_idx == -1:
                finished_processes = finished_processes + 1
            else:
                all_datasets[dataset_idx]["cluster_features"][cluster_idx, tx_idx] = res
                pbar.update(1)