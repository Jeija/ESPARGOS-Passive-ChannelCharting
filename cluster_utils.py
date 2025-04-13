#!/usr/bin/env python3

######################
# Clustering Helpers #
######################

# Combine multiple datapoints from different transmitters into clusters of certain duration (e.g., cluster_interval = 1 second).
# Each cluster will contain datapoints from every transmitter.

from tqdm.auto import tqdm
import numpy as np

def cluster_dataset(dataset, cluster_interval = 1):
    dataset["clusters"] = []
    dataset["cluster_positions"] = []
    dataset["cluster_timestamps"] = []

    print(f"Clustering dataset {dataset["filename"]}")
    current_cluster = None

    def finish_cluster():
        nonlocal current_cluster

        current_cluster["last_timestamp"] = current_cluster["timestamps"][-1]
        current_cluster["mean_position"] = np.mean(current_cluster["groundtruth_positions"], axis = 0)
        # CSI to NumPy arrays
        for mac in current_cluster["csi_freq_domain"].keys():
            current_cluster["csi_freq_domain"][mac] = np.asarray(current_cluster["csi_freq_domain"][mac])
        current_cluster["csi_freq_domain"] = [csi for csi in current_cluster["csi_freq_domain"].values()]
        datapoint_count = np.asarray([csi.shape[0] for csi in current_cluster["csi_freq_domain"]])
        if np.any(datapoint_count == 0):
            print("Warning: Cluster has missing TX, ignoring:", datapoint_count)
        else:
            dataset["cluster_positions"].append(current_cluster["mean_position"])
            dataset["cluster_timestamps"].append(np.mean(current_cluster["timestamps"]))
            dataset["clusters"].append(current_cluster)
        current_cluster = None
    
    for d in tqdm(range(dataset["timestamps"].shape[0])):
        # Check if cluster is finished
        if current_cluster is not None:
            if dataset["timestamps"][d] > current_cluster["first_timestamp"] + cluster_interval:
                finish_cluster()
                
        # Need to start new cluster?
        if current_cluster is None:
            current_cluster = dict()
            current_cluster["first_timestamp"] = dataset["timestamps"][d]
            current_cluster["groundtruth_positions"] = []
            current_cluster["timestamps"] = []
            current_cluster["csi_freq_domain"] = dict()
            for mac in dataset["unique_macs"]:
                current_cluster["csi_freq_domain"][mac] = []

        # On every iteration: Add CSI, timestamp and position to cluster
        mac = dataset["source_macs"][d]
        current_cluster["csi_freq_domain"][mac].append(dataset["csi_freq_domain"][d])
        current_cluster["groundtruth_positions"].append(dataset["groundtruth_positions"][d])
        current_cluster["timestamps"].append(dataset["timestamps"][d])

    if current_cluster is not None:
        finish_cluster()

    dataset["cluster_positions"] = np.asarray(dataset["cluster_positions"])
    dataset["cluster_timestamps"] = np.asarray(dataset["cluster_timestamps"])