#!/usr/bin/env python3
# Helpers for loading espargos-0007 dataset

from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import hashlib
import json
import os

DATASET_DIR = "espargos-0007"
MAC_PREFIX = bytes([0x0a, 0xee, 0xf5])

###################################
# Load metadata (array positions) #
###################################
array_positions = []
array_normalvectors = []
array_upvectors = []
array_rightvectors = []

with open(os.path.join(DATASET_DIR, "spec.json")) as specfile:
    spec = json.load(specfile)
    for antenna in spec["antennas"]:
        array_positions.append(np.asarray(antenna["location"]))
        array_upvectors.append(np.asarray(antenna["upvector"]))
        array_rightvectors.append(np.asarray(antenna["rightvector"]))

        normalvector = np.cross(np.asarray(antenna["rightvector"]), np.asarray(antenna["upvector"]))
        normalvector = normalvector / np.linalg.norm(normalvector)
        array_normalvectors.append(normalvector)

array_positions = np.asarray(array_positions)
array_normalvectors = np.asarray(array_normalvectors)
array_upvectors = np.asarray(array_upvectors)
array_rightvectors = np.asarray(array_rightvectors)
centroid = np.mean(array_positions, axis = 0)

ARRAY_COUNT = len(array_positions)
TX_COUNT = 4
ROW_COUNT = 2
COL_COUNT = 4
SUBCARRIER_COUNT = 53

###############################
# Training / Test Set Loaders #
###############################
feature_description = {
    "csi": tf.io.FixedLenFeature([], tf.string, default_value = ''),
    "time": tf.io.FixedLenFeature([], tf.string, default_value = ''),
    "rssi": tf.io.FixedLenFeature([], tf.string, default_value = ''),
    "mac": tf.io.FixedLenFeature([], tf.string, default_value = ''),
    "pos": tf.io.FixedLenFeature([], tf.string, default_value = '')
}

def record_parse_function(proto):
    record = tf.io.parse_single_example(proto, feature_description)

    csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type = tf.complex64), (ARRAY_COUNT, ROW_COUNT, COL_COUNT, SUBCARRIER_COUNT))
    time = tf.ensure_shape(tf.io.parse_tensor(record["time"], out_type = tf.float64), ())
    rssi = tf.ensure_shape(tf.io.parse_tensor(record["rssi"], out_type = tf.float32), (4, 2, 4))
    pos = tf.ensure_shape(tf.io.parse_tensor(record["pos"], out_type = tf.float64), (3))
    source_mac = tf.ensure_shape(tf.io.parse_tensor(record["mac"], out_type = tf.string), ())

    return csi, pos, time, rssi, source_mac

def mac_filter(csi, pos, time, rssi, source_mac):
    return tf.strings.substr(source_mac, 0, 3) == MAC_PREFIX

def weight_csi_with_rssi(csi, pos, time, rssi, source_mac):
    csi = tf.cast((10**(rssi / 20))[:,:,:,tf.newaxis], tf.complex64) * csi
    return csi, pos, time, rssi, source_mac

def load_dataset(files):
    datasets = list()

    for filename in tqdm(files):
        print(f"Loading {filename}")

        tf_dataset = tf.data.TFRecordDataset([filename])
        tf_dataset = tf_dataset.map(record_parse_function)
        tf_dataset = tf_dataset.filter(mac_filter)
        tf_dataset = tf_dataset.map(weight_csi_with_rssi)

        dataset = dict()
        csi_freq_domain = []
        timestamps = []
        groundtruth_positions = []
        rssis = []
        source_macs = []
        unique_macs = set()
        BATCHSIZE = 1000

        for csi, pos, time, rssi, source_mac in tf_dataset.batch(BATCHSIZE):
            csi_fdomain = np.fft.ifftshift(csi.numpy(), axes = -1)
        
            csi_freq_domain.append(csi_fdomain)
            timestamps.append(time.numpy())
            rssis.append(rssi.numpy())
            groundtruth_positions.append(pos.numpy())
            for mac in source_mac.numpy():
                macstr = ":".join([f"{b:02x}" for b in mac])
                unique_macs.add(macstr)
                source_macs.append(macstr)

        csi_freq_domain = np.concatenate(csi_freq_domain)
        timestamps = np.concatenate(timestamps)
        groundtruth_positions = np.concatenate(groundtruth_positions)
        rssis = np.concatenate(rssis)

        dataset["filename"] = filename
        dataset["csi_freq_domain"] = csi_freq_domain
        dataset["timestamps"] = timestamps
        dataset["groundtruth_positions"] = groundtruth_positions
        dataset["rssis"] = rssis
        dataset["source_macs"] = source_macs
        assert(len(unique_macs) == TX_COUNT)
        dataset["unique_macs"] = sorted(unique_macs)

        datasets.append(dataset)

    return datasets

def hash_dataset_names(datasets):
    filenames = [ds["filename"] for ds in datasets]
    dataset_path_hash = hashlib.sha1(":".join(filenames).encode()).hexdigest()
    return dataset_path_hash

TRAINING_SET_ROBOT_FILES = [
    os.path.join(DATASET_DIR, "espargos-0007-meanders-nw-se-1.tfrecords"),
    os.path.join(DATASET_DIR, "espargos-0007-meanders-sw-ne-1.tfrecords"),
    os.path.join(DATASET_DIR, "espargos-0007-randomwalk-1.tfrecords")
]

TEST_SET_ROBOT_FILES = [
    os.path.join(DATASET_DIR, "espargos-0007-randomwalk-2.tfrecords")
]

TEST_SET_HUMAN_FILES = [
    os.path.join(DATASET_DIR, "espargos-0007-human-helmet-randomwalk-1.tfrecords")
]


