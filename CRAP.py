#!/usr/bin/env python3

import numpy as np

# Implementation of the CRAP Algorithm as presented in
# M. Henninger, S. Mandelli, A. Grudnitsky, T. Wild, S. ten Brink: "CRAP: Clutter Removal with Acquisitions Under Phase Noise"
#
# Note that the original CRAP algorithm actually assumes empty channels, but we only have measurements with a target in the channel.

def acquire_clutter(csi_dataset, order = 5):
    """
    Aquire clutter channel estimate from CSI (for a single transmitter).
    order is the assumed clutter order.
    """
    C = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    R = np.einsum("na,nb->ab", C, np.conj(C), optimize = True)
    w, v = np.linalg.eigh(R)

    return v[:,::-1][:,:order]

def remove_clutter(csi_dataset, clutter_subspace):
    """
    Remove clutter from channel measurements (for a single transmitter).
    """
    h = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    clutter = clutter_subspace @ np.einsum("sl,ds->ld", np.conj(clutter_subspace), h)
    h_noclutter = h - np.transpose(clutter)

    return np.reshape(h_noclutter, csi_dataset.shape)