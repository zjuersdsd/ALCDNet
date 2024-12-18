import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import pickle
import os
import scipy.io as sio
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch



def compute_PR(pair_dist, poses, map_tree_poses,distance_thr,is_distance=True, ignore_last=False):
    real_loop = []
    detected_loop = []
    distances = []
    last = poses.shape[0]
    if ignore_last:
        last = last-1

    for i in tqdm(range(100, last)):
        min_range = max(0, i-99)
        current_pose = poses[i][:2, 3]
        indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), distance_thr)
        valid_idxs = list(set(indices[0]) - set(range(min_range, last)))
        if len(valid_idxs) > 0:
            real_loop.append(1)

        else:
            real_loop.append(0)

        if is_distance:
            candidate = pair_dist[i, :i-99].argmin()
            detected_loop.append(-pair_dist[i, candidate].cpu().numpy())
        else:
            candidate = pair_dist[i, :i-99].argmax()
            detected_loop.append(pair_dist[i, candidate].cpu().numpy())
        candidate_pose = poses[candidate][:2, 3]
        distances.append(np.linalg.norm(candidate_pose-current_pose))
    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)
    print(real_loop)
    precision_fn = []
    recall_fn = []
    for thr in np.unique(detected_loop):

        asd = detected_loop <= thr
        asd = asd & real_loop
        asd = asd & (distances <= distance_thr)
        tp = asd.sum()
        fn = (detected_loop <= thr) & (distances > distance_thr) & real_loop
        fn2 = (detected_loop > thr) & real_loop
        fn = fn.sum() + fn2.sum()
        fp = (detected_loop <= thr) & (distances > distance_thr) & (1 - real_loop)
        fp = fp.sum()
        if (tp+fp) > 0:
            precision_fn.append(tp/(tp+fp))
        else:
            precision_fn.append(1.)
        recall_fn.append(tp/(tp+fn))
    precision_fp = []
    recall_fp = []
    for thr in np.unique(detected_loop):
        asd = detected_loop <= thr
        asd = asd & real_loop
        asd = asd & (distances <= distance_thr)
        tp = asd.sum()
        fp = (detected_loop <= thr) & (distances > distance_thr)
        fp = fp.sum()
        fn = (detected_loop > thr) & (real_loop)
        fn = fn.sum()
        if (tp+fp) > 0:
            precision_fp.append(tp/(tp+fp))
        else:
            precision_fp.append(1.)
        recall_fp.append(tp/(tp+fn))

    return precision_fn, recall_fn, precision_fp, recall_fp


def compute_PR_pairs(pair_dist, poses, is_distance=True, ignore_last=False, positive_range=4, ignore_below=-1):
    real_loop = []
    detected_loop = []
    last = poses.shape[0]
    if ignore_last:
        last = last-1
    for i in tqdm(range(100, last)):
        current_pose = poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= positive_range:
                real_loop.append(1)
            elif dist_pose <= ignore_below:
                continue
            else:
                real_loop.append(0)
            if is_distance:
                detected_loop.append(-pair_dist[i, j])
            else:
                detected_loop.append(pair_dist[i, j])
    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)

    return precision, recall


def compute_AP(precision, recall):
    ap = 0.
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i-1])*precision[i]
    return ap
