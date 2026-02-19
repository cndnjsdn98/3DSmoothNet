# demo.py ---
#
# Filename: demo.py
# Description: Demo of the 3DSmoothNet pipeline. 
# Comment: Some functions adapated from the open3d library http://www.open3d.org/
#
# Author: Gojcic Zan, Zhou Caifa
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Paper: https://arxiv.org/abs/1811.06879
# Created: 03.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:


import tensorflow as tf
import copy
import numpy as np
import os
import subprocess
import open3d as o3d
from open3d import *

FRAG1_COLOR = [1.0, 0.75, 0.0]
FRAG2_COLOR = [0, 0.629, 0.9]
SPHERE_COLOR_1 = [0,1,0.1]
SPHERE_COLOR_2 = [1, 0.3, 0.05]


def keypoints_to_spheres(keypoints, sphere_color):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color(sphere_color)
    return spheres

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color(FRAG1_COLOR)
    target_temp.paint_uniform_color(FRAG2_COLOR)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def ransac_from_correspondences(src_pc, tgt_pc, corres, distance_threshold):
    return o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pc, tgt_pc,
        corres,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 0.999),
    )

def execute_global_registration(
        source_down, target_down, reference_desc, target_desc, distance_threshold):

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, reference_desc, target_desc,
            False,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def l2_normalize_feature_inplace(feat):
    X = np.asarray(feat.data)          # (dim, N)
    n = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
    feat.data = X / n

def correspondences_ratio_test(src_feat: o3d.pipelines.registration.Feature,
                              tgt_feat: o3d.pipelines.registration.Feature,
                              ratio: float = 0.85):
    # src_feat.data: (D, Ns), tgt_feat.data: (D, Nt)
    X = np.asarray(src_feat.data).T  # (Ns, D)
    Y = np.asarray(tgt_feat.data).T  # (Nt, D)

    # Pairwise squared distances (Ns x Nt) -- with Ns=Nt=1000, this is fine
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)

    # 1st and 2nd nearest neighbor distances in target for each source
    idx_part = np.argpartition(d2, kth=1, axis=1)[:, :2]   # (Ns, 2) unordered
    d2_part = np.take_along_axis(d2, idx_part, axis=1)     # (Ns, 2)

    # order them so [:,0] is best
    order = np.argsort(d2_part, axis=1)
    nn1 = idx_part[np.arange(idx_part.shape[0]), order[:, 0]]
    nn2 = idx_part[np.arange(idx_part.shape[0]), order[:, 1]]
    d1 = d2_part[np.arange(d2_part.shape[0]), order[:, 0]]
    d2b = d2_part[np.arange(d2_part.shape[0]), order[:, 1]]

    keep = d1 < (ratio * ratio) * (d2b + 1e-12)

    corres = np.stack([np.arange(X.shape[0]), nn1], axis=1)[keep]
    return o3d.utility.Vector2iVector(corres.astype(np.int32)), int(keep.sum())

# Run the input parametrization
point_cloud_files = ["./data/demo/cloud_bin_0.ply", "./data/demo/cloud_bin_1.ply"]
keypoints_files = ["./data/demo/cloud_bin_0_keypoints.txt", "./data/demo/cloud_bin_1_keypoints.txt"]



for i in range(0,len(point_cloud_files)):
    args = "./3DSmoothNet -f " + point_cloud_files[i] + " -k " + keypoints_files[i] +  " -o ./data/demo/sdv/"
    subprocess.call(args, shell=True)

print('Input parametrization complete. Start inference')


# Run the inference as shell 
args = "python main_cnn.py --run_mode=test --evaluate_input_folder=./data/demo/sdv/  --evaluate_output_folder=./data/demo"
subprocess.call(args, shell=True)

print('Inference completed perform nearest neighbor search and registration')


# Load the descriptors and estimate the transformation parameters using RANSAC
reference_desc = np.load('./data/demo/32_dim/cloud_bin_0.ply_0.150000_16_1.750000_3DSmoothNet.npz')
reference_desc = reference_desc['data']


test_desc = np.load('./data/demo/32_dim/cloud_bin_1.ply_0.150000_16_1.750000_3DSmoothNet.npz')
test_desc = test_desc['data']

# Save as open3d feature 
ref = open3d.pipelines.registration.Feature()
ref.data = reference_desc.T

test = open3d.pipelines.registration.Feature()
test.data = test_desc.T

# Load point cloud and extract the keypoints
reference_pc = o3d.io.read_point_cloud(point_cloud_files[0])
test_pc = o3d.io.read_point_cloud(point_cloud_files[1])

indices_ref = np.genfromtxt(keypoints_files[0])
indices_test = np.genfromtxt(keypoints_files[1])

reference_pc_keypoints = np.asarray(reference_pc.points)[indices_ref.astype(int),:]
test_pc_keypoints = np.asarray(test_pc.points)[indices_test.astype(int),:]

print(indices_ref[0])
print(np.asarray(reference_pc.points)[indices_ref[0].astype(int), :])
print(reference_pc_keypoints[0])

# Save ad open3d point clouds
ref_key = geometry.PointCloud()
ref_key.points = utility.Vector3dVector(reference_pc_keypoints)

test_key = geometry.PointCloud()
test_key.points = utility.Vector3dVector(test_pc_keypoints)

print(ref_key)
print(ref)
print(test_key)
print(test)

reference_pc.estimate_normals()
reference_pc.paint_uniform_color(FRAG1_COLOR)
test_pc.estimate_normals()
test_pc.paint_uniform_color(FRAG2_COLOR)

o3d.visualization.draw_geometries([reference_pc, test_pc, keypoints_to_spheres(ref_key, SPHERE_COLOR_1), keypoints_to_spheres(test_key, SPHERE_COLOR_2)], front=[0, 0, -1.0])

corr = o3d.pipelines.registration.correspondences_from_features(
    ref, test, mutual_filter=False
)
corr = np.asarray(corr)
print("NN correspondences:", corr.shape) 

result_ransac = execute_global_registration(ref_key, test_key,
            ref, test, 0.05)
print("RANSAC fitness:", result_ransac.fitness)
print("RANSAC inlier_rmse:", result_ransac.inlier_rmse)
print("RANSAC #correspondences:", len(result_ransac.correspondence_set))
print("RANSAC transformation:\n", result_ransac.transformation)

l2_normalize_feature_inplace(ref)
l2_normalize_feature_inplace(test)
corres, n_keep = correspondences_ratio_test(ref, test, ratio=0.85)
print("kept correspondences after ratio test:", n_keep)

# IMPORTANT: for debugging, make this BIGGER than you think
result_ransac = ransac_from_correspondences(ref_key, test_key, corres, 0.05)
print("RANSAC fitness:", result_ransac.fitness)
print("RANSAC inlier_rmse:", result_ransac.inlier_rmse)
print("RANSAC #correspondences:", len(result_ransac.correspondence_set))
print("RANSAC transformation:\n", result_ransac.transformation)

# First plot the original state of the point clouds
# draw_registration_result(reference_pc, test_pc, np.identity(4))


# Plot point clouds after registration
draw_registration_result(reference_pc, test_pc,
            result_ransac.transformation)


