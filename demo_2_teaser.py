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


import numpy as np
import open3d as o3d
from core import config

from core.utils import compute_transformation_diff, draw_registration_result, \
                       keypoints_to_spheres
from core.teaserpp_utils import find_mutually_nn_keypoints, \
                                execute_teaser_global_registration

try:
    reg = o3d.pipelines.registration
except AttributeError:
    reg = o3d.registration

FRAG1_COLOR = [1.0, 0.75, 0.0]
FRAG2_COLOR = [0, 0.629, 0.9]
SPHERE_COLOR_1 = [0,1,0.1]
SPHERE_COLOR_2 = [1, 0.3, 0.05]
NOISE_BOUND = 0.01
N_OUTLIERS = 20
OUTLIER_TRANSLATION_LB = 0.001
OUTLIER_TRANSLATION_UB = 0.05



def main(config_arguments):
    # Run the input parametrization
    point_cloud_files = ["./data/bunny/bun_zipper_res3.ply", "./data/bunny/bun_zipper_res3_noise.ply"]
    keypoints_files = ["./data/bunny/bun_zipper_res3.ply_keypoints", "./data/bunny/bun_zipper_res3_noise.ply_keypoints"]
    T = np.array(
        [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
        [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
        [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
        [0, 0, 0, 1]])

    # Load reference point cloud
    reference_pc = o3d.io.read_point_cloud(point_cloud_files[0])
    # Create test point cloud by transforming and inducing noise
    test_pc = o3d.io.read_point_cloud(point_cloud_files[1])

    voxel_grid = 0.01 # Half size of the voxel grid in the unit of the point cloud. Defaults to 0.15.
    n_voxels = 16 # Number of voxels in a side of the grid. Whole grid is nxnxn. Defaults to 16.
    gaussian_width = 0.01 # Width of the Gaussia kernel used for smoothing. Defaults to 1.75.
    # Load the descriptors and estimate the transformation parameters using RANSAC
    reference_desc = np.load('./data/bunny/32_dim/bun_zipper_res3.ply_{:.6f}_{}_{:.6f}_3DSmoothNet.npz'.format(voxel_grid, n_voxels, gaussian_width))
    reference_desc = reference_desc['data']

    test_desc = np.load('./data/bunny/32_dim/bun_zipper_res3_noise.ply_{:.6f}_{}_{:.6f}_3DSmoothNet.npz'.format(voxel_grid, n_voxels, gaussian_width))
    test_desc = test_desc['data']

    # Save as open3d feature 
    ref = reg.Feature()
    ref.data = reference_desc.T

    test = reg.Feature()
    test.data = test_desc.T

    indices_ref = np.genfromtxt(keypoints_files[0])
    indices_test = np.genfromtxt(keypoints_files[1])

    reference_pc_keypoints = np.asarray(reference_pc.points)[indices_ref.astype(int),:]
    test_pc_keypoints = np.asarray(test_pc.points)[indices_test.astype(int),:]

    # Save ad open3d point clouds
    ref_key = o3d.geometry.PointCloud()
    ref_key.points = o3d.utility.Vector3dVector(reference_pc_keypoints)

    test_key = o3d.geometry.PointCloud()
    test_key.points = o3d.utility.Vector3dVector(test_pc_keypoints)

    # First plot the original state of the point clouds
    reference_pc.estimate_normals()
    reference_pc.paint_uniform_color(FRAG1_COLOR)
    test_pc.estimate_normals()
    test_pc.paint_uniform_color(FRAG2_COLOR)
    o3d.visualization.draw_geometries([reference_pc, test_pc, keypoints_to_spheres(ref_key, SPHERE_COLOR_1), keypoints_to_spheres(test_key, SPHERE_COLOR_2)])


    ref_matched_key, test_matched_key = find_mutually_nn_keypoints(
        ref_key, test_key, ref, test
    )
    ref_matched_key = np.squeeze(ref_matched_key)
    test_matched_key = np.squeeze(test_matched_key)

    est_mat, max_clique, dt = execute_teaser_global_registration(ref_matched_key, test_matched_key, config_arguments)
    print("\nTEASER++ #correspondences:", len(ref_matched_key))
    print("TEASER++ transformation:\n", est_mat)
    rot_error, trans_error = compute_transformation_diff(est_mat, T)
    print("TEASER++ Rotation error (deg): ", rot_error)
    print("TEASER++ Translation error (m): ", trans_error)
    # Plot point clouds after registration
    draw_registration_result(reference_pc, test_pc,
                est_mat)
    

if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)
