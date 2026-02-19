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
from sklearn.neighbors import KDTree
import time
from timeit import default_timer as timer
import math
import teaserpp_python


FRAG1_COLOR = [1.0, 0.75, 0.0]
FRAG2_COLOR = [0, 0.629, 0.9]
SPHERE_COLOR_1 = [0,1,0.1]
SPHERE_COLOR_2 = [1, 0.3, 0.05]
NOISE_BOUND = 0.01
N_OUTLIERS = 20
OUTLIER_TRANSLATION_LB = 0.001
OUTLIER_TRANSLATION_UB = 0.05

def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints 

    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]
    print("\n---------")
    print("n samples: " + str(n_samples))
    print(ref_features)
    print(test_features)
    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test_features)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    print("Number of matched keypoints: " + str(len(ref_match_idx)))

    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)


def execute_teaser_global_registration(source, target):
    """
    Use TEASER++ to perform global registration
    """
    # Prepare TEASER++ Solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve with TEASER++
    start = timer()
    teaserpp_solver.solve(source, target)
    end = timer()
    est_solution = teaserpp_solver.getSolution()
    print(est_solution)
    est_mat = compose_mat4_from_teaserpp_solution(est_solution)
    max_clique = teaserpp_solver.getTranslationInliersMap()
    print("Max clique size:", len(max_clique))
    final_inliers = teaserpp_solver.getTranslationInliers()
    return est_mat, max_clique, end - start

def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M


def get_angular_error(R_gt, R_est):
    """
    Get angular error
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est))-1) / 2.0
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        rotError = math.fabs(math.acos(A));
        return math.degrees(rotError)
    except ValueError:
        import pdb; pdb.set_trace()
        return 99999
    
def compute_transformation_diff(est_mat, gt_mat):
    """
    Compute difference between two 4-by-4 SE3 transformation matrix
    """
    R_gt = gt_mat[:3,:3]
    R_est = est_mat[:3,:3]
    rot_error = get_angular_error(R_gt, R_est)

    t_gt = gt_mat[:,-1]
    t_est = est_mat[:,-1]
    trans_error = np.linalg.norm(t_gt - t_est)

    return rot_error, trans_error

def estimate_spacing(pcd, k=2):
    # k=2 so the nearest neighbor excluding itself is returned
    dists = pcd.compute_nearest_neighbor_distance()
    return float(np.median(dists))

def iss_keypoints_to_indices(pcd: o3d.geometry.PointCloud,
                             iss_kp: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Map ISS keypoint XYZ locations to nearest-neighbor indices in the original pcd.
    Returns int array of shape [K].
    """
    kpts = np.asarray(iss_kp.points)
    if kpts.size == 0:
        raise ValueError("No ISS keypoints found. Tune ISS parameters or check the point cloud.")

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    idxs = np.empty((kpts.shape[0],), dtype=np.int64)
    for i, xyz in enumerate(kpts):
        _, nn_idx, _ = kdtree.search_knn_vector_3d(xyz, 1)
        idxs[i] = nn_idx[0]

    # remove duplicates (ISS points can map to same nearest input point)
    idxs = np.unique(idxs)
    return idxs

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
        ransac_n=8,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 500),
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

def main():
    # Run the input parametrization
    point_cloud_files = ["./data/bunny/bun_zipper_res3.ply", "./data/bunny/bun_zipper_res3_noise.ply"]
    keypoints_files = ["./data/bunny/bun_zipper_res3.ply_keypoints", "./data/bunny/bun_zipper_res3_noise.ply_keypoints"]

    # Load reference point cloud
    reference_pc = o3d.io.read_point_cloud(point_cloud_files[0])
    # Create test point cloud by transforming and inducing noise
    test_pc = o3d.io.read_point_cloud(point_cloud_files[0])
    # Apply arbitrary scale, translation and rotation
    T = np.array(
        [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
        [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
        [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
        [0, 0, 0, 1]])

    test_pc.transform(T)
    test = np.transpose(np.asarray(test_pc.points))
    # Add some noise
    # N = np.transpose(np.asarray(reference_pc.points)).shape[1]
    # test += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND
    # # Add some outliers
    # outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    # for i in range(outlier_indices.size):
    #     shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
    #     test[:, outlier_indices[i]] += shift.squeeze()
    test_pc = geometry.PointCloud()
    test_pc.points = utility.Vector3dVector(test.T)

    # Save test point cloud
    if not test_pc.has_normals():
        test_pc.estimate_normals()
    ok = o3d.io.write_point_cloud(
        point_cloud_files[1],
        test_pc,
        write_ascii=False,        # False = binary (smaller/faster), True = ASCII
        compressed=False,         # True can reduce size for some formats
        print_progress=True
    )
    if not ok:
        raise RuntimeError("Failed to write point cloud to output.ply")

    # Compute key points
    spacing = estimate_spacing(reference_pc)
    print("median spacing:", spacing)
    salient = 6.0 * spacing
    nonmax  = 2.0 * spacing

    tic = time.time()
    reference_key = o3d.geometry.keypoint.compute_iss_keypoints(
        reference_pc,
        salient_radius=salient,
        non_max_radius=nonmax,
        gamma_21=0.975,
        gamma_32=0.975,
        min_neighbors=5
    )
    test_key = o3d.geometry.keypoint.compute_iss_keypoints(
        test_pc,
        salient_radius=salient,
        non_max_radius=nonmax,
        gamma_21=0.975,
        gamma_32=0.975,
        min_neighbors=5
    )
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))
    print("Reference keypoints: ")
    print(reference_key)
    print("Test keypoints: ")
    print(test_key)

    reference_key_idx = iss_keypoints_to_indices(reference_pc, reference_key)
    test_key_idx = iss_keypoints_to_indices(test_pc, test_key)

    with open(keypoints_files[0], 'w') as f:
        f.write('\n'.join(str(idx) for idx in reference_key_idx))
    with open(keypoints_files[1], 'w') as f:
        f.write('\n'.join(str(idx) for idx in test_key_idx))

    # Clear evaluate folders
    args = "rm ./data/bunny/sdv/*.csv ./data/bunny/32_dim/*"
    subprocess.call(args, shell=True)

    voxel_grid = 0.01 # Half size of the voxel grid in the unit of the point cloud. Defaults to 0.15.
    n_voxels = 16 # Number of voxels in a side of the grid. Whole grid is nxnxn. Defaults to 16.
    gaussian_width = 0.01 # Width of the Gaussia kernel used for smoothing. Defaults to 1.75.
    for i in range(0,len(point_cloud_files)):
        args = "./3DSmoothNet -f " + point_cloud_files[i] + " -k " + keypoints_files[i] + " -o ./data/bunny/sdv/ -r " + str(voxel_grid) + " -n " + str(n_voxels) + " -h " + str(gaussian_width)
        subprocess.call(args, shell=True)

    print('Input parametrization complete. Start inference')


    # Run the inference as shell 
    args = "python main_cnn.py --run_mode=test --evaluate_input_folder=./data/bunny/sdv/  --evaluate_output_folder=./data/bunny --input_dim=" + str(n_voxels**3)
    subprocess.call(args, shell=True)

    print('Inference completed perform nearest neighbor search and registration')

    return
    # Load the descriptors and estimate the transformation parameters using RANSAC
    reference_desc = np.load(f'./data/bunny/32_dim/bun_zipper_res3.ply_{voxel_grid:.6f}_{n_voxels}_{gaussian_width:.6f}_3DSmoothNet.npz')
    reference_desc = reference_desc['data']

    test_desc = np.load(f'./data/bunny/32_dim/bun_zipper_res3_noise.ply_{voxel_grid:.6f}_{n_voxels}_{gaussian_width:.6f}_3DSmoothNet.npz')
    test_desc = test_desc['data']

    # Save as open3d feature 
    ref = open3d.pipelines.registration.Feature()
    ref.data = reference_desc.T

    test = open3d.pipelines.registration.Feature()
    test.data = test_desc.T

    indices_ref = np.genfromtxt(keypoints_files[0])
    indices_test = np.genfromtxt(keypoints_files[1])

    reference_pc_keypoints = np.asarray(reference_pc.points)[indices_ref.astype(int),:]
    test_pc_keypoints = np.asarray(test_pc.points)[indices_test.astype(int),:]

    # Save ad open3d point clouds
    ref_key = geometry.PointCloud()
    ref_key.points = utility.Vector3dVector(reference_pc_keypoints)

    test_key = geometry.PointCloud()
    test_key.points = utility.Vector3dVector(test_pc_keypoints)

    # First plot the original state of the point clouds
    reference_pc.estimate_normals()
    reference_pc.paint_uniform_color(FRAG1_COLOR)
    test_pc.estimate_normals()
    test_pc.paint_uniform_color(FRAG2_COLOR)
    o3d.visualization.draw_geometries([reference_pc, test_pc, keypoints_to_spheres(ref_key, SPHERE_COLOR_1), keypoints_to_spheres(test_key, SPHERE_COLOR_2)], front=[0, 0, -1.0])

    corr = o3d.pipelines.registration.correspondences_from_features(
        ref, test, mutual_filter=False
    )
    corr = np.asarray(corr)
    # print(corr)
    # print("NN correspondences:", corr.shape) 

    result_ransac = execute_global_registration(ref_key, test_key,
                ref, test, 0.01)
    print("RANSAC fitness:", result_ransac.fitness)
    print("RANSAC inlier_rmse:", result_ransac.inlier_rmse)
    print("RANSAC #correspondences:", len(result_ransac.correspondence_set))
    print("RANSAC transformation:\n", result_ransac.transformation)
    rot_error, trans_error = compute_transformation_diff(result_ransac.transformation, T)
    print("RANSAC Rotation error (deg): ", rot_error)
    print("RANSAC Translation error (m): ", trans_error)
    # Plot point clouds after registration
    # draw_registration_result(reference_pc, test_pc,
    #             result_ransac.transformation)

    ref_matched_key, test_matched_key = find_mutually_nn_keypoints(
        ref_key, test_key, ref, test
    )
    ref_matched_key = np.squeeze(ref_matched_key)
    test_matched_key = np.squeeze(test_matched_key)

    print(ref_matched_key)
    print(test_matched_key)
    est_mat, max_clique, dt = execute_teaser_global_registration(ref_matched_key, test_matched_key)
    print("\nTEASER++ #correspondences:", len(ref_matched_key))
    print("TEASER++ transformation:\n", est_mat)
    rot_error, trans_error = compute_transformation_diff(est_mat, T)
    print("TEASER++ Rotation error (deg): ", rot_error)
    print("TEASER++ Translation error (m): ", trans_error)
    # Plot point clouds after registration
    draw_registration_result(reference_pc, test_pc,
                est_mat)
    
if __name__ == "__main__":
    main()
