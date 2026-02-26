import numpy as np
import math
import copy

import open3d as o3d

try:
    reg = o3d.pipelines.registration
except AttributeError:
    reg = o3d.registration

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

def draw_registration_result(source, target, transformation, FRAG1_COLOR=None, FRAG2_COLOR=None):
    if FRAG1_COLOR is None or FRAG2_COLOR is None:
        FRAG1_COLOR = [1.0, 0.75, 0.0]
        FRAG2_COLOR = [0, 0.629, 0.9]
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.estimate_normals()
    target_temp.estimate_normals()
    source_temp.paint_uniform_color(FRAG1_COLOR)
    target_temp.paint_uniform_color(FRAG2_COLOR)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


################ RANSAC ##################
def execute_global_registration(
        source_down, target_down, reference_desc, target_desc, distance_threshold):

    result = reg.registration_ransac_based_on_feature_matching(
            source_down, target_down, reference_desc, target_desc,
            distance_threshold,
            reg.TransformationEstimationPointToPoint(False), 4,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            reg.RANSACConvergenceCriteria(4000000, 500))
    return result


################ TEASER++ ##############

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
        rotError = math.fabs(math.acos(A))
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
