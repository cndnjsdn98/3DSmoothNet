import numpy as np
import open3d as o3d
import time
import glob

from core.utils import iss_keypoints_to_indices, estimate_spacing
from core import config

FRAG1_COLOR = [1.0, 0.75, 0.0]
FRAG2_COLOR = [0, 0.629, 0.9]
SPHERE_COLOR_1 = [0,1,0.1]
SPHERE_COLOR_2 = [1, 0.3, 0.05]
NOISE_BOUND = 0.001
N_OUTLIERS = 20
OUTLIER_TRANSLATION_LB = 0.001
OUTLIER_TRANSLATION_UB = 0.05


def main(config_arguments):
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
    # test = np.transpose(np.asarray(test_pc.points))
    # # Add some noise
    # N = np.transpose(np.asarray(reference_pc.points)).shape[1]
    # test += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND
    # # Add some outliers
    # outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    # for i in range(outlier_indices.size):
    #     shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
    #     test[:, outlier_indices[i]] += shift.squeeze()
    # test_pc = o3d.geometry.PointCloud()
    # test_pc.points = o3d.utility.Vector3dVector(test.T)

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
    salient = config_arguments.salient_factor * spacing
    nonmax  = config_arguments.nonmax_factor * spacing

    tic = time.time()
    reference_key = o3d.geometry.keypoint.compute_iss_keypoints(
        reference_pc,
        salient_radius=salient,
        non_max_radius=nonmax,
        gamma_21=config_arguments.gamma_21,
        gamma_32=config_arguments.gamma_32,
        min_neighbors=config_arguments.min_neighbors
    )
    test_key = o3d.geometry.keypoint.compute_iss_keypoints(
        test_pc,
        salient_radius=salient,
        non_max_radius=nonmax,
        gamma_21=config_arguments.gamma_21,
        gamma_32=config_arguments.gamma_32,
        min_neighbors=config_arguments.min_neighbors
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


if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)
