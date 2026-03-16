import numpy as np
import open3d as o3d
import time
import glob
import pickle
import copy

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
    point_cloud_files = ["./data/battery_pack/battery_pack.ply", "./data/battery_pack/battery_pack_test.ply"]

    # Create test point cloud by transforming and inducing noise
    # ref_pc = o3d.io.read_point_cloud(point_cloud_files[0])
    # test_pc = copy.deepcopy(ref_pc)

    ref_pc = o3d.io.read_triangle_mesh(point_cloud_files[0])
    test_pc = copy.deepcopy(ref_pc)
    test_pc.compute_vertex_normals()  # optional but helpful for rendering

    # Apply arbitrary scale, translation and rotation
    T = np.array(
        [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, 1.15576939e3],
        [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e2],
        [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e4],
        [0, 0, 0, 1]])

    test_pc.transform(T)
    # Save test point cloud
    # if not test_pc.has_normals():
    #     test_pc.estimate_normals()
    # ok = o3d.io.write_point_cloud(
    #     point_cloud_files[1],
    #     test_pc,
    #     write_ascii=False,        # False = binary (smaller/faster), True = ASCII
    #     compressed=False,         # True can reduce size for some formats
    #     print_progress=True
    # )

    ok = o3d.io.write_triangle_mesh(
        point_cloud_files[1],
        test_pc,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True
    )

    if not ok:
        raise RuntimeError("Failed to write point cloud to output.ply")
    
    gt = {'T': T}
    with open('./data/battery_pack/gt.pkl', 'wb') as f:
        pickle.dump(gt, f)


if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)
