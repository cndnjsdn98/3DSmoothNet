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
NOISE_BOUND = 0.01
N_OUTLIERS = 20
OUTLIER_TRANSLATION_LB = 0.001
OUTLIER_TRANSLATION_UB = 0.05


def main(config_arguments):
    # Run the input parametrization
    point_cloud_files = glob.glob(config_arguments.input_pcl_folder + '*.ply')

    # Load reference point clouds
    for pc_file in point_cloud_files:
        pc = o3d.io.read_point_cloud(pc_file)
        print("\n---------------------")
        print("Loaded: " + pc_file)
        # Compute key points
        spacing = estimate_spacing(pc)
        print("median spacing:", spacing)
        salient = config_arguments.salient_factor * spacing
        nonmax  = config_arguments.nonmax_factor * spacing

        tic = time.time()
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
            pc,
            salient_radius=salient,
            non_max_radius=nonmax,
            gamma_21=config_arguments.gamma_21,
            gamma_32=config_arguments.gamma_32,
            min_neighbors=config_arguments.min_neighbors
        )
        toc = 1000 * (time.time() - tic)
        print("ISS Computation took {:.0f} [ms]".format(toc))
        print("Number of keypoints: ")
        print(keypoints)

        key_idx = iss_keypoints_to_indices(pc, keypoints)
        keypoint_file = pc_file + "_keypoints"

        with open(keypoint_file, 'w') as f:
            f.write('\n'.join(str(idx) for idx in key_idx))


if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)
