import numpy as np
import open3d as o3d
import time
import glob
import os 

from core.utils import iss_keypoints_to_indices, estimate_spacing, keypoints_to_spheres
from core import config

FRAG1_COLOR = [1.0, 0.75, 0.0]
FRAG2_COLOR = [0, 0.629, 0.9]
SPHERE_COLOR_1 = [0,1,0.1]
SPHERE_COLOR_2 = [1, 0.3, 0.05]

def main(config_arguments):
    # Run the input parametrization
    point_cloud_files = glob.glob(os.path.join(config_arguments.input_pcl_folder, "subsampled", '*.ply'))
    pcls = {}
    keys = {}

    # Load reference point clouds
    for pc_file in point_cloud_files:
        pc = o3d.io.read_point_cloud(pc_file)
        print("---------------------")
        print("Loaded: " + pc_file)
        pts = np.asarray(pc.points)

        # Compute median point spacing
        # spacing = 0
        # for i in range(10):
        #     center = pts[np.random.randint(len(pts))]
        #     half = np.array([5, 5, 5])   # example patch half-size
        #     bbox = o3d.geometry.AxisAlignedBoundingBox(center - half, center + half)
        #     pcd_patch = pc.crop(bbox)
        #     spacing += estimate_spacing(pcd_patch)
        # spacing /= 10
        # print("median spacing:", spacing)
              
        # Compute key points
        # salient = config_arguments.salient_factor * spacing
        # nonmax  = config_arguments.nonmax_factor * spacing
        salient = config_arguments.salient_factor
        nonmax  = config_arguments.nonmax_factor 

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
        print("Number of keypoints: {}\n".format(len(keypoints.points)))

        key_idx = iss_keypoints_to_indices(pc, keypoints)
        keypoint_file = pc_file + "_keypoints"

        with open(keypoint_file, 'w') as f:
            f.write('\n'.join(str(idx) for idx in key_idx))

        if config_arguments.visualize:
            pcls[pc_file] =  pc
            keys[pc_file] = keypoints

    # Plot point clouds
    if config_arguments.visualize:
        # Load reference point cloud
        reference_pc = pcls[point_cloud_files[0]]
        ref_key = keys[point_cloud_files[0]]
        test_pc = pcls[point_cloud_files[1]]
        test_key = keys[point_cloud_files[1]]
        # reference_pc.estimate_normals()
        # reference_pc.paint_uniform_color(FRAG1_COLOR)
        # test_pc.estimate_normals()
        # test_pc.paint_uniform_color(FRAG2_COLOR)
        o3d.visualization.draw_geometries([reference_pc, test_pc, keypoints_to_spheres(ref_key, SPHERE_COLOR_1, 3), keypoints_to_spheres(test_key, SPHERE_COLOR_2, 3)])

if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)
