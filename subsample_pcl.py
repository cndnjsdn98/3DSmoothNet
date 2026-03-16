
import open3d as o3d
import glob
import copy
import sys
import os

from core.utils import iss_keypoints_to_indices, estimate_spacing, keypoints_to_spheres
from core import config

FRAG_ORIGINAL = [1.0, 0.75, 0.0]    # Yellow
FRAG_SMOOTH = [0, 0.629, 0.9]       #Blue

def main(config_arguments):
    # Run the input parametrization
    point_cloud_files = glob.glob(config_arguments.input_pcl_folder + '*.ply')
    # pc_file = config_arguments.input_pcl_folder + pcl_name

    for pc_file in point_cloud_files:

        # Load point cloud
        pc = o3d.io.read_point_cloud(pc_file)
        print("---------------------")
        print("Loaded: " + pc_file)
        pc_original = copy.deepcopy(pc).translate((0, 1000, 0))

        pc = pc.voxel_down_sample(voxel_size=config_arguments.voxel_size)  
        pc = pc.remove_non_finite_points()
        pc, ind = pc.remove_statistical_outlier(
            nb_neighbors=config_arguments.nb_neighbors,
            std_ratio=config_arguments.std_ratio
        )
        if config_arguments.visualize:
            pc.estimate_normals()
            pc.paint_uniform_color(FRAG_SMOOTH)
            pc_original.estimate_normals()
            pc_original.paint_uniform_color(FRAG_ORIGINAL)
            print("\\a")
            sys.stdout.flush()
            o3d.visualization.draw_geometries([pc, pc_original])

        pc_file_name = os.path.basename(pc_file)
        voxel_pc_dir = os.path.join(config_arguments.input_pcl_folder, "subsampled", pc_file_name)
        success = o3d.io.write_point_cloud(
            voxel_pc_dir, 
            pc, 
            write_ascii=False,   # Matches "Binary encoding"
            compressed=False,    # Keep standard binary if desired
            print_progress=True  # Optional: show a progress bar
        )
if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)
