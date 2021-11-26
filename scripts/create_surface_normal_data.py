import argparse
import os
from time import time

import scipy.spatial
from tqdm import tqdm
import numpy as np
import open3d as o3d

import burg_toolkit as burg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=str,
                        default='/home/rudorfem/datasets/object_libraries/test_library/test_library.yaml',
                        help='path to object library file')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='where to put generated files, default is in subdirs of object library')
    parser.add_argument('-pc', '--create_pc', action='store_true', default=False,
                        help='use this option to render views and create point clouds')
    parser.add_argument('-pcn', '--create_pcn', action='store_true', default=False,
                        help='use this to compute the normals, assumes folder with point clouds exists')
    return parser.parse_args()


def show_pcn(pcn):
    o3d_pc = burg.util.numpy_pc_to_o3d(pcn)
    obj_list = [o3d_pc, burg.visualization.create_plane()]
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=1 / 10000,
        cone_radius=1.5 / 10000,
        cylinder_height=5.0 / 1000,
        cone_height=4.0 / 1000,
        resolution=20,
        cylinder_split=4,
        cone_split=1)
    arrow.compute_vertex_normals()
    for i in range(len(pcn)):
        my_arrow = o3d.geometry.TriangleMesh(arrow)
        my_arrow.rotate(burg.util.rotation_to_align_vectors([0, 0, 1], pcn[i, 3:6]), center=[0, 0, 0])
        my_arrow.translate(pcn[i, 0:3])
        obj_list.append(my_arrow)

    burg.visualization.show_geometries(obj_list)


def render_point_clouds(object_library_fn, output_dir=None):
    lib_dir = os.path.dirname(object_library_fn)
    lib = burg.ObjectLibrary.from_yaml(object_library_fn)
    if output_dir is None:
        output_dir = lib_dir

    pc_temp_data = os.path.join(output_dir, 'pc_data')
    r = burg.render.MeshRenderer(output_dir=pc_temp_data)
    cpg = burg.render.CameraPoseGenerator(lower_hemisphere=True)
    poses = cpg.icosphere(subdivisions=2, in_plane_rotations=1, scales=1).reshape(-1, 4, 4)
    print('number of poses:', len(poses))

    start_time = time()
    for key, obj in lib.items():
        print('object:', key)
        r.render_depth(obj.mesh, poses, sub_dir=key, depth_fn_type='npy-pc', depth_fn_func=lambda j: f'pc{j}')
    print('required time:', time()-start_time)


def create_normals_data(object_library_fn, output_dir=None, n_points=1024):
    lib_dir = os.path.dirname(object_library_fn)
    lib = burg.ObjectLibrary.from_yaml(object_library_fn)
    if output_dir is None:
        output_dir = lib_dir

    pc_temp_data = os.path.join(output_dir, 'pc_data')
    pc_and_normal_data = os.path.join(output_dir, f'pcn_data_{n_points}')

    start_time = time()
    for key, obj in lib.items():
        # read point cloud data
        print('object:', key)
        shape_dir = os.path.join(pc_and_normal_data, key)
        burg.io.make_sure_directory_exists(shape_dir)
        pc_files = [f for f in os.listdir(os.path.join(pc_temp_data, key)) if f.endswith('.npy')]
        view_idx = 0
        for pc_fn in tqdm(pc_files):
            pc = np.load(os.path.join(pc_temp_data, key, pc_fn))
            if len(pc) < n_points:
                print(f'skipping {pc_fn}, has only {len(pc)} points')
                continue

            # downsample point cloud
            indices = burg.sampling.farthest_point_sampling(pc, n_points)
            subsampled_pc = pc[indices]
            normals = burg.mesh_processing.compute_interpolated_vertex_normals(obj.mesh, subsampled_pc)

            pcn = np.concatenate([subsampled_pc, normals], axis=1)
            np.save(os.path.join(shape_dir, f'pcn{view_idx}.npy'), pcn)
            view_idx += 1

    print('required time:', time()-start_time)


if __name__ == '__main__':
    args = parse_args()
    if args.create_pc:
        render_point_clouds(args.lib, args.output_dir)
    if args.create_pcn:
        create_normals_data(args.lib, args.output_dir)
