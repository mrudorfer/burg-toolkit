import copy

import configparser
import open3d as o3d
import trimesh
import numpy as np

import burg_toolkit as burg


def load_objects_and_generate_urdf():
    cfg_fn = '../config/config.cfg'
    print('using config file in:', cfg_fn)

    cfg = configparser.ConfigParser()
    cfg.read(cfg_fn)
    reader = burg.io.BaseviMatlabScenesReader(cfg['General'])

    # load object library
    print('read object library')
    object_library, index2name = reader.read_object_library()
    object_library.yell()

    target_object = object_library['flatheadScrewdriver']

    mesh = target_object.mesh
    burg.mesh_processing.check_properties(mesh)
    inertia, com = burg.mesh_processing.compute_mesh_inertia(mesh, target_object.mass)
    print('inertia:\n', inertia)

    object_library.generate_urdf_files('../data/tmp', overwrite_existing=True)


def inspect_mesh_watertightness():
    mesh_fn = 'E:/datasets/YCB/006_mustard_bottle/google_16k/nontextured.stl'

    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    print(f'loaded mesh {mesh_fn}')
    burg.mesh_processing.check_properties(mesh)
    tr_mesh = burg.util.o3d_mesh_to_trimesh(mesh)

    # find the resting poses for the object
    min_prob = 0.05
    max_num = 4
    print(f'searching at most {max_num} resting positions with prob at least {min_prob}...')
    transforms, probs = trimesh.poses.compute_stable_poses(tr_mesh)
    transforms = transforms[probs >= min_prob][:max_num]
    print(f'\tfound {len(probs)}, of which {np.count_nonzero(probs >= min_prob)} are sufficiently' +
          f' probable, of which we use {len(transforms)}')

    for tf in transforms:
        vis_mesh = copy.deepcopy(mesh)
        vis_mesh.transform(tf)
        burg.visualization.show_o3d_point_clouds([vis_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)])


if __name__ == "__main__":
    #load_objects_and_generate_urdf()
    inspect_mesh_watertightness()
