import os
import argparse
import copy
import configparser

import numpy as np
import open3d as o3d
import trimesh

import burg_toolkit as burg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/linux-config.cfg', help='path to config file')
    parser.add_argument('-s', '--shape', type=str, default='mug', help='name of shape to process, None processes all')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/rudorfem/datasets/YCB_grasp/', help='where to put generated files')
    return parser.parse_args()


def check_mesh_dimensions(mesh):
    # according to GPNet:
    # longest side < 150mm
    # shortest side > 60mm
    def _dims_ok(max_dim, min_dim):
        if maxd > 0.15:
            return False
        if mind < 0.06:
            return False
        return True

    dims = burg.mesh_processing.dimensions(mesh)
    maxd = np.max(dims)
    mind = np.min(dims)
    ok = _dims_ok(maxd, mind)
    print('dims are ok?', ok)
    if not ok:
        # check if scaling is possible so that it would fit
        # scale max dimension to 0.15 and then check min dimension
        if mind * 0.15 / maxd > 0.06:
            print('object could be scaled')
        else:
            print('object cannot be scaled properly')
    return ok


if __name__ == "__main__":
    print('generate dataset')
    arguments = parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(arguments.config)
    reader = burg.io.BaseviMatlabScenesReader(cfg['General'])

    print('read object library')
    object_library, index2name = reader.read_object_library()
    object_library.yell()

    # filter objects
    if arguments.shape is not None:
        keys = [arguments.shape]
    else:
        keys = object_library.keys()

    for shape_name in keys:
        shape = object_library[shape_name]
        print('****\n', shape_name)
        # check that object complies with dimension requirements
        print('dims:', burg.mesh_processing.dimensions(shape.mesh))
        check_mesh_dimensions(shape.mesh)

        # store files to folder
        shape_dir = os.path.join(arguments.output_dir, 'shapes/')
        shape.make_urdf_file(shape_dir, overwrite_existing=True)

        # sample some grasps
        gripper_model = burg.gripper.ParallelJawGripper(opening_width=0.085,
                                                        finger_length=0.03,
                                                        finger_thickness=0.003)
        ags = burg.sampling.AntipodalGraspSampler()
        ags.mesh = shape.mesh
        ags.gripper = gripper_model
        graspset, contacts = ags.sample(10000)
        ags.verbose = True
        # score 0: collision, score 1: ok so far
        graspset.scores = np.logical_not(ags.check_collisions(graspset, use_width=True))

        # find a resting pose for the object
        tri_mesh = burg.util.o3d_mesh_to_trimesh(shape.mesh)
        transforms, probs = trimesh.poses.compute_stable_poses(tri_mesh)
        print('transforms', transforms.shape)
        print('probs', probs)
        transforms = transforms[probs >= 0.05]
        print('>0.05:', len(transforms))
        np.save(os.path.join(shape_dir, f'{shape_name}_poses'), transforms)

        for i in range(len(transforms)):
            instance = burg.scene.ObjectInstance(shape, transforms[i])
            s = burg.scene.Scene(objects=[instance])
            print('probability:', probs[i])
            burg.visualization.show_scene(s, add_plane=True)

            # change object to store it in correct pose
            orig_mesh = copy.deepcopy(shape.mesh)
            shape.mesh.transform(transforms[i])
            shape.identifier = shape_name + f'_pose_{i}'

            shape.make_urdf_file(shape_dir, overwrite_existing=True)

            # revert changes
            shape.identifier = shape_name
            shape.mesh = orig_mesh

    print('bye')
