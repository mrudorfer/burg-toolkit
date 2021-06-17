import os
import sys

# Add simulator path
base_path = '/home/rudorfem/dev/3d_Grasping/GPNet/'
sim_path = os.path.join(base_path, 'simulator')
sim_test_path = os.path.join(base_path, 'simulator', 'simulateTest')
for add_path in [sim_path, sim_test_path]:
    if add_path not in sys.path:
        sys.path.insert(0, add_path)

import argparse
import copy
import configparser
import csv
import pathlib
import shutil

import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
import pybullet as p

import burg_toolkit as burg
from simulateTest.simulatorTestDemo import *


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
    print('\tdims are ok?', ok)
    if not ok:
        # check if scaling is possible so that it would fit
        # scale max dimension to 0.15 and then check min dimension
        if mind * 0.15 / maxd > 0.06:
            print('\tobject could be scaled')
        else:
            print('\tobject cannot be scaled properly')
    return ok


def visualize_angle_distribution(graspset):
    x_axes = graspset.rotation_matrices[:, :, 2]
    print('x-min', np.min(x_axes, axis=0))
    print('x-max', np.max(x_axes, axis=0))
    pc = burg.util.numpy_pc_to_o3d(x_axes)

    colors = np.zeros((len(graspset), 3))
    cosine = graspset.widths
    mi = np.min(cosine)
    ma = np.max(cosine)
    print(f'cosine min mxa {mi} {ma}')
    colors[:, 0] = (cosine - mi) / (ma - mi)
    colors[:, 1] = 1 - (cosine - mi) / (ma - mi)
    pc.colors = o3d.utility.Vector3dVector(colors)

    burg.visualization.show_o3d_point_clouds([pc, o3d.geometry.TriangleMesh.create_coordinate_frame()])


def get_cosine_values(graspset):
    x_axes = graspset.rotation_matrices[:, :, 0]
    z_axes = graspset.rotation_matrices[:, :, 2]

    y_tangents = - np.cross(x_axes, (0, 0, 1))
    y_tangents = y_tangents / np.linalg.norm(y_tangents, axis=-1)[:, np.newaxis]

    # compute row-wise dot products
    cosines = np.einsum('ij,ij->i', z_axes, y_tangents)
    return cosines


def z_move(graspset, contacts, z_move_length=0.015):
    """
    need to move by plus 15mm away from the center to store data, and for making simulations
    """
    offsets = graspset.rotation_matrices[:, :, 2] * z_move_length
    graspset.translations = graspset.translations + offsets
    if contacts is not None:
        for i in range(2):
            contacts[:, i] = contacts[:, i] + offsets
    return graspset, contacts


def preprocess_shapes(data_cfg, shapes):
    reader = burg.io.BaseviMatlabScenesReader(data_cfg)

    print('preprocessor...')
    print('read object library')
    object_library, index2name = reader.read_object_library()
    object_library.yell()

    for shape_name in shapes:
        shape = object_library[shape_name]
        print('\n************************')
        print('object:', shape_name)

        # print some stats
        burg.mesh_processing.check_properties(shape.mesh)

        # check that object complies with dimension requirements
        print('checking dimensions...')
        print('\tdims:', burg.mesh_processing.dimensions(shape.mesh))
        check_mesh_dimensions(shape.mesh)

        tri_mesh = burg.util.o3d_mesh_to_trimesh(shape.mesh)

        # find the resting poses for the object
        min_prob = 0.05
        max_num = 4
        print(f'searching at most {max_num} resting positions with prob at least {min_prob}...')
        transforms, probs = trimesh.poses.compute_stable_poses(tri_mesh)
        transforms = transforms[probs >= min_prob][:max_num]
        print(f'\tfound {len(probs)}, of which {np.count_nonzero(probs >= min_prob)} are sufficiently' +
              f' probable, of which we use {len(transforms)}')

        # save original mesh and poses as well, although not needed, just so we know what's going on
        shape.make_urdf_file(os.path.join(shape_dir, 'originals'), overwrite_existing=True)
        np.save(os.path.join(shape_dir_originals, f'{shape_name}_poses'), transforms)

        for i in range(len(transforms)):
            print(f'pose {i} with probability:', probs[i])

            # change object to store it in correct pose
            orig_mesh = copy.deepcopy(shape.mesh)
            shape.mesh.transform(transforms[i])
            shape.identifier = shape_name + f'_pose_{i}'
            shape.make_urdf_file(shape_dir_transformed, overwrite_existing=True)

            # find vhacd and store it as well
            p.connect(p.DIRECT)

            transformed_fn = os.path.join(shape_dir_transformed, shape.identifier + '.obj')
            vhacd_fn = os.path.join(shape_dir_vhacd, shape.identifier + '.obj')
            log_fn = os.path.join(shape_dir_vhacd, shape.identifier + '_log.txt')
            p.vhacd(transformed_fn, vhacd_fn, log_fn)
            p.disconnect()

            # we now want to associate the urdf files with the vhacd obj files
            # since vhacd and transformed have the same names which are referenced in urdf, we can just copy urdf files
            # (creating them anew would give new properties as well, which we don't want)
            original_urdf = os.path.join(shape_dir_transformed, shape.identifier + '.urdf')
            shutil.copy2(original_urdf, shape_dir_vhacd)

            # finally add to list of shapes
            with open(shapes_fn, 'a') as f:
                f.write(shape.identifier + '\n')

            # revert changes
            shape.identifier = shape_name
            shape.mesh = orig_mesh


def create_grasp_samples(shapes):
    if shapes is None:
        shapes = []
        with open(shapes_fn, 'r') as f:
            for line in f.readlines():
                shapes.append(line.strip())

    for shape_name in shapes:
        print('\n************************')
        print('object:', shape_name)

        mesh = burg.io.load_mesh(os.path.join(shape_dir_transformed, shape_name))

        # make settings for grasp sampling
        ags = burg.sampling.AntipodalGraspSampler()
        ags.gripper = burg.gripper.ParallelJawGripper(opening_width=0.085,
                                                      finger_length=0.03,
                                                      finger_thickness=0.003)
        ags.n_orientations = 18
        ags.max_targets_per_ref_point = 1
        ags.no_contact_below_z = 0.015
        ags.mesh = mesh

        print('sampling...')
        graspset, contacts = ags.sample(102600)
        print(f'sampled {len(graspset)} grasps in total')
        # visualize_angle_distribution(graspset)

        # todo: remove grasps which are too similar?? near duplicates?
        # we might indeed have near duplicates, as we start using the same ref points again at some point
        # - this could be caught if it should happen, doesn't happen too easily because for every contact, we
        #   will have n_orientations of grasps if a suitable 2nd contact has been found
        # - could introduce re-sampling in case we run out of points

        # shift contacts and centers 15 in z-direction
        print('introducing the z_move shift for saving and simulation')
        graspset, contacts = z_move(graspset, contacts, z_move_length=0.015)

        cosines = get_cosine_values(graspset)
        np.save(os.path.join(annotation_dir, shape_name + '_cos.npy'), cosines)
        np.save(os.path.join(annotation_dir, shape_name + '_contact.npy'), contacts)
        np.save(os.path.join(annotation_dir, shape_name + '_c.npy'), graspset.translations)
        np.save(os.path.join(annotation_dir, shape_name + '_q.npy'), graspset.quaternions)
        np.save(os.path.join(annotation_dir, shape_name + '_d.npy'), graspset.widths)
        print('files saved.')


def write_shape_poses_file(grasp_data, file_path, with_score=False):
    """
    Parameters
    ----------
    grasp_data: dict with shapes as keys and grasps as ndarrays
    file_path: full path to the file to write it in - will be overwritten
    with_score: bool, determines whether or not to add the scores to the file

    Returns
    -------
    creates or overwrites a file with shapes and poses (as used in simulation for example)
    """
    with open(file_path, 'w') as f:
        for shape, grasps in grasp_data.items():
            if len(grasps) > 0:
                f.write(shape + '\n')
                for g in grasps:
                    if with_score:
                        f.write('%f,%f,%f,%f,%f,%f,%f,%f\n' % (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]))
                    else:
                        f.write('%f,%f,%f,%f,%f,%f,%f\n' % (g[0], g[1], g[2], g[3], g[4], g[5], g[6]))


def read_sim_csv_file(filename, keep_num=None):
    """
    This reads the csv log file created during simulation.

    :return: returns a dict with shape id as keys and np array as value.
             the np array is of shape (n, 10): 0:3 pos, 3:7 quat, annotation id, sim result, sim success, empty
             keeps only keep_num entries (as of annotation idx order, which is ordered by descending score)
    """
    print(f'reading csv data from {filename}')
    sim_data = {}
    counters = {}
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(reader):
            shape = row[0]
            if shape not in sim_data.keys():
                # we do not know the array length in advance, so start with 10k
                data_array = np.zeros((10000, 11))
                sim_data[shape] = data_array
                counters[shape] = 0
            elif counters[shape] == len(sim_data[shape]):
                sim_data[shape] = np.resize(sim_data[shape], (len(sim_data[shape]) + 10000, 11))

            sim_data[shape][counters[shape]] = [
                float(row[4]),  # pos: x, y, z
                float(row[5]),
                float(row[6]),
                float(row[10]),  # quat: w, x, y, z, converted from pybullet convention
                float(row[7]),
                float(row[8]),
                float(row[9]),
                int(row[1]),  # annotation id
                int(row[2]),  # simulation result
                int(row[2]) == 0,  # simulation success flag
                -1.   # left empty for rule-based success flag
            ]
            counters[shape] += 1

    # now reduce arrays to their actual content
    for key in sim_data.keys():
        sim_data[key] = np.resize(sim_data[key], (counters[key], 11))
        # also sort by annotation id
        order = np.argsort(sim_data[key][:, 7])
        sim_data[key] = sim_data[key][order]
        sim_data[key] = sim_data[key][:keep_num]

    return sim_data


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def simulate_grasp_samples(shapes=None):
    if shapes is None:
        shapes = []
        with open(shapes_fn, 'r') as f:
            for line in f.readlines():
                shapes.append(line.strip())

    # annotation_dir = '/home/rudorfem/datasets/GPNet_release_data/annotations/candidate/'

    results = []
    for shape in shapes:
        # read the grasp data and put it in some other file which can be read by the simulator
        centers = np.load(os.path.join(annotation_dir, shape + '_c.npy'))
        quats = np.load(os.path.join(annotation_dir, shape + '_q.npy'))
        grasps = np.concatenate([centers, quats], axis=1)
        shape_tmp_fn = os.path.join(tmp_dir, shape + '_poses.txt')
        write_shape_poses_file({shape: grasps}, shape_tmp_fn)

        dict_simulation = dotdict({
            'testFile': shape_tmp_fn,  # current nms_poses_view0.txt file path
            # below: keeping everything at default value!
            'processNum': 10,
            'width': False,
            'gripperFile': os.path.join(sim_path, 'gpnet_data/gripper/parallel_simple.urdf'),
            'objMeshRoot': shape_dir_vhacd,
            # 'objMeshRoot': os.path.join(sim_path, 'gpnet_data/urdf/'),
            'visual': False,
            'dir': 'None'
        })
        main_simulator(dict_simulation)

        sim_data = read_sim_csv_file(shape_tmp_fn[:-4] + '_log.csv')
        success = sim_data[shape][:, 9]
        results.append((np.count_nonzero(success), len(success)))

        bool_array = np.empty(len(success), dtype=np.bool)
        bool_array[:] = success
        np.save(os.path.join(sim_result_dir, shape + '.npy'), bool_array)

    for result in results:
        print(f'{result[0]} of {result[1]} successful')


def visualize_data():
    shapes = []
    with open(shapes_fn, 'r') as f:
        for line in f.readlines():
            shapes.append(line.strip())

    for shape in shapes:
        # read the grasp data and put it in some other file which can be read by the simulator
        centers = np.load(os.path.join(annotation_dir, shape + '_c.npy'))
        quats = np.load(os.path.join(annotation_dir, shape + '_q.npy'))
        contacts = np.load(os.path.join(annotation_dir, shape + '_contact.npy'))
        gs = burg.grasp.GraspSet.from_translations_and_quaternions(centers, quats)

        print('centers:', np.min(centers[:, 2], axis=0))
        print('contact1:', np.min(contacts[:, 0, 2], axis=0))
        print('contact2:', np.min(contacts[:, 1, 2], axis=0))

        scores = np.load(os.path.join(sim_result_dir, shape + '.npy'))
        gs.scores = scores

        gs, contacts = z_move(gs, contacts, z_move_length=-0.015)

        # check the duplicats here!
        dups = np.all(np.isclose(contacts[:, 0], contacts[1230, 0]), axis=-1)
        print('dups', dups.shape)
        print('number:', np.count_nonzero(dups))

        mesh = burg.io.load_mesh(os.path.join(shape_dir, shape + '.obj'))
        pc_centers = burg.util.numpy_pc_to_o3d(gs.translations)
        pc_contacts = burg.util.numpy_pc_to_o3d(contacts.reshape(-1, 3))

        pos_centers = burg.util.numpy_pc_to_o3d(gs[scores > 0.5].translations)
        pos_contacts = burg.util.numpy_pc_to_o3d((contacts[scores > 0.5].reshape(-1, 3)))

        burg.visualization.show_o3d_point_clouds([mesh, pc_centers, pc_contacts, pos_centers, pos_contacts])

        burg.visualization.show_grasp_set([mesh], gs[scores > 0.5], n=100, with_plane=True,
                                          gripper=burg.gripper.ParallelJawGripper(finger_thickness=0.003,
                                                                                  opening_width=0.085))


def inspect_meshes():

    ycb_mug_fn = '/home/rudorfem/datasets/YCB_grasp/shapes/mug_pose_0.obj'
    mug_vhacd_fn = os.path.join(tmp_dir, 'mug_pose_0_vhacd.obj')
    ycb_mug_fn = mug_vhacd_fn
    # ycb_mug_fn = '/home/rudorfem/datasets/YCB-scene-generation/ycbObjectsGoogleScanner/mugMediumResolution.ply'
    gpn_mug_fn = '/home/rudorfem/dev/3d_Grasping/GPNet/simulator/gpnet_data/processed/128ecbc10df5b05d96eaf1340564a4de.obj'

    ycb_mug = burg.io.load_mesh(ycb_mug_fn)
    gpn_mug = burg.io.load_mesh(gpn_mug_fn)

    print('YCB mug')
    burg.mesh_processing.check_properties(ycb_mug)

    print('\nGPN mug')
    burg.mesh_processing.check_properties(gpn_mug)
    return
    burg.visualization.show_o3d_point_clouds([ycb_mug])

    print('processing YCB mug...')
    # t_ycb_mug = burg.util.o3d_mesh_to_trimesh(ycb_mug)
    # c_ycb_mug = trimesh.decomposition.convex_decomposition(t_ycb_mug)  # needs testVHACD to be installed

    do_vhacd = True
    if do_vhacd:
        p.connect(p.DIRECT)
        name_log = os.path.join(tmp_dir, 'log.txt')
        # p.vhacd(ycb_mug_fn, name_out, name_log, alpha=0.04, resolution=50000)
        p.vhacd(ycb_mug_fn, mug_vhacd_fn, name_log)

        vhacd_mesh = burg.io.load_mesh(mug_vhacd_fn)
        burg.visualization.show_o3d_point_clouds([vhacd_mesh])



if __name__ == "__main__":
    print('generate dataset')
    arguments = parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(arguments.config)

    shapes_fn = os.path.join(arguments.output_dir, 'shapes.csv')
    shape_dir = os.path.join(arguments.output_dir, 'shapes/')
    annotation_dir = os.path.join(arguments.output_dir, 'annotations/candidate/')
    sim_result_dir = os.path.join(arguments.output_dir, 'annotations/simulateResult/')
    tmp_dir = os.path.join(arguments.output_dir, 'annotations/tmp/')

    # make sure all paths exist
    for path in [shape_dir, annotation_dir, sim_result_dir, tmp_dir]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    shape_dir_originals = os.path.join(shape_dir, 'originals')
    shape_dir_transformed = os.path.join(shape_dir, 'transformed')
    shape_dir_vhacd = os.path.join(shape_dir, 'vhacd')

    for path in [shape_dir_originals, shape_dir_transformed, shape_dir_vhacd]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # inspect_meshes()
    # preprocess_shapes(cfg['General'], [arguments.shape])
    # create_grasp_samples()
    # simulate_grasp_samples(['128ecbc10df5b05d96eaf1340564a4de'])
    simulate_grasp_samples(['mug_pose_0'])
    # visualize_data()
