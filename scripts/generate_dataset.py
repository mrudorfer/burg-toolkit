import os
import sys
import argparse
import copy
import configparser
import csv
import shutil
import time

import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
import pybullet as p
import pybullet_data

import burg_toolkit as burg

# try importing simulation
try:
    import gpnet_sim
except ImportError:
    gpnet_sim = None
    print('Warning: package gpnet_sim not found. Please install from https://github.com/mrudorfer/GPNet-simulator')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/linux-config.cfg', help='path to config file, ' +
                        'only used for Hectors objects, ignored otherwise')
    parser.add_argument('-y', '--ycb_path', type=str, default=None,
                        help='path to YCB objects in downloaded format, overrides -c option')
    parser.add_argument('-s', '--shape', type=str, default=None, help='name of shape to process, None processes all')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/rudorfem/datasets/YCB_grasp_tmp/',
                        help='where to put generated dataset files')
    parser.add_argument('-l', '--log_file', type=str, default=None, help='name of log file within output dir')
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


class YCBObjectReader:
    """
    Class to read the YCB objects from a directory into an object library.
    Assumes directory structure:
    - base_path
        - shape_name_1
            - model_type
                - model_fn
        - shape_name_2
        - ...
    """
    def __init__(self, base_path, model_type='google_16k', model_fn='nontextured.ply'):
        self.base_path = base_path
        self.model_type = model_type
        self.model_fn = model_fn

    def read_object_library(self):
        shape_names = [x for x in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, x))]
        object_library = burg.scene.ObjectLibrary()

        for shape_name in shape_names:
            # this assumes the directory structure
            model_path = os.path.join(self.base_path, shape_name, self.model_type, self.model_fn)
            mesh = burg.io.load_mesh(model_path)
            obj_type = burg.scene.ObjectType(identifier=shape_name, mesh=mesh)

            object_library[shape_name] = obj_type

        # this is a bloody mass hack
        object_library['003_cracker_box'].mass = 0.411
        object_library['005_tomato_soup_can'].mass = 0.349
        object_library['006_mustard_bottle'].mass = 0.603
        object_library['010_potted_meat_can'].mass = 0.370
        object_library['025_mug'].mass = 0.118
        object_library['044_flat_screwdriver'].mass = 0.0984
        object_library['051_large_clamp'].mass = 0.125
        object_library['056_tennis_ball'].mass = 0.058

        return object_library


def preprocess_shapes(data_cfg, ycb_path, shapes):
    if ycb_path is None:
        if data_cfg is None:
            raise ValueError('either data_cfg or ycb_path must be given')
        reader = burg.io.BaseviMatlabScenesReader(data_cfg)
        print('preprocessor...')
        print('read object library')
        object_library, index2name = reader.read_object_library()
    else:
        reader = YCBObjectReader(base_path=ycb_path)
        object_library = reader.read_object_library()

    object_library.yell()

    if shapes is None:
        shapes = object_library.keys()
    print('will preprocess the following shapes:')
    print(shapes)

    default_inertia = np.eye(3) * 0.001
    mass_factor = 0.001

    for shape_name in shapes:
        shape = object_library[shape_name]
        print('\n************************')
        print('object:', shape_name)

        # print some stats (takes long)
        # burg.mesh_processing.check_properties(shape.mesh)

        # check that object complies with dimension requirements
        print('checking dimensions...')
        print('\tdims:', burg.mesh_processing.dimensions(shape.mesh))
        check_mesh_dimensions(shape.mesh)

        burg.io.save_mesh_and_urdf(shape, shape_dir_originals, default_inertia=default_inertia, mass_factor=mass_factor)

        tri_mesh = burg.util.o3d_mesh_to_trimesh(shape.mesh)

        # find the resting poses for the object
        min_prob = 0.02
        max_num = 10
        min_num = 1  # even if min_prob is not achieved
        print(f'searching {min_num} to {max_num} resting positions with prob at least {min_prob}...')
        # this function requires watertight meshes, not all of them are watertight but it works reasonably well
        # note that applies to meshes that have self-intersecting triangles - holes might be a different story
        transforms, probs = trimesh.poses.compute_stable_poses(tri_mesh)
        used_tf_indices = probs >= min_prob
        if np.count_nonzero(used_tf_indices) < min_num:
            transforms = transforms[:min_num]
        else:
            transforms = transforms[used_tf_indices][:max_num]

        print(f'\tfound {len(probs)}, of which {np.count_nonzero(probs >= min_prob)} are sufficiently' +
              f' probable, and we use {len(transforms)}')
        print('all probs:', probs)

        for i in range(len(transforms)):
            print('**********************')
            print(f'{shape} pose {i} with probability:', probs[i])

            # change object to store it in correct pose
            orig_mesh = copy.deepcopy(shape.mesh)
            shape.mesh.transform(transforms[i])

            name = shape_name + f'_pose_{i}'
            burg.io.save_mesh(os.path.join(shape_dir_transformed, name + '.obj'), shape.mesh)

            # create vhacd and store it as well
            p.connect(p.DIRECT)
            transformed_fn = os.path.join(shape_dir_transformed, name + '.obj')
            vhacd_fn = os.path.join(shape_dir_vhacd, name + '.obj')
            log_fn = os.path.join(shape_dir_vhacd, name + '_log.txt')
            p.vhacd(transformed_fn, vhacd_fn, log_fn)
            p.disconnect()

            # get vhacd mesh
            vhacd_mesh = burg.io.load_mesh(vhacd_fn)

            # create urdf file
            # we use COM of vhacd mesh
            urdf_fn = os.path.join(shape_dir_transformed, name + '.urdf')
            burg.io.save_urdf(urdf_fn, name + '.obj', name, inertia=default_inertia, com=vhacd_mesh.get_center(),
                              mass=shape.mass * mass_factor)
            shutil.copy2(urdf_fn, shape_dir_vhacd)  # associate with vhacd obj files as well

            # ** DOUBLE_CHECK SIM
            # we could be done here, however, at the start of the sim the object may move slightly, mainly because
            # we comute resting poses for complete meshes and use vhacd meshes for simulation
            # therefore we use found pose to initialise a simulation with vhacd and retrieve final resting pose
            p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            oid = p.loadURDF(os.path.join(shape_dir_vhacd, name + '.urdf'))
            pos1, quat1 = p.getBasePositionAndOrientation(oid)
            print('*********************')
            print(f'shape center {shape.mesh.get_center()}')
            print(f'vhacd center {vhacd_mesh.get_center()}')
            print(f'pos {pos1}, quat {quat1}, before simulation')
            dt = 1/240
            seconds = 5
            for _ in range(int(seconds/dt)):
                p.stepSimulation()
                # time.sleep(2*dt)
            pos2, quat2 = p.getBasePositionAndOrientation(oid)
            p.disconnect()

            # ** APPLY POSE ADJUSTMENTS FOUND IN SIM

            print(f'pos {pos2}, quat {quat2}, after simulation')
            print('*********************')
            diff_pos = np.asarray(pos2) - np.asarray(pos1)

            print('min bounds before adjustments')
            print('shape', shape.mesh.get_min_bound())
            print('vhacd', vhacd_mesh.get_min_bound())

            # apply orientation
            rot_mat = burg.util.tf_from_pos_quat(quat=quat2, convention='pybullet')[:3, :3]
            rot_center = vhacd_mesh.get_center()
            shape.mesh.rotate(rot_mat, center=rot_center)
            vhacd_mesh.rotate(rot_mat, center=rot_center)

            # apply translation
            shape.mesh.translate(diff_pos, relative=True)
            vhacd_mesh.translate(diff_pos, relative=True)

            print('min bounds after adjustments')
            print('shape', shape.mesh.get_min_bound())
            print('vhacd', vhacd_mesh.get_min_bound())

            # finally put center of mass onto the z-axis
            target_pos = np.zeros(3)
            target_pos[2] = shape.mesh.get_center()[2]
            shape.mesh.translate(target_pos, relative=False)
            target_pos[2] = vhacd_mesh.get_center()[2]
            vhacd_mesh.translate(target_pos, relative=False)

            # ** STORE FINAL FILES
            # since the mesh got transformed, we save it again (and recreate urdf files)
            burg.io.save_mesh(os.path.join(shape_dir_transformed, name + '.obj'), shape.mesh)
            burg.io.save_mesh(vhacd_fn, vhacd_mesh)
            burg.io.save_urdf(urdf_fn, name + '.obj', name, inertia=default_inertia, com=vhacd_mesh.get_center(),
                              mass=shape.mass * mass_factor)
            shutil.copy2(urdf_fn, shape_dir_vhacd)  # associate with vhacd obj files as well

            # finally add to list of shapes
            with open(shapes_fn, 'a') as f:
                f.write(name + '\n')

            # revert changes to get clean state for next pose
            shape.mesh = orig_mesh


def get_relevant_shapes(shapes=None):
    """
    will retrieve all shapes from the shape file, except when certain shapes were specified anyways
    """
    all_shapes = []
    with open(shapes_fn, 'r') as f:
        for line in f.readlines():
            all_shapes.append(line.strip())
    if shapes is None:
        return all_shapes

    for shape in shapes:
        if shape not in all_shapes:
            raise ValueError(f'shape "{shape}" is not available. list of all available shapes: {all_shapes}')
    return shapes


def create_grasp_samples(shapes=None):
    shapes = get_relevant_shapes(shapes)

    for shape_name in shapes:
        print('\n************************')
        print('object:', shape_name)

        mesh = burg.io.load_mesh(os.path.join(shape_dir_transformed, shape_name + '.obj'))

        # make settings for grasp sampling
        ags = burg.sampling.AntipodalGraspSampler()
        ags.gripper = burg.gripper.ParallelJawGripper(opening_width=0.085,
                                                      finger_length=0.03,
                                                      finger_thickness=0.003)
        ags.n_orientations = 18
        ags.max_targets_per_ref_point = 1
        ags.no_contact_below_z = 0.015
        ags.only_grasp_from_above = True
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


def simulate_grasp_samples(shapes=None):
    shapes = get_relevant_shapes(shapes)

    # annotation_dir = '/home/rudorfem/datasets/GPNet_release_data/annotations/candidate/'

    if gpnet_sim is None:
        raise ImportError('cannot simulate grasp samples as package gpnet_sim is not loaded.')

    results = []
    for shape in shapes:
        # read the grasp data and put it in some other file which can be read by the simulator
        centers = np.load(os.path.join(annotation_dir, shape + '_c.npy'))
        quats = np.load(os.path.join(annotation_dir, shape + '_q.npy'))
        grasps = np.concatenate([centers, quats], axis=1)
        shape_tmp_fn = os.path.join(tmp_dir, shape + '_poses.txt')
        write_shape_poses_file({shape: grasps}, shape_tmp_fn)

        conf = gpnet_sim.default_conf()
        conf.testFile = shape_tmp_fn
        conf.objMeshRoot = shape_dir_vhacd
        # todo: this path should be accessible in a better way, but this is gpnet_sim package's problem
        conf.gripperFile = '/home/rudorfem/dev/GPNet-simulator/gpnet_data/gripper/parallel_simple.urdf'
        gpnet_sim.simulate(conf)

        sim_data = read_sim_csv_file(shape_tmp_fn[:-4] + '_log.csv')
        success = sim_data[shape][:, 9]
        results.append((shape, np.count_nonzero(success), len(success)))

        bool_array = np.empty(len(success), dtype=bool)
        bool_array[:] = success
        np.save(os.path.join(sim_result_dir, shape + '.npy'), bool_array)

    for result in results:
        print(f'{result[0]}: {result[1]} of {result[2]} successful')


def show_meshes(shapes=None):
    shapes = get_relevant_shapes(shapes)

    meshes = [burg.visualization.create_plane()]
    for shape in shapes:
        meshes.append(burg.io.load_mesh(os.path.join(shape_dir_transformed, shape + '.obj')))

    burg.visualization.show_o3d_point_clouds(meshes)


def visualize_data(shapes=None):
    shapes = get_relevant_shapes(shapes)

    for shape in shapes:
        # read the grasp data and put it in some other file which can be read by the simulator
        print('***********')
        print(f'shape: {shape}')
        centers = np.load(os.path.join(annotation_dir, shape + '_c.npy'))
        quats = np.load(os.path.join(annotation_dir, shape + '_q.npy'))
        contacts = np.load(os.path.join(annotation_dir, shape + '_contact.npy'))
        gs = burg.grasp.GraspSet.from_translations_and_quaternions(centers, quats)

        scores = np.load(os.path.join(sim_result_dir, shape + '.npy'))
        gs.scores = scores
        n_success = np.count_nonzero(scores > 0.5)

        gs, contacts = z_move(gs, contacts, z_move_length=-0.015)

        # check the duplicats here!
        dups = np.all(np.isclose(contacts[:, 0], contacts[1230, 0]), axis=-1)
        print('dups', dups.shape)
        print('number:', np.count_nonzero(dups))

        mesh = burg.io.load_mesh(os.path.join(shape_dir_transformed, shape + '.obj'))
        pc_centers = burg.util.numpy_pc_to_o3d(gs[scores < 0.5].translations)
        pc_contacts = burg.util.numpy_pc_to_o3d(contacts[scores < 0.5].reshape(-1, 3))

        pc_centers.paint_uniform_color([0.8, 0.8, 0.8])
        pc_contacts.paint_uniform_color([0.8, 0.3, 0.2])

        pos_centers = burg.util.numpy_pc_to_o3d(gs[scores > 0.5].translations)
        pos_contacts = burg.util.numpy_pc_to_o3d((contacts[scores > 0.5].reshape(-1, 3)))

        pos_centers.paint_uniform_color([0.5, 0.5, 0.5])
        pos_contacts.paint_uniform_color([0.3, 0.8, 0.2])

        if n_success == 0:
            print('no successful grasps found')
            vis_objs = [mesh, pc_centers, pc_contacts,
                        burg.visualization.create_plane()]
        else:
            vis_objs = [mesh, pc_centers, pc_contacts, pos_centers, pos_contacts,
                        burg.visualization.create_plane()]
        print('showing contacts and centers: \n\tsuccessful centers darker than unsuccessful ones' +
              '\n\tsuccessful contacts green, unsuccessful red')
        burg.visualization.show_o3d_point_clouds(vis_objs)

        print('showing some example grasp streaks')
        burg.visualization.show_grasp_set([mesh], gs[3*18:5*18], with_plane=True, score_color_func=lambda s: [1-s, s, 0],
                                          gripper=burg.gripper.ParallelJawGripper(finger_thickness=0.003,
                                                                                  opening_width=0.085))
        if n_success == 0:
            print('cannot show any successful grasps... proceeding with next shape')
            continue

        print('showing up to 100 random successful grasps')
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


def see_vhacd_in_sim(shapes=None):
    shapes = get_relevant_shapes(shapes)

    for shape in shapes:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        fn = os.path.join(shape_dir_vhacd, shape + '.urdf')
        print(fn)
        oid = p.loadURDF(fn)

        print('position and quaternion at start:\n', p.getBasePositionAndOrientation(oid))
        print('press enter to start simulation')
        input()

        seconds = 10
        dt = 1/240
        for i in range(int(seconds/dt)):
            p.stepSimulation()
            time.sleep(dt*2)

        print('position and quaternion at end:\n', p.getBasePositionAndOrientation(oid))
        p.disconnect()


def generate_depth_images(shapes=None):
    shapes = get_relevant_shapes(shapes)

    for shape in shapes:
        mesh = burg.io.load_mesh(os.path.join(shape_dir_transformed, shape + '.obj'))

        # gpnet camera parameters (not that it matters much..)
        camera = burg.scene.Camera()
        camera.set_resolution(320, 240)
        camera.set_intrinsic_parameters(fx=350, fy=350, cx=160, cy=120)
        renderer = burg.render.MeshRenderer(images_dir, camera, fn_func=lambda i: f'render{i:d}Depth0001',
                                            fn_type='tum')

        cpg = burg.render.CameraPoseGenerator(cam_distance_min=0.4, cam_distance_max=0.6,
                                              center_point=mesh.get_center())
        poses = cpg.icosphere(subdivisions=3, scales=1, in_plane_rotations=1, random_distances=True)
        # plot_camera_poses(poses)
        print(f'rendering {len(poses)} depth images')
        renderer.render_depth(mesh, poses, sub_dir=shape)


def create_aabb_file(shapes=None):
    shapes = get_relevant_shapes(shapes)
    aabb_arr = np.empty(len(shapes), dtype=([('objId', 'S32'), ('aabbValue', '<f8', (6,))]))

    for i, shape in enumerate(shapes):
        mesh = burg.io.load_mesh(os.path.join(shape_dir_transformed, shape + '.obj'))
        print(mesh.get_center())
        aabb = mesh.get_axis_aligned_bounding_box()
        aabb_arr[i]['objId'] = shape
        aabb_arr[i]['aabbValue'][:3] = aabb.min_bound
        aabb_arr[i]['aabbValue'][3:6] = aabb.max_bound

    print(aabb_arr)
    np.save(os.path.join(base_dir, 'aabbValue.npy'), aabb_arr)


def browse_shapes(shapes=None):
    shapes = get_relevant_shapes(shapes)

    for shape in shapes:
        fn = os.path.join(shape_dir_transformed, shape + '.obj')
        mesh = burg.io.load_mesh(fn)
        print(f'{shape}\tdims: {burg.mesh_processing.dimensions(mesh)}')
        plane = burg.visualization.create_plane()
        burg.visualization.show_o3d_point_clouds([plane, mesh])


if __name__ == "__main__":
    arguments = parse_args()

    if arguments.ycb_path is None:
        cfg = configparser.ConfigParser()
        cfg.read(arguments.config)
        cfg = cfg['General']
    else:
        cfg = None

    base_dir = arguments.output_dir
    # redirect output to log file
    if arguments.log_file is not None:
        orig_stdout = sys.stdout
        f = open(os.path.join(base_dir, arguments.log_file + '.log'), 'w')
        sys.stdout = f

    print(f'generate dataset - called {time.strftime("%a, %d %b %Y at %H:%M:%S")} with arguments:')
    for key, value in vars(arguments).items():
        print(f'\t{key}:\t{value}')
    print('****************************')

    shapes_fn = os.path.join(arguments.output_dir, 'shapes.csv')
    shape_dir = os.path.join(arguments.output_dir, 'shapes/')
    annotation_dir = os.path.join(arguments.output_dir, 'annotations/candidate/')
    sim_result_dir = os.path.join(arguments.output_dir, 'annotations/simulateResult/')
    tmp_dir = os.path.join(arguments.output_dir, 'annotations/tmp/')

    # make sure all paths exist
    burg.io.make_sure_directory_exists([shape_dir, annotation_dir, sim_result_dir, tmp_dir])

    shape_dir_originals = os.path.join(shape_dir, 'originals')
    shape_dir_transformed = os.path.join(shape_dir, 'transformed')
    shape_dir_vhacd = os.path.join(shape_dir, 'vhacd')

    images_dir = os.path.join(arguments.output_dir, 'images/')

    burg.io.make_sure_directory_exists([shape_dir_originals, shape_dir_transformed, shape_dir_vhacd, images_dir])

    # inspect_meshes()
    if arguments.shape is not None:
        arguments.shape = [arguments.shape]
    # preprocess_shapes(cfg, arguments.ycb_path, arguments.shape)
    # browse_shapes(arguments.shape)
    # see_vhacd_in_sim(arguments.shape)
    # create_grasp_samples(arguments.shape)
    simulate_grasp_samples(arguments.shape)
    # generate_depth_images(arguments.shape)
    # create_aabb_file()
    # visualize_data()
    # show_meshes()

    if arguments.log_file is not None:
        sys.stdout = orig_stdout
        f.close()
