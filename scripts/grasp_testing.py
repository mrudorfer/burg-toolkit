import os
from timeit import default_timer as timer
import numpy as np
import configparser

import burg_toolkit as burg

SAVE_FILE = os.path.join('..', 'sampled_grasps.npy')


def test_distance_and_coverage():
    # testing the distance function
    initial_translations = np.random.random((50, 3))
    gs = burg.grasp.GraspSet.from_translations(initial_translations)

    theta = 0 / 180 * np.pi
    rot_mat = np.asarray([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    grasp = gs[0]
    grasp.translation = np.asarray([0, 0, 0.003])
    grasp.rotation_matrix = rot_mat
    gs[0] = grasp
    print(grasp)

    theta = 15 / 180 * np.pi
    rot_mat = np.asarray([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    grasp = gs[1]
    grasp.translation = np.asarray([0, 0, 0])
    grasp.rotation_matrix = rot_mat
    gs[1] = grasp
    print(grasp)

    print('average gripper point distances between 20 and 50 elem graspset')
    print(burg.metrics.avg_gripper_point_distances(gs[0:20], gs).shape)

    dist = burg.metrics.combined_distances(gs[0], gs[1])
    print('computation of pairwise_distances (15 degree and 3 mm)', dist.shape, dist)

    t1 = timer()
    print('computation of coverage 20/50:', burg.metrics.coverage_brute_force(gs, gs[0:20]))
    print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.metrics.coverage(gs, gs[0:20], print_timings=True))
    print('this took:', timer() - t1, 'seconds')

    grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
    grasp_file = '061_foam_brick/grasps.h5'
    grasp_set, com = burg.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

    t1 = timer()
    # this is unable to allocate enough memory for len(gs)=500
    #print('computation of coverage 20/50:', gdt.grasp.coverage_brute_force(grasp_set, gs))
    #print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.metrics.coverage(grasp_set, gs, print_timings=True))
    print('in total, this took:', timer() - t1, 'seconds')


def test_antipodal_grasp_sampling():
    # read config file
    cfg_fn = '../config/config.cfg'
    cfg_fn = os.path.abspath(cfg_fn)
    print('using config file in:', cfg_fn)

    cfg = configparser.ConfigParser()
    cfg.read(cfg_fn)
    reader = burg.io.BaseviMatlabScenesReader(cfg['General'])

    # object lib
    print('read object library')
    object_library, index2name = reader.read_object_library()
    [print(f'\t{idx}: {name}') for idx, name in index2name.items()]

    # determine target object
    target_obj_name = 'cheezeIt'
    target_obj = object_library[target_obj_name]
    print('using', target_obj_name, 'object')

    # convert mesh to point cloud
    target_obj.point_cloud = burg.mesh_processing.poisson_disk_sampling(target_obj.mesh)
    gripper = burg.gripper.ParallelJawGripper()
    grasp_set = burg.sampling.sample_antipodal_grasps(
        target_obj.point_cloud,
        gripper,
        n=5,
        max_sum_of_angles=30,
        visualize=True
    )
    print('grasp_set', grasp_set)

    # print('saving grasp set to', SAVE_FILE)
    # with open(SAVE_FILE, 'wb') as f:
    #    np.save(f, grasp_set.internal_array)

    burg.visualization.show_grasp_set([target_obj.mesh], grasp_set, gripper=gripper)


def test_new_antipodal_grasp_sampling():
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    mesh_fn = '../data/samples/flathead-screwdriver/flatheadScrewdriverMediumResolution.ply'

    ags = burg.sampling.AntipodalGraspSampler()
    ags.mesh = burg.io.load_mesh(mesh_fn)
    ags.gripper = gripper_model
    ags.n_orientations = 18
    ags.verbose = True
    ags.max_targets_per_ref_point = 2
    graspset, contacts = ags.sample(100)
    # gs.scores = ags.check_collisions(gs, use_width=False)  # need to install python-fcl
    print('contacts.shape', contacts.shape)
    burg.visualization.show_grasp_set([ags.mesh], graspset, gripper=gripper_model, use_width=False,
                                      score_color_func=lambda s: [s, 1-s, 0], with_plane=True)


def test_rotation_to_align_vectors():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([0, 1, 0])
    r = burg.util.rotation_to_align_vectors(vec_a, vec_b)
    print('vec_a', vec_a)
    print('vec_b', vec_b)
    print('R*vec_a', np.dot(r, vec_a.reshape(3, 1)))

    vec_a = np.array([1, 0, 0])
    vec_b = np.array([-1, 0, 0])
    r = burg.util.rotation_to_align_vectors(vec_a, vec_b)
    print('vec_a', vec_a)
    print('vec_b', vec_b)
    print('R*vec_a', np.dot(r, vec_a.reshape(3, 1)))


def test_angles():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([-1, 0, 0])
    mask = np.array([0])

    a = burg.util.angle(vec_a, vec_b, sign_array=mask)
    print(a)
    print(mask)


def test_cone_sampling():
    axis = [0, 1, 0]
    angle = np.pi/4
    rays = burg.sampling.rays_within_cone(axis, angle, n=100)

    print(rays.shape)


if __name__ == "__main__":
    print('hi')
    # test_distance_and_coverage()
    # test_antipodal_grasp_sampling()
    # test_rotation_to_align_vectors()
    # test_angles()
    # test_cone_sampling()
    test_new_antipodal_grasp_sampling()
    print('bye')
