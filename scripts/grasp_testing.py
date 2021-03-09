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

    theta = 15 / 180 * np.pi
    rot_mat = np.asarray([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    grasp = gs[1]
    grasp.translation = np.asarray([0, 0, 0])
    grasp.rotation_matrix = rot_mat
    gs[1] = grasp

    dist = burg.grasp.pairwise_distances(gs[0], gs[1])
    print('computation of pairwise_distances (15 degree and 3 mm)', dist.shape, dist)
    dist = gs[0].distance_to(gs[1])
    print('computation of distance_to (15 degree and 3 mm)', dist.shape, dist)

    t1 = timer()
    print('computation of coverage 20/50:', burg.grasp.coverage_brute_force(gs, gs[0:20]))
    print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.grasp.coverage(gs, gs[0:20], print_timings=True))
    print('this took:', timer() - t1, 'seconds')

    grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
    grasp_file = '061_foam_brick/grasps.h5'
    grasp_set, com = burg.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

    t1 = timer()
    # this is unable to allocate enough memory for len(gs)=500
    #print('computation of coverage 20/50:', gdt.grasp.coverage_brute_force(grasp_set, gs))
    #print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.grasp.coverage(grasp_set, gs, print_timings=True))
    print('in total, this took:', timer() - t1, 'seconds')


def test_antipodal_grasp_sampling():
    # read config file
    cfg_fn = '../config/config.cfg'
    cfg_fn = os.path.abspath(cfg_fn)
    print('using config file in:', cfg_fn)

    cfg = configparser.ConfigParser()
    cfg.read(cfg_fn)

    # object lib
    print('read object library')
    object_library = burg.io.read_object_library(cfg['General']['object_lib_fn'])
    print('found', len(object_library), 'objects')

    # find the foamBrick
    target_obj = []
    for obj in object_library:
        print(obj.name)
        if obj.name == 'cheezeIt':
            target_obj = obj
            print('using', target_obj.name, 'object')

    # read the mesh as point cloud
    print('reading mesh and converting to point cloud')
    mesh_fn = os.path.join(
        cfg['General']['models_dir'],
        target_obj.name +
        cfg['General']['mesh_fn_ext']
    )
    point_cloud = burg.mesh_processing.convert_mesh_to_point_cloud(mesh_fn, with_normals=True)

    # add them to object info
    target_obj.point_cloud = point_cloud
    target_obj.point_cloud[:, 0:3] -= target_obj.displacement

    grasp_set = burg.sampling.sample_antipodal_grasps(
        target_obj.point_cloud,
        burg.gripper.ParallelJawGripper(),
        n=5,
        max_sum_of_angles=30,
        visualize=False
    )
    print('grasp_set', grasp_set.internal_array.shape)

    print('saving grasp set to', SAVE_FILE)
    with open(SAVE_FILE, 'wb') as f:
        np.save(f, grasp_set.internal_array)

    # gdt.visualization.show_np_point_clouds(target_obj.point_cloud)


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


if __name__ == "__main__":
    print('hi')
    # test_distance_and_coverage()
    test_antipodal_grasp_sampling()
    # test_rotation_to_align_vectors()
    # test_angles()
    print('bye')
