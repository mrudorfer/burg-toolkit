import os

import numpy as np
import open3d as o3d

import burg_toolkit as burg

# here's a collection of paths... choose correct combination in options
# base directories
exp_dir_linux = '/home/rudorfem/dev/exp_GPNet_Deco/'
exp_dir_win = 'E:/data/UoB/research/BURG/ShapeGrasp/exp_GPNet_Deco/'

shapes_dir_linux = '/home/rudorfem/dev/3d_Grasping/GPNet/simulator/gpnet_data/processed/'
shapes_dir_win = 'E:/Projekte/3d_Grasping/GPNet/simulator/gpnet_data/processed'

# experiments directories
basel_noLRsched = 'GPNet_basel_27march_tanh_grid_noLrSched/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/'
basel_wLRsched = 'GPNet_basel_27march_tanh_grid_withLrSched/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/'
deco_noLRsched = 'deco_fixLR_noSched_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/'
deco_wLRsched = 'deco_fixLR_lrSched_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/'

#######################
# options
SHOW_GRASPS_BEFORE_NMS = True
EPOCH = 450
SHAPES_DIR = shapes_dir_linux
BASE_EXP_DIR = exp_dir_linux
EXP_DIR = deco_noLRsched
#######################

# files
sim_log_fn = f'test/epoch{EPOCH}/nms_poses_view0_log.csv'
all_grasps_dir = f'test/epoch{EPOCH}/view0/'


# dict for mapping the score from simulation to color
s2c = {
    0: ([0.1, 0.8, 0.1], 'success', 'green'),
    1: ([0.8, 0.1, 0.1], 'collision with ground', 'red'),
    2: ([0.4, 0.1, 0.1], 'collision with object', 'dark red'),
    3: ([0.1, 0.1, 0.8], 'untouched', 'blue'),
    4: ([0.1, 0.1, 0.4], 'incorrect contact', 'dark blue'),
    5: ([0.1, 0.4, 0.1], 'fallen', 'dark green'),
    6: ([0.4, 0.4, 0.4], 'timeout', 'gray')
}


def score_to_color(s):
    if s in s2c.keys():
        return s2c[s][0]
    else:
        return [0, 0, 0]


def read_simulation_log_file(log_fn):
    """
    :param log_fn: the path to the log file (output from GPNet)

    :return: dictionary with shape id (string) as key and grasp.GraspSet as value
    """
    # need temporary structure because data is unordered in log file
    tmp_objects = {}
    with open(log_fn, 'r') as logReader:
        lines = logReader
        for line in lines:
            msg = line.strip()
            msgList = msg.split(',')
            objId, success_status = msgList[0], int(msgList[2])
            # create np array with position, quaternion and status
            grasp_array = np.empty(8)
            for i in range(4, 11):
                grasp_array[i-4] = float(msgList[i])
            grasp_array[7] = success_status

            # initialize
            if objId not in tmp_objects.keys():
                tmp_objects[objId] = {
                    'grasp_list': []
                }
            # fill in data
            tmp_objects[objId]['grasp_list'].append(grasp_array)

    # now we need to get the data in our data types
    grasp_sets = {}
    for key, item in tmp_objects.items():
        # the quaternion is in shape (x, y, z, w)
        # our implementation expects (w, x, y, z), let's change it accordingly
        grasps = np.asarray(item['grasp_list'])
        grasps[:, [3, 4, 5, 6]] = grasps[:, [6, 3, 4, 5]]

        gs = burg.grasp.GraspSet.from_translations_and_quaternions(translations=grasps[:, 0:3],
                                                                   quaternions=grasps[:, 3:7])
        gs.scores = grasps[:, 7]

        grasp_sets[key] = gs

    return grasp_sets


def inspect_grasps(grasp_sets, shape_dir, npz_files_dir=None):
    # now do the visualization
    print('showing', len(grasp_sets.keys()), 'objects with grasps')

    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.02,
                                                    finger_thickness=0.003)  # probably needs some adjustments

    # ground plane
    l, w, h = 0.3, 0.3, 0.001
    ground_plane = o3d.geometry.TriangleMesh.create_box(l, w, h)
    ground_plane.compute_triangle_normals()
    ground_plane.translate(np.array([-l / 2, -w / 2, -h]))

    for obj, grasp_set in grasp_sets.items():
        # print a summary
        print('shape', obj, 'has', len(grasp_set), 'grasps')
        if obj != 'fa23aa60ec51c8e4c40fe5637f0a27e1':
            continue
        scores = grasp_set.scores
        for key in s2c.keys():
            print(f'* {(scores == key).sum()} x {s2c[key][1]} ({s2c[key][2]})')

        mesh_fn = os.path.join(shape_dir, obj + '.obj')
        obj_mesh = burg.io.load_mesh(mesh_fn)

        burg.visualization.show_grasp_set(
            [obj_mesh, ground_plane],
            grasp_set,
            score_color_func=score_to_color,
            gripper=gripper_model
        )

        if npz_files_dir is not None:
            file_path = os.path.join(npz_files_dir, obj + '.npz')
            if not os.path.isfile(file_path):
                print(f'{file_path} is not a file, skipping this one')
                continue
            with np.load(file_path) as data:
                positions = data['centers']
                orientations = data['quaternions']  # w, x, y, z presumably
                gs = burg.grasp.GraspSet.from_translations_and_quaternions(translations=positions,
                                                                           quaternions=orientations)
                gs.scores = data['scores']
                n_show = 800
                print(f'showing {n_show} of {len(gs)} grasps')
                burg.visualization.show_grasp_set([obj_mesh, ground_plane], gs, n=n_show,
                                                  score_color_func=lambda s: [0, 2*(s-0.5), 0],
                                                  gripper=gripper_model)


if __name__ == "__main__":
    print('hi')
    gs_dict = read_simulation_log_file(os.path.join(BASE_EXP_DIR, EXP_DIR, sim_log_fn))
    inspect_grasps(gs_dict, SHAPES_DIR, os.path.join(BASE_EXP_DIR, EXP_DIR, all_grasps_dir))
    print('bye')
