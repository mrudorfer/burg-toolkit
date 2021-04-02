import os

import numpy as np
import open3d as o3d

import burg_toolkit as burg

# here's a collection of paths... choose correct ones in main (end of file)
log_fn_basel = '/home/rudorfem/dev/exp_grasping/exp_GPNet/basel_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'
log_fn_deco15k = '/home/rudorfem/dev/exp_grasping/exp_GPNet/deco_wPretrain_bnldrop_max15k_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'
log_fn_deco20k = '/home/rudorfem/dev/exp_grasping/exp_GPNet/deco_wPretrain_bnldrop_max20k_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'

log_fn_basel_win = 'E:/data/UoB/research/BURG/ShapeGrasp/[BEFORE_FIX_LR]exp_grasping/exp_GPNet_orig/exp_GPNet_basel_wGrid_tanh/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'
log_fn_deco20k_win = "E:/data/UoB/research/BURG/ShapeGrasp/[BEFORE_FIX_LR]exp_grasping/exp_GPNet/deco_wPretrain_bnldrop_max20k_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv"

# note that simulator uses urdf folder instead of these processed obj files
shapes_dir_linux = '/home/rudorfem/dev/3d_Grasping/GPNet/simulator/gpnet_data/processed/'
shapes_dir_win = 'E:/Projekte/3d_Grasping/GPNet/simulator/gpnet_data/processed'

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


def read_log_file(log_fn):
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

        gs = burg.grasp.GraspSet.from_translations_and_quaternions(grasps[:, 0:7])
        gs.scores = grasps[:, 7]

        grasp_sets[key] = gs

    return grasp_sets


def inspect_grasps(grasp_sets, shape_dir):
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


if __name__ == "__main__":
    print('hi')
    # gs_dict = read_log_file(log_fn_deco20k_win)
    gs_dict = read_log_file(log_fn_basel_win)
    inspect_grasps(gs_dict, shapes_dir_win)
    print('bye')
