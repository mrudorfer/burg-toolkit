import os

import numpy as np
import open3d as o3d

import grasp_data_toolkit as gdt

print('hi')

log_fn_basel = '/home/rudorfem/dev/exp_grasping/exp_GPNet/basel_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'
log_fn_deco15k = '/home/rudorfem/dev/exp_grasping/exp_GPNet/deco_wPretrain_bnldrop_max15k_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'
log_fn_deco20k = '/home/rudorfem/dev/exp_grasping/exp_GPNet/deco_wPretrain_bnldrop_max20k_tanh_grid/gridlen22_gridnum10/bs1_wd0.0001_lr0.001_lamb0.01_ratio1.0_posi0.3_sgd/test/epoch500/nms_poses_view0_log.csv'
# log_fn = '/home/rudorfem/dev/3d_Grasping/GPNet/simulator/gpnet_data/prediction/nms_poses_view0_log.csv'
shapes_dir = '/home/rudorfem/dev/3d_Grasping/GPNet/simulator/gpnet_data/processed/'
# note that simulator uses urdf folder instead of processed folder

log_fn = log_fn_basel

# read log file
# need dict structure because data is unordered in log file
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

# now we need to get the data in order
grasp_sets = {}
for key, item in tmp_objects.items():
    # the quaternion is in shape (x, y, z, w)
    # our implementation expects (w, x, y, z), let's change it accordingly
    grasps = np.asarray(item['grasp_list'])
    grasps[:, [3, 4, 5, 6]] = grasps[:, [6, 3, 4, 5]]

    gs = gdt.grasp.GraspSet.from_translations_and_quaternions(grasps[:, 0:7])
    gs.scores = grasps[:, 7]

    grasp_sets[key] = gs

# now do the visualization
print('showing', len(grasp_sets.keys()), 'objects with grasps')

gripper_model = gdt.gripper.ParallelJawGripper(ref_frame=gdt.gripper.RefFrame.TCP,
                                               finger_length=0.02)  # probably needs some adjustments

# ground plane
l, w, h = 0.3, 0.3, 0.001
ground_plane = o3d.geometry.TriangleMesh.create_box(l, w, h)
ground_plane.compute_triangle_normals()
ground_plane.translate(np.array([-l / 2, -w / 2, -h]))

for obj, grasp_set in grasp_sets.items():
    # print a summary
    print('shape', obj, 'has', len(grasp_set), 'grasps')
    scores = grasp_set.scores
    print('* ', (scores==0).sum(), 'successful (green)')
    print('* ', (scores==1).sum(), 'collision with ground (red)')
    print('* ', (scores==2).sum(), 'collision with obj (dark red)')
    print('* ', (scores==3).sum(), 'untouched (blue)')
    print('* ', (scores==4).sum(), 'incorrect contact (dark blue)')
    print('* ', (scores==5).sum(), 'fallen (dark green)')
    print('* ', (scores==6).sum(), 'timeout (gray) ')

    # prepare the visualization
    # object
    mesh_fn = os.path.join(shapes_dir, obj + '.obj')
    point_cloud = gdt.mesh_processing.convert_mesh_to_point_cloud(mesh_fn, with_normals=True)
    obj_pc = gdt.util.numpy_pc_to_o3d(point_cloud)

    # gripper poses
    grippers = []
    for tf, score in zip(grasp_set.poses, grasp_set.scores):
        gripper_vis = o3d.geometry.TriangleMesh(gripper_model.mesh)
        gripper_vis.transform(tf)

        # colorise based on score
        if score == 0:
            # success
            color = [0.1, 0.8, 0.1]
        elif score == 1:
            # collision with ground
            color = [0.8, 0.1, 0.1]
        elif score == 2:
            # collision with obj
            color = [0.4, 0.1, 0.1]
        elif score == 3:
            # untouched
            color = [0.1, 0.1, 0.8]
        elif score in (4, 5):
            # incorrect contact
            color = [0.1, 0.1, 0.4]
        elif score == 5:
            # fallen
            color = [0.1, 0.4, 0.1]
        elif score == 6:
            # time out
            color = [0.4, 0.4, 0.4]
        else:
            # should not happen
            color = [0, 0, 0]

        gripper_vis.paint_uniform_color(color)
        gripper_vis.compute_triangle_normals()
        grippers.append(gripper_vis)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

    obj_list = [obj_pc, ground_plane, frame]
    obj_list.extend(grippers)
    gdt.visualization.show_o3d_point_clouds(obj_list)

print('bye')