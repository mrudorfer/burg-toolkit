import os

import configparser
import open3d as o3d
import burg_toolkit as burg

print('hi, it''s me, show_grasps.py')

# read config file
cfg_fn = '../config/config.cfg'
print('using config file in:', cfg_fn)

cfg = configparser.ConfigParser()
cfg.read(cfg_fn)
reader = burg.io.BaseviMatlabScenesReader(cfg['General'])

# load object library
print('read object library')
object_library, index2name = reader.read_object_library()
[print(f'\t{idx}: {name}') for idx, name in index2name.items()]

target_obj_name = 'foamBrick'
target_obj = object_library[target_obj_name]
print(f'using {target_obj_name} object')

grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
grasp_file = '061_foam_brick/grasps.h5'
grasp_set, com = burg.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

complete_grasps = o3d.geometry.PointCloud()
complete_grasps.points = o3d.utility.Vector3dVector(grasp_set.translations)
print('com', complete_grasps.get_center())
complete_grasps.translate(-com)
print('com', complete_grasps.get_center())

visualization_objects = [target_obj.mesh, complete_grasps]
burg.visualization.show_o3d_point_clouds(visualization_objects)
