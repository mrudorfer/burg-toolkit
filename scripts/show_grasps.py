import configparser
import os
import open3d as o3d
import burg_toolkit as burg

print('hi, it''s me, show_grasps.py')

# read config file
cfg_fn = '../config/config.cfg'
print('using config file in:', cfg_fn)

cfg = configparser.ConfigParser()
cfg.read(cfg_fn)

# object lib
print('read object library')
object_library = burg.io.read_object_library(cfg['General']['object_lib_fn'])
print('found', len(object_library), 'objects')

target_obj = []
for obj in object_library:
    if obj.name == 'foamBrick':
        target_obj = obj
        print('using', target_obj.name, 'object')
        break

# read the meshes as point clouds
print('reading mesh and converting to point cloud')
mesh_fn = os.path.join(
        cfg['General']['models_dir'],
        target_obj.name +
        cfg['General']['mesh_fn_ext']
)
point_cloud = burg.mesh_processing.convert_mesh_to_point_cloud(mesh_fn, with_normals=True)

# add them to object info
target_obj.point_cloud = point_cloud
o3d_pc = burg.util.numpy_pc_to_o3d(point_cloud)
o3d_pc.translate(-target_obj.displacement)

print('object displacement:', target_obj.displacement[:])

grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
grasp_file = '061_foam_brick/grasps.h5'
grasp_set, com = burg.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

complete_grasps = o3d.geometry.PointCloud()
complete_grasps.points = o3d.utility.Vector3dVector(grasp_set.translations)
print('com', complete_grasps.get_center())
complete_grasps.translate(-com)
print('com', complete_grasps.get_center())

# divide the complete set of grasps into subsets for better visualization
# we also already put the object point cloud into the list, so it will be object 0 in visualization
grasps_list = [o3d_pc, complete_grasps]
subset_size = 500000
print('number of grasps', len(grasp_set))
#for i in range(0, grasp_data.shape[0], subset_size):
#    grasps = o3d.geometry.PointCloud()
#    print(i)
#    grasps.points = o3d.utility.Vector3dVector(grasp_data[i:min(grasp_data.shape[0], i+subset_size), 0:3])
#    grasps_list.append(grasps)

burg.visualization.colorize_point_clouds(grasps_list)
burg.visualization.show_o3d_point_clouds(grasps_list)
