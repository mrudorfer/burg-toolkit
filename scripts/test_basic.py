import configparser
import os
from grasp_data_toolkit import read_write_data, mesh_processing, visualization

print('hi, it''s me, test_basic.py')

# read config file
cfg_fn = '../config/config.cfg'
print('using config file in:', cfg_fn)

cfg = configparser.ConfigParser()
cfg.read(cfg_fn)

# object lib
print('read object library')
object_library = read_write_data.read_object_library(cfg['General']['object_lib_fn'])
print('found', len(object_library), 'objects')

# read the meshes as point clouds
print('reading object meshes and converting to point cloud')
mesh_fns = [
    os.path.join(
        cfg['General']['models_dir'],
        obj.name +
        cfg['General']['mesh_fn_ext']
    ) for obj in object_library
]
point_clouds = mesh_processing.convert_mesh_to_point_cloud(mesh_fns, with_normals=True)

# add them to object info
for obj, pc in zip(object_library, point_clouds):
    obj.point_cloud = pc

# read bg_obj point cloud
print('reading table mesh and converting to point cloud')
table_path = os.path.join(
    cfg['General']['bg_models_dir'],
    cfg['General']['table_fn']
)
table_scale_factor = float(cfg['General']['table_scale_factor'])
table_pc = mesh_processing.convert_mesh_to_point_cloud(table_path, with_normals=True, scale_factor=table_scale_factor)

# get file names of scene data
file_names = read_write_data.get_scene_filenames(cfg['General']['scenes_dir'])

# pick one and read them
files = file_names[0]
print('loading scene data from the following files:')
print('\theap:', os.path.abspath(files['heap_fn']))
print('\timages:', os.path.abspath(files['image_data_fn']))
scene = read_write_data.read_scene_files(files)
print('scene has', len(scene.objects), 'objects and', len(scene.views), 'views')

scene.bg_objects[0].point_cloud = table_pc

# visualize point cloud
print('visualizing scene point cloud')
visualization.show_aligned_scene_point_clouds(scene, scene.views, object_library)


