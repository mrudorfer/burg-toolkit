import configparser
import open3d as o3d

import burg_toolkit as burg

cfg_fn = '../config/config.cfg'
print('using config file in:', cfg_fn)

cfg = configparser.ConfigParser()
cfg.read(cfg_fn)
reader = burg.io.BaseviMatlabScenesReader(cfg['General'])

# load object library
print('read object library')
object_library, index2name = reader.read_object_library()
object_library.yell()

target_object = object_library['flatheadScrewdriver']

mesh = target_object.mesh
burg.mesh_processing.check_properties(mesh)
inertia, com = burg.mesh_processing.compute_mesh_inertia(mesh, target_object.mass)
print('inertia:\n', inertia)

object_library.generate_urdf_files('../data/tmp', overwrite_existing=True)
