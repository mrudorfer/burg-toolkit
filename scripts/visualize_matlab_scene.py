import os

import argparse
import configparser
import numpy as np
import burg_toolkit as burg

parser = argparse.ArgumentParser(description='visualize a scene generetad with MATLAB scene generator project')
parser.add_argument('-c', '--config_fn', default='../config/config.cfg', type=str, metavar='FILE',
                    help='path to config file')

print('hi, it''s me, visualize_matlab_scene.py')

# read config file
cfg_fn = parser.parse_args().config_fn
print('using config file in:', os.path.abspath(cfg_fn))

cfg = configparser.ConfigParser()
cfg.read(cfg_fn)

reader = burg.io.BaseviMatlabScenesReader(cfg['General'])

print('read object library')
object_library, index2name = reader.read_object_library()
object_library.yell()

# get file names of scene data, pick one and read that stuff
file_names = reader.get_scene_filenames()
files = file_names[0]
print('loading scene data from the following files:')
print('\theap:', os.path.abspath(files['heap_fn']))
print('\timages:', os.path.abspath(files['image_data_fn']))
scene = reader.read_scene_files(files)
print('scene has', len(scene.objects), 'objects and', len(scene.views), 'views')
[print(instance) for instance in [*scene.bg_objects, *scene.objects]]

# visualize point cloud
print('visualizing scene point cloud')
burg.visualization.show_aligned_scene_point_clouds(scene, scene.views)

print('generating urdf files for object library')
object_library.generate_urdf_files('../data/tmp')

target_object_instance = scene.objects[2]
print(f'sampling and visualizing grasps for {target_object_instance.object_type.identifier}')

gripper_model = burg.gripper.ParallelJawGripper(ref_frame=burg.gripper.RefFrame.TCP,
                                                finger_length=0.03,
                                                finger_thickness=0.003)
ags = burg.sampling.AntipodalGraspSampler()
ags.mesh = target_object_instance.object_type.mesh
ags.gripper = gripper_model
ags.verbose = False
gs = ags.sample(1)
burg.visualization.show_grasp_set([ags.mesh], gs[0], gripper_mesh=gripper_model.mesh, n=1)

sim = burg.sim.SceneGraspSimulator(None, None, scene, verbose=True)
sim.simulate_grasp_set(gs[0])
