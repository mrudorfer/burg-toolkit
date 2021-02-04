import os
from timeit import default_timer as timer
import numpy as np
import grasp_data_toolkit as gdt

# testing the distance function
initial_translations = np.random.random((50, 3))
gs = gdt.grasp.GraspSet.from_translations(initial_translations)

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

dist = gdt.grasp.pairwise_distances(gs[0], gs[1])
print('computation of pairwise_distances (15 degree and 3 mm)', dist.shape, dist)
dist = gs[0].distance_to(gs[1])
print('computation of distance_to (15 degree and 3 mm)', dist.shape, dist)

t1 = timer()
print('computation of coverage 20/50:', gdt.grasp.coverage_brute_force(gs, gs[0:20]))
print('this took:', timer() - t1, 'seconds')

t1 = timer()
print('coverage kd-tree:', gdt.grasp.coverage(gs, gs[0:20], print_timings=True))
print('this took:', timer() - t1, 'seconds')

grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
grasp_file = '061_foam_brick/grasps.h5'
grasp_set, com = gdt.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

t1 = timer()
# this is unable to allocate enough memory for len(gs)=500
#print('computation of coverage 20/50:', gdt.grasp.coverage_brute_force(grasp_set, gs))
#print('this took:', timer() - t1, 'seconds')

t1 = timer()
print('coverage kd-tree:', gdt.grasp.coverage(grasp_set, gs, print_timings=True))
print('in total, this took:', timer() - t1, 'seconds')


