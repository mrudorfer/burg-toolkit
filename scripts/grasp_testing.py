import numpy as np
import grasp_data_toolkit as gdt

# testing the distance function
initial_translations = np.zeros((2, 3), dtype=np.float32)
gs = gdt.grasp.GraspSet.from_translations(initial_translations)

theta = 0 / 180 * np.pi
rot_mat = np.asarray([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

grasp = gs[0]
grasp.translation = np.asarray([0, 0, 0.003]) # 1mm
grasp.rotation_matrix = rot_mat
gs[0] = grasp

theta = 15 / 180 * np.pi
rot_mat = np.asarray([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])

grasp = gs[1]
grasp.rotation_matrix = rot_mat
gs[1] = grasp
print('distance:', gs[0].distance_to(gs[1]))
