import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

import numpy as np
import burg_toolkit as burg


def main():
    # env = PybulletSim(gui_enabled=args.gui, gripper_selection=True, num_cam=args.num_cam)
    scene, lib, _ = burg.Scene.from_yaml('/home/rudorfem/tmp/scene.yaml')
    logging.debug('hi i am saying sth')
    # burg.visualization.show_geometries([scene])

    single_object_scene = burg.Scene(objects=[scene.objects[0]])

    sim = burg.sim.GraspSimulator(scene, verbose=True)
    # sim = burg.sim.GraspSimulator(single_object_scene, verbose=True)
    pose = np.eye(4)
    pose[2, 3] = 0.15
    grasp = burg.Grasp()
    grasp.pose = pose
    # sim.execute_grasp('franka', grasp, scene.objects[0])
    sim.execute_grasp('ezgripper', grasp, scene.objects[0])
    # sim.execute_grasp('robotiq_2f_85', grasp, scene.objects[0])


if __name__ == '__main__':
    main()
