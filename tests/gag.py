import numpy as np
import burg_toolkit as burg


def main():
    burg.log_to_console()
    scene, lib, _ = burg.Scene.from_yaml('/home/rudorfem/tmp/scene.yaml')
    print('hi i am saying sth')
    # burg.visualization.show_geometries([scene])

    def create_test_scene(obj_name='077_rubiks_cube'):
        instance = burg.ObjectInstance.from_stable_pose(lib[obj_name], pose_idx=0, x=0.1, y=0.1)
        return burg.Scene(objects=[instance])

    scene = create_test_scene()

    sim = burg.sim.GraspSimulator(scene, verbose=True)
    pose = np.eye(4)
    pose[0:2, 3] = 0.1
    pose[2, 3] = 0.04
    # pose[2, 3] = 0.1
    grasp = burg.Grasp()
    grasp.pose = pose
    # sim.execute_grasp('franka', grasp, scene.objects[0])
    # sim.execute_grasp('ezgripper', grasp, scene.objects[0])
    sim.execute_grasp('wsg_32', grasp, scene.objects[0], gripper_scale=1.45)
    # sim.execute_grasp('robotiq_2f_85', grasp, scene.objects[0])


if __name__ == '__main__':
    main()
