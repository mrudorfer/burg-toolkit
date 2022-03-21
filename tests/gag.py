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
    target_obj = scene.objects[0]

    sim = burg.sim.GraspSimulator(scene, verbose=True)
    pose = np.eye(4)
    pose[0:2, 3] = 0.10
    pose[2, 3] = 0.02
    # pose[2, 3] = 0.1
    grasp = burg.Grasp()
    grasp.pose = pose

    # for recording stuff
    centroid = burg.mesh_processing.centroid(target_obj.get_mesh())
    camera_position = [0.2, -0.4, 0.3 + centroid[2]]
    look_at = np.array(centroid)
    look_at[2] += 0.1
    camera_pose = burg.util.look_at(position=camera_position, target=look_at, flip=True)
    camera = burg.render.Camera.create_kinect_like()

    # sim.configure_recording('/home/rudorfem/tmp/grasp-vids/franka/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('franka', grasp, target_obj)
    return
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/ezgripper/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('ezgripper', grasp, target_obj)
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/wsg_32/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('wsg_32', grasp, target_obj, gripper_scale=1.45, gripper_opening_width=1)
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/wsg_50/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('wsg_50', grasp, target_obj, gripper_opening_width=1)
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/sawyer/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('sawyer', grasp, target_obj, gripper_scale=1.1, gripper_opening_width=1)
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/robotiq_2f_85/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('robotiq_2f_85', grasp, target_obj, gripper_opening_width=0.9)
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/robotiq_2f_140/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('robotiq_2f_140', grasp, target_obj, gripper_opening_width=0.8)
    sim.configure_recording('/home/rudorfem/tmp/grasp-vids/barrett_hand_2f/frame', camera, camera_pose, fps=30)
    sim.execute_grasp('barrett_hand_2f', grasp, target_obj, gripper_opening_width=1.0)

    # sim.execute_grasp('rg2', grasp, target_obj, gripper_opening_width=0.8)


if __name__ == '__main__':
    main()
