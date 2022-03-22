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
    # burg.visualization.show_grasp_set(objects=[scene], gs=grasp,
    #                                   gripper=burg.gripper.TwoFingerGripperVisualisation())

    grippers = {
        burg.gripper.BarrettHand2F: [1.0, 1.0],
        burg.gripper.BarrettHand: [1.0, 1.0],
        burg.gripper.EZGripper: [1.0, 1.0],
        burg.gripper.Robotiq3F: [1.0, 1.0],
        burg.gripper.Kinova3F: [1.0, 1.0],
        burg.gripper.Franka: [1.0, 1.0],
        burg.gripper.WSG32: [1.45, 1.0],
        burg.gripper.WSG50: [1.0, 1.0],
        burg.gripper.Sawyer: [1.1, 1.0],
        burg.gripper.Robotiq2F85: [1.0, 0.9],
        burg.gripper.Robotiq2F140: [1.0, 0.8],
    }

    for gripper, [gripper_size, open_scale] in grippers.items():
        print(gripper.__name__)
        # sim.configure_recording(f'/home/rudorfem/tmp/grasp-vids/{gripper.__name__}/', camera, camera_pose, fps=30)
        sim.execute_grasp(gripper, grasp, target_obj, gripper_scale=gripper_size, gripper_opening_width=open_scale)


if __name__ == '__main__':
    main()
