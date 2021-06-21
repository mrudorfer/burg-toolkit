import argparse
import configparser
import numpy as np
import burg_toolkit as burg


def parse_args():
    parser = argparse.ArgumentParser(description='test the simulation of burg toolkit')
    parser.add_argument('-c', '--config_fn', default='../config/config.cfg', type=str, metavar='FILE',
                        help='path to config file')
    return parser.parse_args()


def simulate_demo(data_conf):
    reader = burg.io.BaseviMatlabScenesReader(data_conf)

    print('read object library')
    object_library, index2name = reader.read_object_library()
    object_library.yell()

    print('generating urdf files for object library')
    object_library.generate_urdf_files('../data/tmp')

    obj_pose = np.array([
            [ 0.88092497, -0.02657804, -0.47250906, -0.0183833 ],
            [ 0.47205707, -0.02167491,  0.88130149,  0.05141494],
            [-0.03366485, -0.99941173, -0.00654762,  0.03823685],
            [ 0.,          0.,          0.,          1.        ]])

    target_object_instance = burg.scene.ObjectInstance(object_library['foamBrick'], pose=obj_pose)

    gripper_model = burg.gripper.Robotiq2F85()
    grasp_pose = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, -0.01],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    g = burg.grasp.Grasp()
    g.pose = grasp_pose

    # show grasp for mesh of object type
    burg.visualization.show_grasp_set([target_object_instance.object_type.mesh], g, gripper=gripper_model)

    # transform grasp to pose of object instance and show that
    g.transform(target_object_instance.pose)
    burg.visualization.show_grasp_set([target_object_instance.get_mesh()], g, gripper=gripper_model, with_plane=True)

    # simulation should look like that as well
    sim = burg.sim.SingleObjectGraspSimulator(target_object=target_object_instance, gripper=gripper_model, verbose=True)
    sim.simulate_grasp_set(g)
    sim.dismiss()


if __name__ == "__main__":
    arguments = parse_args()
    cfg = configparser.ConfigParser()
    # read config file and use the section that contains the data paths
    cfg.read(arguments.config_fn)
    simulate_demo(cfg['General'])
