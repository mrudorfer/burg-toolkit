import numpy as np
import open3d as o3d

import burg_toolkit as burg


class PoseIndicator(burg.gripper.ParallelJawGripper):
    def __init__(self, mesh):
        super().__init__(mesh=mesh)


def visualize_camera_positions(generator):
    print('number of poses', generator.number_of_poses)
    count = 0
    for _ in enumerate(generator.poses()):
        count += 1
    print('actual count', count)

    poses = np.empty(shape=(generator.number_of_poses, 4, 4))
    for i, pose in enumerate(generator.poses()):
        poses[i] = pose

    gs = burg.grasp.GraspSet.from_poses(poses)
    indicator = PoseIndicator(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    burg.visualization.show_grasp_set([o3d.geometry.TriangleMesh.create_sphere(radius=0.1)], gs, with_plane=True,
                                      gripper=indicator)


def look_at_func():
    print('testing look at function')
    print(burg.util.look_at([1, 1, 1], target=[0, 0, 0], up=[1, 1, 1]))

    positions = np.arange(60).reshape(-1, 3) / 60
    positions[:, 0] *= np.arange(len(positions)) / len(positions)
    positions[:, 1] *= np.arange(len(positions)) / len(positions)
    poses = burg.util.look_at(positions, target=[0, 0, 0], up=[0, 0, 1])
    print(poses)

    gs = burg.grasp.GraspSet.from_poses(poses)
    indicator = PoseIndicator(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    burg.visualization.show_grasp_set([o3d.geometry.TriangleMesh.create_sphere(radius=0.1)], gs, with_plane=True,
                                      gripper=indicator)


if __name__ == "__main__":
    look_at_func()
    visualize_camera_positions(
        burg.camera_pose_generators.RandomCameraPoseGenerator(number_of_poses=60, lower_hemisphere=False,
                                                              cam_distance_min=0.5, cam_distance_max=1.0))
    visualize_camera_positions(burg.camera_pose_generators.IcoSphereCameraPoseGenerator(in_plane_rotations=3,
                                                                                        subdivisions=2,
                                                                                        scales=1,
                                                                                        random_distances=True))
