import numpy as np
import open3d as o3d

import burg_toolkit as burg


class PoseIndicator(burg.gripper.ParallelJawGripper):
    def __init__(self, mesh):
        super().__init__(mesh=mesh)


def plot_camera_poses(poses, additional_objects=None):
    gs = burg.grasp.GraspSet.from_poses(poses)
    indicator = PoseIndicator(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    if not additional_objects:
        additional_objects = []
    additional_objects.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.1))
    burg.visualization.show_grasp_set(additional_objects, gs, with_plane=True,
                                      gripper=indicator)


def visualize_camera_positions():
    cpg = burg.render.CameraPoseGenerator(cam_distance_min=0.4, cam_distance_max=0.55)
    random_poses = cpg.random(120)
    ico_poses = cpg.icosphere(subdivisions=3, in_plane_rotations=1, random_distances=True, scales=1)
    print('ico poses:', ico_poses.shape)

    cam_info = np.load('E:/data/UoB/tmp/CameraInfo.npy')
    cam_pos = np.empty((len(cam_info), 3))
    cam_quat = np.empty((len(cam_info), 4))
    for i in range(len(cam_info)):
        cam_pos[i] = cam_info[i][1]
        cam_quat[i] = cam_info[i][2]

    gs = burg.grasp.GraspSet.from_translations_and_quaternions(cam_pos, cam_quat)

    for poses in [random_poses, ico_poses, gs.poses]:
        plot_camera_poses(poses)


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


def do_some_rendering():
    mesh_fn = '../data/samples/flathead-screwdriver/flatheadScrewdriverMediumResolution.ply'
    mesh = burg.io.load_mesh(mesh_fn)
    # mesh.translate([2, 2, 2])

    camera = burg.scene.Camera()
    camera.set_resolution(320, 240)
    camera.set_intrinsic_parameters(fx=350, fy=350, cx=160, cy=120)
    renderer = burg.render.MeshRenderer('../data/tmp/', camera, fn_func=lambda i: f'render{i:d}Depth0001', fn_type='png')

    # cpg = burg.render.CameraPoseGenerator(center_point=[2, 2, 2])
    cpg = burg.render.CameraPoseGenerator(center_point=[0, 0.1, 0.1])
    poses = cpg.icosphere(subdivisions=2, in_plane_rotations=1, scales=1)[:4]
    plot_camera_poses(poses, additional_objects=[mesh])

    renderer.render_depth(mesh, poses, sub_dir='flathead-screwdriver/')


if __name__ == "__main__":
    # look_at_func()
    # visualize_camera_positions()
    do_some_rendering()
