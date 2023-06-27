"""
===============================
Point Cloud Rendering Tutorial
===============================

This script is meant as a tutorial for rendering point clouds for objects and scenes.
It is recommended to go through the `setup_scenes_example.py` first, so you know how to work with objects and scenes.
We will build upon an existing scene and render a full point cloud as well as partial point clouds from depth
images.
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt

import burg_toolkit as burg


FRAME_SIZE = 0.03


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=str,
                        default='../examples/scenes/scene01.yaml',
                        help='path to a scene file')
    return parser.parse_args()


def sample_complete_point_clouds(scene):
    print('*************************')
    print('We are now creating a full point cloud of the first object. This can be done by sampling '
          '(pseudo-random) points from the mesh surface.')

    # first we have to get the 3d mesh model of the object instance
    mesh = scene.objects[0].get_mesh()

    # now we can apply the poisson_disk_sampling method from the mesh_processing module.
    # poisson disk sampling makes the points spread more regularly over the object's surface, by avoiding areas
    # that are too sparsely or too densely populated. this is preferably over just random sampling.
    # tip:
    # by adjusting the parameter `radius`, you can change the density of the point cloud. essentially, we are sampling
    # more points than required, and then we are removing points when there are multiple points within that radius.
    point_cloud = burg.mesh_processing.poisson_disk_sampling(mesh, radius=0.003)

    # the result is an Open3D.geometry.PointCloud object, which can be visualised using the burg_toolkit
    print('The visualizer now displays the scene and the sampled point cloud. However, you are likely not seeing the '
          'point cloud very well, because it is in the same location as the object mesh. We will fix that in a sec.')
    print('<close the visualizer to continue>')
    burg.visualization.show_geometries([scene, point_cloud])

    # let's remove the first object from the scene to help see the point cloud
    first_obj_instance = scene.objects.pop(0)  # this removes the object from the scene and returns it
    print('We now show the same scene, but removed the first object so that we can see the sampled point cloud.')
    print('<close the visualizer to continue>')
    burg.visualization.show_geometries([scene, point_cloud])
    scene.objects.insert(0, first_obj_instance)  # add the object to the scene again

    print('*************************')
    print('Let us now create a point cloud of the entire scene. (This might take a moment.)')
    # get all meshes that belong to our scene
    meshes = scene.get_mesh_list(with_plane=True)  # you can choose whether you want to include the plane or not

    # we are creating an empty list where we will add in all the point clouds
    # then we go through the list of meshes and use the poisson disk sampling on each of them
    point_clouds = []
    for mesh in meshes:
        point_cloud = burg.mesh_processing.poisson_disk_sampling(mesh)
        point_clouds.append(point_cloud)

    # and that's all! let's see.
    burg.visualization.show_geometries(point_clouds)  # point_clouds are already in a list, so we can skip the brackets

    print('*************************')
    print('Congratulations, you now know how to sample complete point clouds from objects or scenes!')
    print('*************************')


def create_point_clouds_from_depth_images(scene):
    print('*************************')
    print('In order to create partial point clouds from depth images, we first need to get a depth image.\n'
          'This is done assuming a certain camera type and synthetically creating (also called "rendering")\n'
          'an image. From that image we can extract the xyz coordinates for each pixel using the camera\n'
          'parameters.')

    # let us create a camera first
    # a camera has certain parameters (you can look up the documentation of the Camera class), including the
    # resolution of the images, focal point and focal length, etc. (also called "camera intrinsics")
    # we just assume a camera that has parameters similar to a Microsoft Kinect RGB-D camera.
    camera = burg.render.Camera.create_kinect_like()
    camera_position = [0.0, 0.0, 0.2]  # x y z coordinates where we would like the camera to be
    # a pose is position + orientation and there are several representations such as quaternions, Euler angles, etc.
    # we use a 4x4 homogenous transformation matrix as representation. the top left 3 rows and 3 columns are the
    # 3x3 rotation matrix and the first 3 elements of the last column are the xyz position.
    # the last row is always [0 0 0 1].
    camera_pose = np.eye(4)  # this is the identity matrix
    camera_pose[:3, 3] = camera_position

    print('The first step is to create a camera and position it in the scene. The camera position and orientation\n'
          'is described by a 4x4 homogenous transformation matrix. The top left 3 rows and 3 columns are the\n'
          '3x3 rotation matrix and the first 3 elements of the last column are the xyz position. The last row\n'
          'is always [0 0 0 1].')
    print(f'We choose the following camera pose - the 4x4 homogenous transformation matrix:\n {camera_pose}')
    print(f'The rotation matrix is in the top left of the transformation matrix:\n {camera_pose[:3, :3]}')
    print(f'And the position of the camera are in the last column of the transformation matrix:\n {camera_pose[:3, 3]}')

    print('<press any key to continue>')
    input()

    print('For understanding the camera pose, we visualize the relevant coordinate systems.\n'
          'The convention for visualizing coordinate systems is, that the x-axis is red, the y-axis is green, and\n'
          'the z-axis is blue (XYZ = RGB). You can see the reference frame in the corner of the scene''s ground\n'
          'plane. The blue arrow and hence the z-axis points upwards.\n'
          'Since we defined our camera to have a z coordinate of z=0.2, it will be floating 0.2m up in the air.\n'
          'The orientation is currently the same as the reference frame.\n'
          'Feel free to change the camera position, re-run this script and see how the visualisation changes.')
    print('<close the visualizer to continue>')

    # let us visualise the camera's pose
    # create_frame generates a mesh object of a coordinate system for visualization purposes
    # we can adjust its size as well as its pose
    scene_origin_vis = burg.visualization.create_frame(size=FRAME_SIZE, pose=np.eye(4))
    camera_vis = burg.visualization.create_frame(size=FRAME_SIZE, pose=camera_pose)
    burg.visualization.show_geometries([scene, camera_vis, scene_origin_vis])

    print('We have now defined the position of the camera, but which way is it looking?\n'
          'Generally, the camera view is aligned with the z-axis. There are different conventions though:\n'
          '1) The z-axis points forwards, x-axis to the right, and y-axis down.\n'
          '2) The z-axis points backwards, x-axis to the right, and y-axis up.\n'
          'E.g. OpenGL uses 2), while OpenCV uses 1). Unity or other tools use yet other conventions. You can\n'
          'read more about it here:\n'
          'https://medium.com/check-visit-computer-vision/converting-camera-poses-from-opencv-to-opengl-can-be-easy-27ff6c413bdb')
    print('For our example here, we will use convention 2), i.e. the camera looks along the negative z-axis, or, \n'
          'in other words, the z-axis points away from the target that we are looking at.\n'
          'The visualizer now shows the camera pose when it is looking at the center of our scene.')
    print('<close the visualizer to continue>')

    # let's define first the point that our camera should look at.
    # this will be at the center of our scene, and slightly above the xy-plane (i.e. z = 5cm)
    target = [scene.ground_area[0]/2, scene.ground_area[1]/2, 0.05]

    # we can now use a utility function that automatically computes the required orientation of our camera
    # we need to use flip=True to get the rotation according to the convention number 2)
    camera_pose = burg.util.look_at(camera_position, target, flip=True)

    # and visualise
    camera_vis = burg.visualization.create_frame(size=FRAME_SIZE, pose=camera_pose)
    burg.visualization.show_geometries([scene, camera_vis, scene_origin_vis])

    print('Using the camera and the camera pose, we can now render an RGB and a depth image of the scene.\n'
          'Note that the RGB image (on the left) is not very realistic, as we are not using any textures.\n'
          'The depth image (on the right) encodes the distance to each pixel as colour.')
    print('(Note that your object library may need to have the URDF and VHACD files. Read the comments in this\n'
          'script and check out the setup_scene_example.py if you get an error.)')

    # in order to render images, we are creating a render-engine object
    # there are two render engines available in the BURG toolkit, PyRenderEngine and PyBulletRenderEngine.
    # the latter is utilising PyBullet, so the objects from our object library need to have the URDF and VHACD files
    # if they are missing, check out the setup_scene_example.py on how to create them (note that the example does not
    # overwrite the original library, and instead creates another one with the suffix _roundtrip - so you might need to
    # do that manually.)
    # render_engine = burg.render.PyBulletRenderEngine()  # requires urdf + vhacd
    render_engine = burg.render.PyRenderEngine()  # does not require urdf + vhacd
    render_engine.setup_scene(scene, camera, with_plane=True)  # we can also disable the plane
    rgb_image, depth_image = render_engine.render(camera_pose)  # later, we can re-use the render engine to create more

    print(f'RGB image is of shape {rgb_image.shape}; depth image is of shape {depth_image.shape}.')
    print('<close the visualizer to continue>')

    # show the two images using matplotlib (plt)
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb_image)
    ax[1].imshow(depth_image)
    plt.show()

    # now we create the point cloud from the depth image, using the camera info
    point_cloud = camera.point_cloud_from_depth(depth_image, camera_pose)
    print('And finally, using the camera parameters, we can convert the depth image into a point cloud.')
    print(f'The resulting point cloud is a numpy array of shape {point_cloud.shape}, i.e., it has '
          f'{point_cloud.shape[0]} points.')
    print('You may want to post-process it by removing points that are too far away, or by subsampling it to a\n'
          'desired size. For the latter, you can use burg.sampling.farthest_point_sampling().')
    print('<close the visualizer to continue>')

    burg.visualization.show_geometries([point_cloud, camera_vis, scene_origin_vis])

    print('*************************')
    print('Congratulations, you now know how to render depth images and create point clouds!')
    print('*************************')


if __name__ == "__main__":
    args = parse_args()  # this reads in the arguments given to this script
    print(f'Attempting to load scene from given file: {args.scene}')
    scene, lib, _ = burg.Scene.from_yaml(args.scene)
    print(f'Scene and corresponding object library loaded successfully. Here are their details:')
    print(lib)  # use lib.print_details() if you want to see more details
    print(scene)
    print('<close the visualizer to continue>')
    burg.visualization.show_geometries([scene])  # note the brackets, as we have to pass a list of things to show

    # go through tutorial for sampling complete point clouds (irrespective of view point)
    sample_complete_point_clouds(scene)

    # go through tutorial for creating point clouds from depth
    create_point_clouds_from_depth_images(scene)






