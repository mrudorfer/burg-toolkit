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

import burg_toolkit as burg


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


def create_point_clouds_from_depth_images(scene):
    pass


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






