import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

from . import scene
from . import util


def show_o3d_point_clouds(point_clouds, colorize=True):
    """
    receives a list of point clouds and visualizes them interactively

    :param point_clouds: list of point clouds as o3d objects
    :param colorize: if True, point clouds will be shown in different colors (this is the default)

    :return: returns when the user closed the window
    """
    if colorize:
        colorize_point_clouds(point_clouds)
    o3d.visualization.draw(point_clouds)


def show_np_point_clouds(point_clouds, colorize=True):
    """
    receives a list of point clouds and visualizes them interactively

    :param point_clouds: list of point clouds as numpy arrays Nx3 (or 6?)
    :param colorize: if True, point clouds will be shown in different colors (this is the default)

    :return: returns when the user closed the window
    """

    # first convert from numpy to o3d
    pc_objs = util.numpy_pc_to_o3d(point_clouds)
    if colorize:
        colorize_point_clouds(pc_objs)

    show_o3d_point_clouds(pc_objs)


def colorize_point_clouds(point_clouds, colormap_name='tab20'):
    """
    gets a list of o3d point clouds and adds unique colors to them

    :param point_clouds: list of o3d point clouds
    :param colormap_name: name of the matplotlib colormap to use, defaults to 'tab20'

    :return: the same list of o3d point clouds (but they are also adjusted in-place)
    """

    # this colormap offers 20 different qualitative colors
    colormap = plt.get_cmap(colormap_name)
    color_idx = 0

    for o3d_pc in point_clouds:
        color = np.asarray(colormap(color_idx)[0:3])
        o3d_pc.paint_uniform_color(color)
        color_idx = (color_idx + 1) % colormap.N

    return point_clouds


def _get_object_point_clouds(scene: scene.Scene, object_library, with_bg_objs=True, colorize=True):
    """
    gathers list of o3d point clouds for the given scene

    :param scene: the scene
    :param object_library: list of object types
    :param with_bg_objs: if True, list includes point clouds of background objects as well
    :param colorize: if True, each object gets a unique color

    :return: list of o3d point clouds
    """

    o3d_pcs = []
    for obj in scene.objects:
        # stored indices are 1..14 instead of 0..13 because of MATLAB, so subtract one
        obj_type = object_library[obj.library_index - 1]
        o3d_pc = util.numpy_pc_to_o3d(obj_type.point_cloud)

        # transform point cloud to correct pose
        # apply displacement (meshes were being centered in MATLAB)
        o3d_pc.translate(-obj_type.displacement)

        # apply transformation according to scene
        o3d_pc.transform(obj.pose)

        o3d_pcs.append(o3d_pc)

    # also add background objects
    if with_bg_objs:
        for bg_obj in scene.bg_objects:
            # convert point cloud, apply tf and append
            o3d_pc = util.numpy_pc_to_o3d(bg_obj.point_cloud)
            o3d_pc.transform(bg_obj.pose)
            o3d_pcs.append(o3d_pc)

    if colorize:
        colorize_point_clouds(o3d_pcs)

    return o3d_pcs


def _get_partial_point_cloud_from_view(view: scene.CameraView):
    """
    creates a partial point cloud from the depth image and given intrinsic/extrinsic parameters

    :param view: instance of core_types.CameraView

    :return: an o3d point cloud
    """

    # there is some magic happening here, due to a very strange bug:
    # open3d crashes when I create an o3d image from view.depth_image, but if I just copy its contents to a new
    # image, it seems to work well. I have no clue what is going on here.
    test_image = np.zeros(shape=view.depth_image.shape)
    test_image[:] = view.depth_image[:]

    # o3d can't handle the inf values, so set them to zero
    test_image[test_image == np.inf] = 0

    # create depth image
    o3d_depth_image = o3d.geometry.Image(test_image.astype(np.float32))

    # create point cloud from depth
    pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth_image,
        view.camera.get_o3d_intrinsics(),
        extrinsic=view.camera.pose,
        depth_scale=1.0,
        depth_trunc=1.0,
        stride=2,
        project_valid_depth_only=True
    )

    return pc


def show_full_scene_point_cloud(scene: scene.Scene, object_library, with_bg_objs=True):
    """
    shows the complete (ground truth) point cloud of a scene

    :param scene: a core_types.Scene object
    :param object_library: list of core_types.ObjectType objects
    :param with_bg_objs: whether to show background objects as well

    :return: returns when viewer is closed by user
    """
    o3d_pcs = _get_object_point_clouds(scene, object_library, with_bg_objs=with_bg_objs)

    # and visualize
    show_o3d_point_clouds(o3d_pcs)


def show_partial_point_cloud(view: scene.CameraView):
    """
    shows the scene point cloud generated from a depth image

    :param view: the scene view that is to be shown

    :return: returns when user closes the viewer
    """

    pc = _get_partial_point_cloud_from_view(view)
    show_o3d_point_clouds(pc)


def show_aligned_scene_point_clouds(scene: scene.Scene, views, object_library):
    """
    shows the full point cloud and overlays the partial point cloud from a view (or a list of views)

    :param scene: the scene
    :param views: instance of core_types.CameraView, or list of views
    :param object_library: list of object types

    :return: returns when the user closes the viewer
    """

    o3d_pcs = _get_object_point_clouds(scene, object_library, with_bg_objs=True)

    if not type(views) is list:
        views = [views]

    for view in views:
        o3d_pcs.append(_get_partial_point_cloud_from_view(view))

    show_o3d_point_clouds(o3d_pcs)

