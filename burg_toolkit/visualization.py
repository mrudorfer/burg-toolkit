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
        if type(o3d_pc) is o3d.geometry.TriangleMesh:
            if o3d_pc.has_vertex_colors():
                continue
        if type(o3d_pc) is o3d.geometry.PointCloud:
            if o3d_pc.has_colors():
                continue
        color = np.asarray(colormap(color_idx)[0:3])
        o3d_pc.paint_uniform_color(color)
        color_idx = (color_idx + 1) % colormap.N

    return point_clouds


def _get_scene_geometries(scene: scene.Scene, object_library, with_bg_objs=True, colorize=True):
    """
    gathers list of o3d meshes or point clouds for the given scene

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

        if obj_type.mesh is not None:
            o3d_obj = o3d.geometry.TriangleMesh(obj_type.mesh)
        elif obj_type.point_cloud is not None:
            o3d_obj = o3d.geometry.PointCloud(obj_type.point_cloud)
        else:
            raise ValueError('no mesh or point cloud available for object in object library')

        # transform point cloud to correct pose
        # apply displacement (meshes were being centered in MATLAB)
        o3d_obj.translate(-obj_type.displacement)

        # apply transformation according to scene
        o3d_obj.transform(obj.pose)

        o3d_pcs.append(o3d_obj)

    # also add background objects
    if with_bg_objs:
        for bg_obj in scene.bg_objects:
            if bg_obj.mesh is not None:
                o3d_obj = o3d.geometry.TriangleMesh(bg_obj.mesh)
            elif bg_obj.point_cloud is not None:
                o3d_obj = o3d.geometry.PointCloud(bg_obj.point_cloud)
            else:
                raise ValueError('no mesh or point cloud available for bg_object in scene')

            o3d_obj.transform(bg_obj.pose)
            o3d_pcs.append(o3d_obj)

    if colorize:
        colorize_point_clouds(o3d_pcs)

    return o3d_pcs


def show_full_scene_point_cloud(scene: scene.Scene, object_library, with_bg_objs=True):
    """
    shows the complete (ground truth) point cloud of a scene

    :param scene: a core_types.Scene object
    :param object_library: list of core_types.ObjectType objects
    :param with_bg_objs: whether to show background objects as well

    :return: returns when viewer is closed by user
    """
    o3d_pcs = _get_scene_geometries(scene, object_library, with_bg_objs=with_bg_objs)

    # and visualize
    show_o3d_point_clouds(o3d_pcs)


def show_aligned_scene_point_clouds(scene: scene.Scene, views, object_library):
    """
    shows the full point cloud and overlays the partial point cloud from a view (or a list of views)

    :param scene: the scene
    :param views: instance of core_types.CameraView, or list of views
    :param object_library: list of object types

    :return: returns when the user closes the viewer
    """

    o3d_pcs = _get_scene_geometries(scene, object_library, with_bg_objs=True)

    if not type(views) is list:
        views = [views]

    for view in views:
        o3d_pcs.append(view.to_point_cloud())

    show_o3d_point_clouds(o3d_pcs)
