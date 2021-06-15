import copy

import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

from . import scene
from . import util
from . import grasp


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


def _get_scene_geometries(scene: scene.Scene, with_bg_objs=True):
    """
    gathers list of o3d meshes or point clouds for the given scene

    :param scene: the scene
    :param with_bg_objs: if True, list includes point clouds of background objects as well

    :return: list of o3d point clouds
    """

    object_list = scene.objects
    if with_bg_objs:
        object_list.extend(scene.bg_objects)

    o3d_pcs = []
    for obj in object_list:
        obj_type = obj.object_type

        if obj_type.mesh is not None:
            o3d_obj = o3d.geometry.TriangleMesh(obj_type.mesh)
        elif obj_type.point_cloud is not None:
            o3d_obj = o3d.geometry.PointCloud(obj_type.point_cloud)
        else:
            raise ValueError('no mesh or point cloud available for object in object library')

        # apply transformation according to scene and append to list
        o3d_obj.transform(obj.pose)
        o3d_pcs.append(o3d_obj)

    return o3d_pcs


def show_scene(scene: scene.Scene, with_bg_objs=True, add_plane=False):
    """
    shows the objects of a scene

    :param scene: a core_types.Scene object
    :param with_bg_objs: whether to show background objects as well, defaults to True
    :param add_plane: whether to add a plane to the scene, defaults to False

    :return: returns when viewer is closed by user
    """
    o3d_pcs = _get_scene_geometries(scene, with_bg_objs=with_bg_objs)
    if add_plane:
        o3d_pcs.append(create_plane())

    # and visualize
    show_o3d_point_clouds(o3d_pcs)


def show_aligned_scene_point_clouds(scene: scene.Scene, views):
    """
    shows the full point cloud and overlays the partial point cloud from a view (or a list of views)

    :param scene: the scene
    :param views: instance of scene.CameraView, or list thereof

    :return: returns when the user closes the viewer
    """

    o3d_pcs = _get_scene_geometries(scene, with_bg_objs=True)

    if not type(views) is list:
        views = [views]

    for view in views:
        o3d_pcs.append(view.to_point_cloud())

    show_o3d_point_clouds(o3d_pcs)


def create_plane(l=0.3, w=0.3, h=0.001):
    ground_plane = o3d.geometry.TriangleMesh.create_box(l, w, h)
    ground_plane.compute_triangle_normals()
    ground_plane.translate(np.array([-l / 2, -w / 2, -h]))
    return ground_plane


def show_grasp_set(objects: list, gs, gripper=None, n=None, score_color_func=None, with_plane=False, use_width=False):
    """
    visualizes a given grasp set with the specified gripper.

    :param objects: list of objects to show in the scene, must be o3d geometries (mesh, point cloud, etc.)
    :param gs: the GraspSet to visualize (can also be a single Grasp)
    :param gripper: the gripper to use, if none provided just coordinate frames will be displayed
    :param n: int number of grasps from set to display, if None, all grasps will be shown
    :param score_color_func: handle to a function that maps the score to a color [0..1, 0..1, 0..1]
                             if None, some coloring scheme will be used irrespective of score
    :param with_plane: if True, a plane at z=0 will be displayed
    :param use_width: if True, will squeeze the gripper model to the width of the grasps
    """
    if type(gs) is grasp.Grasp:
        gs = gs.as_grasp_set()

    if n is not None:
        n = np.minimum(n, len(gs))
        indices = np.random.choice(len(gs), n, replace=False)
        gs = gs[indices]

    if with_plane:
        objects.append(create_plane())

    for g in gs:
        if gripper is None:
            gripper_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            tf = np.eye(4)
        else:
            gripper_vis = copy.deepcopy(gripper.mesh)
            tf = gripper.tf_base_to_TCP

        tf_squeeze = np.eye(4)
        if use_width:
            tf_squeeze[0, 0] = (g.width + 0.005) / gripper.opening_width

        gripper_vis.transform(g.pose @ tf_squeeze @ tf)

        if score_color_func is not None:
            gripper_vis.paint_uniform_color(score_color_func(g.score))

        objects.append(gripper_vis)

    colorize_point_clouds(objects)
    lookat = np.asarray([0.0, 0.0, 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    front = np.asarray([0.0, 1.0, 0.0])
    zoom = 0.9
    o3d.visualization.draw(objects)  #, lookat=lookat, up=up, front=front, zoom=zoom)


def show_grasp_set_in_scene(scene: scene.Scene, gs: grasp.GraspSet, gripper=None, n=None, score_color_func=None):
    """
    visualizes a given grasp set with the specified gripper within a scene environment.

    :param scene: a scene containing object instances
    :param gs: the GraspSet to visualize (can also be a single Grasp)
    :param gripper: the gripper to use, if none provided just coordinate frames will be displayed
    :param n: int number of grasps from set to display, if None, all grasps will be shown
    :param score_color_func: handle to a function that maps the score to a color [0..1, 0..1, 0..1]
                             if None, some coloring scheme will be used irrespective of score
    """
    scene_objects = _get_scene_geometries(scene, with_bg_objs=True)
    show_grasp_set(scene_objects, gs=gs, gripper=gripper, n=n, score_color_func=score_color_func)
