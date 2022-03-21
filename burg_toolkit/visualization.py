import copy

import open3d as o3d
import numpy as np
import trimesh
from matplotlib import pyplot as plt

from . import core
from . import util
from . import grasp


def _convert_geometries_to_o3d_objects(geometry_list):
    """
    Receives a list of geometries and displays them. Geometries can be of various types:
    o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, numpy-array (point cloud), trimesh.Trimesh, also the data
    types from burg toolkit are supported: core.ObjectType, core.ObjectInstance, core.Scene.
    For core.Scene objects, the `get_mesh_list` will be called with default arguments. If you want to have more control
    over this, just provide the mesh list directly instead of the scene.

    :param geometry_list: geometries to be displayed, of various types.

    :return: list of geometries that can be visualised with open3d
    """
    o3d_objs = []
    for geometry in geometry_list:
        if isinstance(geometry, o3d.geometry.TriangleMesh) or isinstance(geometry, o3d.geometry.PointCloud):
            o3d_objs.append(geometry)
        elif isinstance(geometry, trimesh.Trimesh):
            o3d_objs.append(geometry.as_open3d)
        elif isinstance(geometry, np.ndarray):
            o3d_objs.append(util.numpy_pc_to_o3d(geometry))
        elif isinstance(geometry, core.ObjectType):
            o3d_objs.append(geometry.mesh)
        elif isinstance(geometry, core.ObjectInstance):
            o3d_objs.append(geometry.get_mesh())
        elif isinstance(geometry, core.Scene):
            o3d_objs.extend(geometry.get_mesh_list())
        else:
            raise TypeError(f'geometry in list is of unsupported type {type(geometry)}')

    return o3d_objs


def show_geometries(geometry_list, colorize=True):
    """
    Receives a list of geometries and displays them. Geometries can be of various types:
    o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, numpy-array (point cloud), trimesh.Trimesh, also the data
    types from burg toolkit are supported: core.ObjectType, core.ObjectInstance, core.Scene.
    For core.Scene objects, the `get_mesh_list` will be called with default arguments. If you want to have more control
    over this, just provide the mesh list directly instead of the scene.

    :param geometry_list: geometries to be displayed, of various types.
    :param colorize: whether or not to paint objects in some distinguished color.
    """
    o3d_objs = _convert_geometries_to_o3d_objects(geometry_list)

    if colorize:
        _colorize_o3d_objects(o3d_objs)
    o3d.visualization.draw(o3d_objs)


def _colorize_o3d_objects(o3d_objects, colormap_name='tab20'):
    """
    gets a list of o3d point clouds and adds unique colors to them (in-place)

    :param o3d_objects: list of o3d point clouds
    :param colormap_name: name of the matplotlib colormap to use, defaults to 'tab20'

    :return: the same list of o3d point clouds (they are adjusted in-place, so this return value should be rarely used)
    """

    # this colormap offers 20 different qualitative colors
    colormap = plt.get_cmap(colormap_name)
    color_idx = 0

    for o3d_pc in o3d_objects:
        if type(o3d_pc) is o3d.geometry.TriangleMesh:
            if o3d_pc.has_vertex_colors():
                continue
        if type(o3d_pc) is o3d.geometry.PointCloud:
            if o3d_pc.has_colors():
                continue
        color = np.asarray(colormap(color_idx)[0:3])
        o3d_pc.paint_uniform_color(color)
        color_idx = (color_idx + 1) % colormap.N

    return o3d_objects


def show_grasp_set(objects: list, gs, gripper=None, n=None, score_color_func=None, with_plane=False, use_width=False):
    """
    visualizes a given grasp set with the specified gripper.

    :param objects: list of objects to show in the scene, see `show_geometries()` for type info
    :param gs: the GraspSet to visualize (can also be a single Grasp)
    :param gripper: the GripperVisualisation to use, if none provided just coordinate frames will be displayed
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
            gripper_vis = create_frame()
        else:
            gripper_vis = copy.deepcopy(gripper.mesh)

        tf_squeeze = np.eye(4)
        if use_width:
            assert gripper is not None, 'gripper must be given for visualisation with width'
            tf_squeeze[0, 0] = (g.width + 0.005) / gripper.opening_width

        gripper_vis.transform(g.pose @ tf_squeeze)

        if score_color_func is not None:
            gripper_vis.paint_uniform_color(score_color_func(g.score))

        objects.append(gripper_vis)

    show_geometries(objects)


def create_plane(size=(0.5, 0.5), centered=True, h=0.001):
    """
    Create a plane for visualisation purposes.

    :param size: (x, y) tuple with width and length in x/y directions
    :param centered: If True, the plane will be centered around origin. Otherwise in positive xy with corner at origin.
    :param h: The thickness of the visual plane.

    :return: o3d.geometry.TriangleMesh with the plane
    """
    x, y = size
    ground_plane = o3d.geometry.TriangleMesh.create_box(x, y, h)
    ground_plane.compute_triangle_normals()
    ground_plane.translate(np.array([0, 0, -h]))
    if centered:
        ground_plane.translate(np.array([-x / 2, -y / 2, 0]))
    return ground_plane


def create_frame(size=0.01, pose=None):
    """
    Create a coordinate system for visualisation purposes.

    :param size: float, side length of the axes
    :param pose: (4, 4) ndarray, pose of the frame (if None, placed at origin)

    :return: o3d.geometry.TriangleMesh with the frame
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if pose is not None:
        frame.transform(pose)
    return frame
