import logging

import numpy as np
import trimesh
import open3d as o3d
from open3d import visualization

from . import core
from . import scene_sim


def check_properties(mesh):
    """
    Utility function to check properties of a mesh. Will be printed to standard output.

    :param mesh: open3d.geometry.TriangleMesh
    """
    has_triangle_normals = mesh.has_triangle_normals()
    has_vertex_normals = mesh.has_vertex_normals()
    has_texture = mesh.has_textures()
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()
    _trimesh = as_trimesh(mesh)
    convex = trimesh.convex.is_convex(_trimesh)

    print(f"  no vertices:            {len(mesh.vertices)}")
    print(f"  no triangles:           {len(mesh.triangles)}")
    print(f"  dims (x, y, z):         {dimensions(mesh)}")
    print(f"  has triangle normals:   {has_triangle_normals}")
    print(f"  has vertex normals:     {has_vertex_normals}")
    print(f"  has textures:           {has_texture}")
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")
    print(f"  convex:                 {convex}")
    print(f"  components:             {_trimesh.body_count}")


def dimensions(mesh):
    """
    Returns the extent of the mesh in [x, y, z] directions, i.e. of the axis aligned bbox.

    :param mesh: open3d.geometry.TriangleMesh

    :return: (3) np array
    """
    return mesh.get_max_bound().flatten() - mesh.get_min_bound().flatten()


def poisson_disk_sampling(mesh, radius=0.003, n_points=None, with_normals=True, init_factor=5):
    """
    Performs poisson disk sampling.
    Per default it uses the radius, but if `n_points` is given it just samples as many points.
    Ideally, we fit circles with a certain radius on the surface area of the mesh. The center points
    will then form the point cloud. In practice, we just randomly sample a certain number of points (depending
    on the surface area of the mesh, the radius and init_factor), then we eliminate points which do not fit well.

    :param mesh: open3d.geometry.TriangleMesh
    :param radius: the smaller the radius, the higher the point density will be
    :param init_factor: we will initially sample `init_factor` times more points than will be needed, if better
                        accuracy is desired this can be increased, if performance is important this should be decreased
    :param with_normals: whether or not the points shall have normals (default is True)
    :param n_points: int, if given, it will simply sample as many points (which are approx evenly distributed)

    :return: open3d.geometry.PointCloud object
    """
    if n_points is None:
        # we assume the surface area to be square and use a square packing of circles
        # the number n_s of circles along the side-length s can then be estimated with
        s = np.sqrt(mesh.get_surface_area())
        n_s = (s + 2*radius) / (2*radius)
        n_points = int(n_s**2)
        print(f'going for {n_points} points')

    pc = mesh.sample_points_poisson_disk(
        number_of_points=n_points,
        init_factor=init_factor,
        use_triangle_normal=with_normals
    )

    return pc


def as_trimesh(mesh):
    """
    Makes sure the mesh is represented as trimesh. Raises an error if unrecognised mesh type.

    :param mesh: Can be open3d.geometry.TriangleMesh or trimesh.Trimesh

    :return: trimesh.Trimesh
    """
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        # todo: what happens if mesh does not have normals?
        return trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals),
                               triangle_normals=np.asarray(mesh.triangle_normals))
    raise TypeError(f'Given mesh must be trimesh or o3d mesh. Got {type(mesh)} instead.')


def compute_mesh_inertia(mesh, mass):
    """
    Computes the inertia of the given mesh, but may not work if mesh is not watertight.

    :param mesh: open3d.geometry.TriangleMesh or trimesh.Trimesh
    :param mass: mass of object in kg

    :return: moment of inertia matrix, (3, 3) float ndarray, origin center of mass (3) float ndarray
    """
    mesh = as_trimesh(mesh)
    if not mesh.is_watertight:
        logging.warning('Computing Inertia and COM despite mesh not being watertight. Be careful.')

    # trimesh meshes are density-based, so let's set density based on given mass and mesh volume
    mesh.density = mass / mesh.volume
    return mesh.moment_inertia, mesh.center_mass


def center_of_mass(mesh):
    """
    Computes the center of mass of a given mesh. May not work if mesh is not watertight.

    :param mesh: open3d.geometry.TriangleMesh or trimesh.Trimesh

    :return: (3) float ndarray
    """
    mesh = as_trimesh(mesh)
    if not mesh.is_watertight:
        logging.warning('Computing center of mass despite mesh not being watertight. Be careful.')
    return mesh.center_mass


def centroid(mesh):
    """
    Computes an approximate center of the mesh. This is not the center of mass. Meshes do not need to be watertight.

    :param mesh: open3d.geometry.TriangleMesh or Trimesh

    :return: ndarray(3) with coordinates of a centroid
    """
    mesh = as_trimesh(mesh)
    return mesh.centroid


def _get_prob_indices(probs, min_prob, max_num, min_num):
    min_prob = min_prob or 0  # catch None
    min_num = min_num or 0
    if len(probs) < min_num:
        logging.warning(f'Requested to give at least {min_num} probs from an array of {len(probs)} probs. ' +
                        f'Could not meet the condition, returning only {len(probs)} elements.')
        return np.arange(len(probs))

    above_min_prob = probs >= min_prob
    n_above = np.count_nonzero(above_min_prob)
    n_choose = min(max(n_above, min_num), max_num)
    indices = np.argpartition(probs, -n_choose)[-n_choose:]
    return indices


def compute_stable_poses(object_type, verify_in_sim=False, min_prob=0.02, max_num=10, min_num=1):
    """
    Computes stable resting poses for this object. Uses the trimesh function stable_poses based on the object's mesh.
    Produces at least `min_num` poses and max `max_num` poses, except when these values are set to `None`.
    Returns only poses with computed probability of at least `min_prob`, unless needs to return less likely poses to
    reach `min_num`.
    If `verify_in_sim` is set, the poses will also be simulated in pybullet until a rest pose is found. Note that to
    use this feature, the object_type needs to have a urdf_fn.

    :param object_type: core.ObjectType
    :param verify_in_sim: If True, the poses will be verified in simulation, i.e. simulation is run until object rests.
                          This option requires `urdf_fn` of the given `object_type` to be set.
    :param min_prob: Will only return poses with likelihood above this value. Set to 0 or None to disregard.
    :param max_num: Maximum number of poses to return. Set to None for no limit.
    :param min_num: Minimum number of poses to return, even if likelihood below `min_prob`. Set to 0/None to disregard.

    :return: ndarray of poses (n, 4, 4), ndarray of probabilities (n)
    """
    if verify_in_sim and object_type.urdf_fn is None:
        raise ValueError('If verify_in_sim set to True, you also need to provide urdf_fn')

    mesh = as_trimesh(object_type.mesh)
    transforms, probs = trimesh.poses.compute_stable_poses(mesh)
    used_tf_indices = _get_prob_indices(probs, min_prob, max_num, min_num)
    print('probs and indices')
    print(probs)
    print(used_tf_indices)
    transforms = transforms[used_tf_indices]
    probs = probs[used_tf_indices]
    logging.debug(f'\tfound {len(probs)} stable poses with probs between {np.min(probs)} and {np.max(probs)}')

    if verify_in_sim:
        simulator = scene_sim.SceneSimulator()
        for i in range(len(transforms)):
            instance = core.ObjectInstance(object_type, pose=transforms[i])
            simulator.simulate_object_instance(instance)  # pose of the instance is automatically updated
            transforms[i] = instance.pose

    # roughly center object at xy=0, because trimesh and simulation put it *somewhere* on a plane
    for i in range(len(transforms)):
        instance = core.ObjectInstance(object_type, pose=transforms[i])
        c = centroid(instance.get_mesh())
        instance.pose[0, 3] = instance.pose[0, 3] - c[0]
        instance.pose[1, 3] = instance.pose[1, 3] - c[1]

    # check for near-duplicate poses?
    # could just use some pose-based distance measure to filter, potentially adding up probabilities

    return transforms, probs

