import numpy as np


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

    print(f"  has triangle normals:   {has_triangle_normals}")
    print(f"  has vertex normals:     {has_vertex_normals}")
    print(f"  has textures:           {has_texture}")
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")


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


def collision(mesh, point_cloud) -> bool:
    """
    Checks if any points of given point_cloud are inside the mesh.

    :param mesh:
    :param point_cloud:

    :return: True if there is at least one point in collision with the mesh
    """
    print('WARNING: collision() not implemented yet. Defaults to False.')
    return False
