import logging

import numpy as np
import trimesh
import open3d as o3d

from . import util


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
    above_min_prob = probs >= min_prob

    # assert sorted list of probabilities (descending)
    if np.count_nonzero(above_min_prob) < min_num:
        indices = np.nonzero(probs[:min_num])
    else:
        indices = probs[above_min_prob][:max_num]
    return indices


def compute_stable_poses(mesh, verify_in_sim=False, urdf_fn=None, min_prob=0.02, max_num=10, min_num=1):
    """
    Computes stable resting poses for this object. Uses the trimesh function stable_poses based on the provided mesh.
    Produces at least `min_num` poses and max `max_num` poses, except when these values are set to `None`.
    Returns only poses with computed probability of at least `min_prob`, unless needs to return less likely poses to
    reach `min_num`.

    :param mesh: open3d or trimesh mesh object.
    :param verify_in_sim: If True, the poses will be verified in simulation, i.e. simulation is run until object rests.
                          This option requires `urdf_fn` to be set.
    :param urdf_fn: Path to the urdf file that represents this mesh object in simulation.
    :param min_prob: Will only return poses with likelihood above this value. Set to 0 or None to disregard.
    :param max_num: Maximum number of poses to return. Set to None for no limit.
    :param min_num: Minimum number of poses to return, even if likelihood below `min_prob`. Set to 0/None to disregard.

    :return: ndarray of poses (n, 4, 4), ndarray of probabilities (n)
    """
    if verify_in_sim and urdf_fn is None:
        raise ValueError('If verify_in_sim set to True, you also need to provide urdf_fn')

    mesh = as_trimesh(mesh)
    transforms, probs = trimesh.poses.compute_stable_poses(mesh)
    used_tf_indices = _get_prob_indices(probs, min_prob, max_num, min_num)
    transforms = transforms[used_tf_indices]
    probs = probs[used_tf_indices]
    logging.debug(f'\tfound {len(probs)} stable poses with probs between {np.min(probs)} and {np.max(probs)}')

    if not verify_in_sim:
        return transforms, probs

    return NotImplementedError('verify_in_sim is not implemented yet')

    # verify poses in simulation and adjust
    for i in range(len(transforms)):
        # compute the transform required to adjust the pose of the object
        # create pybullet scene with plane and object based on urdf
        print('**********************')
        print(f'{shape} pose {i} with probability:', probs[i])

        # change object to store it in correct pose
        orig_mesh = copy.deepcopy(shape.mesh)
        shape.mesh.transform(transforms[i])

        name = shape_name + f'_pose_{i}'
        burg.io.save_mesh(os.path.join(shape_dir_transformed, name + '.obj'), shape.mesh)

        # create vhacd and store it as well
        p.connect(p.DIRECT)
        transformed_fn = os.path.join(shape_dir_transformed, name + '.obj')
        vhacd_fn = os.path.join(shape_dir_vhacd, name + '.obj')
        log_fn = os.path.join(shape_dir_vhacd, name + '_log.txt')
        p.vhacd(transformed_fn, vhacd_fn, log_fn)
        p.disconnect()

        # get vhacd mesh
        vhacd_mesh = burg.io.load_mesh(vhacd_fn)

        # create urdf file
        # we use COM of vhacd mesh
        urdf_fn = os.path.join(shape_dir_transformed, name + '.urdf')
        burg.io.save_urdf(urdf_fn, name + '.obj', name, inertia=default_inertia, com=vhacd_mesh.get_center(),
                          mass=shape.mass * mass_factor)
        shutil.copy2(urdf_fn, shape_dir_vhacd)  # associate with vhacd obj files as well

        # ** DOUBLE_CHECK SIM
        # we could be done here, however, at the start of the sim the object may move slightly, mainly because
        # we comute resting poses for complete meshes and use vhacd meshes for simulation
        # therefore we use found pose to initialise a simulation with vhacd and retrieve final resting pose
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        oid = p.loadURDF(os.path.join(shape_dir_vhacd, name + '.urdf'))
        pos1, quat1 = p.getBasePositionAndOrientation(oid)
        print('*********************')
        print(f'shape center {shape.mesh.get_center()}')
        print(f'vhacd center {vhacd_mesh.get_center()}')
        print(f'pos {pos1}, quat {quat1}, before simulation')
        dt = 1 / 240
        seconds = 5
        for _ in range(int(seconds / dt)):
            p.stepSimulation()
            # time.sleep(2*dt)
        pos2, quat2 = p.getBasePositionAndOrientation(oid)
        p.disconnect()

        # ** APPLY POSE ADJUSTMENTS FOUND IN SIM

        print(f'pos {pos2}, quat {quat2}, after simulation')
        print('*********************')
        diff_pos = np.asarray(pos2) - np.asarray(pos1)

        print('min bounds before adjustments')
        print('shape', shape.mesh.get_min_bound())
        print('vhacd', vhacd_mesh.get_min_bound())

        # apply orientation
        rot_mat = burg.util.tf_from_pos_quat(quat=quat2, convention='pybullet')[:3, :3]
        rot_center = vhacd_mesh.get_center()
        shape.mesh.rotate(rot_mat, center=rot_center)
        vhacd_mesh.rotate(rot_mat, center=rot_center)

        # apply translation
        shape.mesh.translate(diff_pos, relative=True)
        vhacd_mesh.translate(diff_pos, relative=True)

        print('min bounds after adjustments')
        print('shape', shape.mesh.get_min_bound())
        print('vhacd', vhacd_mesh.get_min_bound())

        # finally put center of mass onto the z-axis
        target_pos = np.zeros(3)
        target_pos[2] = shape.mesh.get_center()[2]
        shape.mesh.translate(target_pos, relative=False)
        target_pos[2] = vhacd_mesh.get_center()[2]
        vhacd_mesh.translate(target_pos, relative=False)

        # ** STORE FINAL FILES
        # since the mesh got transformed, we save it again (and recreate urdf files)
        burg.io.save_mesh(os.path.join(shape_dir_transformed, name + '.obj'), shape.mesh)
        burg.io.save_mesh(vhacd_fn, vhacd_mesh)
        burg.io.save_urdf(urdf_fn, name + '.obj', name, inertia=default_inertia, com=vhacd_mesh.get_center(),
                          mass=shape.mass * mass_factor)
        shutil.copy2(urdf_fn, shape_dir_vhacd)  # associate with vhacd obj files as well

        # finally add to list of shapes
        with open(shapes_fn, 'a') as f:
            f.write(name + '\n')

        # revert changes to get clean state for next pose
        shape.mesh = orig_mesh

