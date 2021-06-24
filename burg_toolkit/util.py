import numpy as np
import open3d as o3d
import trimesh
import quaternion


def rotation_to_align_vectors(vec_a, vec_b):
    """
    Computes the rotation matrix to re-orient vec_a to vec_b.
    The vectors need not be unit vectors.

    :param vec_a: 3d vector
    :param vec_b: 3d vector

    :return: (3, 3) rotation matrix
    """
    # find the vector k = (a+b) and rotate 180° around it, with formula according to
    # https://math.stackexchange.com/a/2672702
    k = (vec_a / np.linalg.norm(vec_a) + vec_b / np.linalg.norm(vec_b)).reshape(3, 1)

    while np.linalg.norm(k) == 0:
        # this means the two vectors are directly opposing each other, as the sum of the vectors is zero
        # we need to find some arbitrary vector orthogonal to vec_a to rotate around it
        u = generate_random_unit_vector()
        k = np.cross(vec_a, u).reshape(3, 1) * 2

    kt = k.reshape(1, 3)
    rot_mat = 2 * np.dot(k, kt) / np.dot(kt, k) - np.eye(3)
    return rot_mat


def generate_random_unit_vector():
    """
    Generates a random unit vector.

    :return: 3-elem np array with magnitude 1
    """
    vec = np.random.normal(size=3)
    mag = np.linalg.norm(vec)
    if mag <= 1e-5:
        return generate_random_unit_vector()
    else:
        return vec / mag


def angle(vec_a, vec_b, sign_array=None, as_degree=True):
    """
    Computes the angle(s) between corresponding vectors in vec_a and vec_b.
    We use atan(|a x b|/(a*b)) which is said to be more numerically stable for small angles (according to
    Birdal, Ilic 2015).
    The vectors don't need to be normalised.
    Vectors can be broadcasted to match dimensions (e.g. if vec_a contains one vector and vec_b contains n vectors).
    Numpy might issue a runtime warning (division by zero), but resulting angles are ok.

    :param sign_array: A numpy array of same shape as output, it will contain the sign of the dot product and hence
                       indicate the direction of the angle. May be useful to differentiate between -0 and 0.
    :param vec_a: (3) or (n, 3) np array
    :param vec_b: (3) or (n, 3) np array
    :param as_degree: boolean indicator whether to provide result as degree (else radian, default True)

    :return: (1) or (n, 1) np array  with the angle between the corresponding vectors (with same indices).
             The values will be in the range [-pi/2, pi/2] or [-90, 90].
    """
    dotp = np.sum(vec_a * vec_b, axis=-1)
    if sign_array is not None:
        sign_array[:] = np.sign(dotp)
    angles = np.arctan(np.linalg.norm(np.cross(vec_a, vec_b), axis=-1) / dotp)
    if np.isnan(angles).any():
        print('warning: encountered nan value, printing corresponding vectors:')
        index = np.argwhere(np.isnan(angles))
        if len(vec_a) == len(angles):
            print('vec_a:', vec_a[index])
        else:
            print('vec_a:', vec_a)
        if len(vec_b) == len(angles):
            print('vec_b:', vec_b[index])
        else:
            print('vec_b:', vec_b)

    if as_degree:
        return np.rad2deg(angles)
    return angles


def look_at(position, target=None, up=None, flip=False):
    """
    Computes the 4x4 matrix where z-axis will be oriented towards the target, and we try to make x-axis orthogonal to
    up vector to get a reproducible in-plane rotation (although this might not always be possible, in these cases we
    first try to use `up[[2, 0, 1]]` to get a repeatable rotation and if that fails as well then we use random
    in plane rotations).

    :param position: (3,) or (n, 3) ndarray, position from where we look
    :param target: (3,) ndarray, position where to look at, if None `[0, 0, 0]` will be used
    :param up: (3,) ndarray, upward vector for reproducible in-plane rotations, if None `[0, 0, 1]` is used
    :param flip: bool, if set to True the z-axes will be flipped, i.e. they will actually point away from the target.
                 This is useful e.g. for getting OpenGL camera coordinates (as used by pyrender).

    :return: (4, 4) or (n, 4, 4) ndarray, transformation matrix
    """
    if isinstance(position, list):
        position = np.array(position)
    position = position.reshape(-1, 3)

    if target is None:
        target = np.array([0, 0, 0])
    elif isinstance(target, list):
        target = np.array(target)

    if up is None:
        up = np.array([0, 0, 1])
    elif isinstance(up, list):
        up = np.array(up)

    z_vec = position - target if flip else target - position
    z_vec = z_vec / np.linalg.norm(z_vec, axis=-1)[:, np.newaxis]

    # handle problems if z_axis and up axis are not linearly independent
    # then try some other canonical in-plane rotation
    # if that fails as well just use random
    x_vec = np.cross(up, z_vec)
    faults = np.linalg.norm(x_vec, axis=-1) < 1e-3
    if np.any(faults):
        x_vec[faults] = np.cross(up[[2, 0, 1]], z_vec[faults])
        faults = np.linalg.norm(x_vec[faults], axis=-1) < 1e-3
        while np.any(faults):
            x_vec[faults] = np.cross(generate_random_unit_vector(), z_vec[faults])
            faults = np.linalg.norm(x_vec[faults], axis=-1) < 1e-3
    x_vec = x_vec / np.linalg.norm(x_vec, axis=-1)[:, np.newaxis]

    y_vec = np.cross(z_vec, x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec, axis=-1)[:, np.newaxis]

    return tf_from_xyz_pos(x_vec, y_vec, z_vec, position)


def numpy_pc_to_o3d(point_clouds):
    """
    converts a point cloud or list of point clouds from numpy arrays of Nx3 or Nx6 to o3d point clouds

    :param point_clouds: single point cloud or list of numpy point clouds Nx3 or Nx6 (points, normals)

    :return: list of o3d point clouds
    """

    single = False
    if not type(point_clouds) is list:
        point_clouds = [point_clouds]
        single = True

    pc_objs = []
    for pc in point_clouds:
        cloud = o3d.geometry.PointCloud()

        # check if point cloud has normals
        if len(pc.shape) != 2:
            print('ERROR: input point cloud has strange number of dimensions:', pc.shape)
            return

        if pc.shape[1] == 3:
            cloud.points = o3d.utility.Vector3dVector(pc)
        elif pc.shape[1] == 6:
            cloud.points = o3d.utility.Vector3dVector(pc[:, 0:3])
            cloud.normals = o3d.utility.Vector3dVector(pc[:, 3:6])
        else:
            print('ERROR: input point cloud has strange shape:', pc.shape)
            return
        pc_objs.append(cloud)

    if single:
        return pc_objs[0]
    else:
        return pc_objs


def o3d_pc_to_numpy(point_clouds):
    """
    converts a point cloud or list of point clouds from o3d point clouds to numpy arrays of Nx3 or Nx6

    :param point_clouds: single point cloud or list of o3d point clouds which may or may not have normals

    :return: list of numpy point clouds with either Nx3 or Nx6 (depends on whether original pc had normals)
    """

    single = False
    if not type(point_clouds) is list:
        point_clouds = [point_clouds]
        single = True

    pc_objs = []
    for pc in point_clouds:
        n = len(pc.points)
        m = 3 + pc.has_normals() * 3
        np_pc = np.empty((n, m))

        np_pc[:, 0:3] = np.asarray(pc.points)
        if pc.has_normals():
            np_pc[:, 3:6] = np.asarray(pc.normals)

        pc_objs.append(np_pc)

    if single:
        return pc_objs[0]
    else:
        return pc_objs


def inspect_ndarray(name, my_array):
    """
    Function used for debugging.
    """
    print(f'array {name}:')
    print(f'* type? {type(my_array)}')
    print(f'* internal type: {my_array.dtype}')
    print(f'* c_contiguous? {my_array.flags["C_CONTIGUOUS"]}')
    print(f'* f_contiguous? {my_array.flags["F_CONTIGUOUS"]}')
    print(f'* behaved? {my_array.flags["BEHAVED"]}')
    print(f'* owns data? {my_array.flags["OWNDATA"]}')
    print(f'* shape: {my_array.shape}')
    print(f'* min: {np.min(my_array)}, max: {np.max(my_array)}, avg: {np.mean(my_array)}')


def o3d_image_from_numpy(np_array, dtype=np.float32, replace_inf_by=None):
    """
    Converts a given numpy array into an open 3d image.
    Is mainly used to ensure a c-contiguous array and correct dtype.
    Should be the case with default ndarrays, but e.g. mat73 package gives fortran-style arrays which need to be
    converted.

    :param np_array: (h, w) or (h, w, c) ndarray
    :param dtype: desired type, defaults to np.float32, can also be uint8 or uint16
    :param replace_inf_by: value that will replace np.inf values in the numpy image. Defaults to None, in which case no
                           replacement will be done.

    :return: open3d.geometry.Image
    """
    c_type_array = np.ascontiguousarray(np_array).astype(dtype)
    if replace_inf_by is not None:
        c_type_array[c_type_array == np.inf] = replace_inf_by
    return o3d.geometry.Image(c_type_array)


def merge_o3d_triangle_meshes(meshes):
    """
    Merges vertices and triangles from different meshes into one mesh.

    :param meshes: a list of meshes (o3d.geometry.TriangleMesh)

    :return: a merged o3d.geometry.TriangleMesh
    """
    vertices = np.empty(shape=(0, 3), dtype=np.float64)
    triangles = np.empty(shape=(0, 3), dtype=np.int)
    for mesh in meshes:
        v = np.asarray(mesh.vertices)  # float list (n, 3)
        t = np.asarray(mesh.triangles)  # int list (n, 3)
        t += len(vertices)  # triangles reference the vertex index
        vertices = np.concatenate([vertices, v])
        triangles = np.concatenate([triangles, t])

    # finally create the merged mesh
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles))

    # some refinement
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    return mesh


def mesh_contains_points(mesh, points):
    """
    Computes the points contained within the specified mesh using trimesh's ray/triangle methods.
    Note that it gives undefined behaviour if a point lies on a triangle.

    :param mesh: A mesh, either as open3d.geometry.TriangleMesh or trimesh.Trimesh
    :param points: A numpy array with shape of (n, k>=3), of which first three columns are used as (x, y, z).

    :return: A numpy array of contained points (m, k).
    """
    _mesh = []
    if type(mesh) is o3d.geometry.TriangleMesh:
        # convert to trimesh
        _mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    elif type(mesh) is trimesh.Trimesh:
        _mesh = mesh
    else:
        raise ValueError('Unexpected type of mesh.')

    # this is reasonably quick, as it first checks the bounding box to narrow down the number of points
    # however, could still take some time if done repeatedly for large meshes and point clouds
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(_mesh)
    bools = intersector.contains_points(points[:, 0:3])  # returns (n,) bool

    return points[bools]


def tf_from_xyz_pos(x_axis, y_axis, z_axis, position=None):
    """
    Constructs a 4x4 transformation matrix for the given inputs.
    Can get arrays of axes, broadcasting possible.

    :param x_axis: (3,) or (n, 3) array with x axes
    :param y_axis: (3,) or (n, 3) array with y axes
    :param z_axis: (3,) or (n, 3) array with z axes
    :param position: (3,) or (n, 3) array with translations (will be zero if None given, which is default)

    :return: (n, 4, 4) homogenous transformation matrices, or (4, 4) if n==1
    """
    x_axis = x_axis.reshape(-1, 3)
    y_axis = y_axis.reshape(-1, 3)
    z_axis = z_axis.reshape(-1, 3)
    if position is None:
        position = np.zeros(3)
    position = position.reshape(-1, 3)

    n = np.max([x_axis.shape[0], y_axis.shape[0], z_axis.shape[0], position.shape[0]])
    if not (x_axis.shape[0] == 1 or x_axis.shape[0] == n) \
            or not (y_axis.shape[0] == 1 or y_axis.shape[0] == n) \
            or not (z_axis.shape[0] == 1 or z_axis.shape[0] == n) \
            or not (position.shape[0] == 1 or position.shape[0] == n):
        raise ValueError('util.tf_from_xyz_pos got unexpected shapes and is unable to broadcast')

    tf = np.zeros((n, 4, 4))
    tf[:, 3, 3] = 1
    tf[:, :3, 0] = x_axis
    tf[:, :3, 1] = y_axis
    tf[:, :3, 2] = z_axis
    tf[:, :3, 3] = position

    if n == 1:
        return np.reshape(tf, (4, 4))
    return tf


def position_and_quaternion_from_tf(tf, convention='wxyz'):
    """
    Function to compute position and rotation from a 4x4 transformation matrix.

    :param tf: (4, 4) homogenous transformation matrix
    :param convention: can be 'wxyz' or 'xyzw' and describes the convention for returned quaternion.
                       E.g. numpy-quaternion package uses 'wxyz' and py_bullet uses 'xyzw'. Defaults to 'wxyz'.
                       For convenience, we can also use 'pybullet' or 'numpy-quaternion' as strings.

    :return: (position(3), quaternion(4))
    """
    # for now we only need it for one transform, but i guess at some point we have to extend it to array of tfs as well
    position = tf[0:3, 3]
    q = quaternion.from_rotation_matrix(tf[0:3, 0:3])
    if convention in ('xyzw', 'pybullet'):
        return position, [q.x, q.y, q.z, q.w]
    if convention in ('wxyz', 'numpy-quaternion'):
        return position, [q.w, q.x, q.y, q.z]
    raise ValueError(f'unknown convention {convention}, needs to be one of xyzw / pybullet / wxyz / numpy-quaternion.')


def o3d_mesh_to_trimesh(o3d_mesh):
    """
    Create a trimesh object from open3d.geometry.TriangleMesh.

    :param o3d_mesh: open3d.geometry.TriangleMesh

    :return: trimesh object
    """
    # todo: what happens if mesh does not have normals?
    t_mesh = trimesh.Trimesh(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles),
                             vertex_normals=np.asarray(o3d_mesh.vertex_normals),
                             triangle_normals=np.asarray(o3d_mesh.triangle_normals))
    return t_mesh
