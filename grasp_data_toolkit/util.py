import numpy as np
import open3d as o3d


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


def angle(vec_a, vec_b, as_degree=True):
    """
    Computes the angle(s) between vec_a and vec_b.
    We use atan(|a x b|/(a*b)) which is said to be more numerically stable for small angles (according to
    Birdal, Ilic 2015).
    The vectors don't need to be normalised.
    Vectors can be broadcasted to match dimensions.

    :param vec_a: (3) or (n, 3) np array
    :param vec_b: (3) or (n, 3) np array
    :param as_degree: boolean indicator whether to provide result as degree (else radian, default True)

    :return: (1) or (n, 1) np array  with the angle between the corresponding vectors (with same indices).
             The values will be in the range [-pi/2, pi/2] or [-90, 90].
    """
    angles = np.arctan(np.linalg.norm(np.cross(vec_a, vec_b), axis=-1) / np.sum(vec_a*vec_b, axis=-1))
    if as_degree:
        return np.rad2deg(angles)
    return angles


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
