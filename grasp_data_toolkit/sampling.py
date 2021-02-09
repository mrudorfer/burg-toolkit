import numpy as np
import trimesh
import open3d as o3d

from . import grasp
from . import gripper
from . import visualization


def _rotation_to_align_vectors(vec_a, vec_b):
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
        u = _generate_random_unit_vector()
        k = np.cross(vec_a, u).reshape(3, 1) * 2

    kt = k.reshape(1, 3)
    rot_mat = 2 * np.dot(k, kt) / np.dot(kt, k) - np.eye(3)
    return rot_mat


def _generate_random_unit_vector():
    """
    Generates a random unit vector.

    :return: 3-elem np array with magnitude 1
    """
    vec = np.random.normal(size=3)
    mag = np.linalg.norm(vec)
    if mag <= 1e-5:
        return _generate_random_unit_vector()
    else:
        return vec / mag


def _angle(vec_a, vec_b, as_degree=True):
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


def sample_antipodal_grasps(point_cloud, gripper_model: gripper.ParallelJawGripper, n=10, apex_angle=30, seed=42,
                            epsilon=1e-05):
    """
    Sampling antipodal grasps from an object point cloud. Sampler looks for points which are opposing each other,
    i.e. have opposing surface normal orientations which are aligned with the vector connecting both points.

    :param epsilon: only grasps are considered, whose points are between (epsilon, opening_width - epsilon) apart [m].
    :param apex_angle: opening angle of the cone in degree (full angle from side to side, not just to central axis)
    :param n: number of grasps to sample, set to `np.Inf` if you want all grasps
    :param gripper_model: the gripper model as instance of `gripper.ParallelJawGripper`
    :param point_cloud: (N, 6) numpy array with points and surface normals
    :param seed: Takes a seed to initialise the random number generator before sampling

    :return: a GraspSet
    """
    n_sampled = 0

    # determine the cone parameters
    height = gripper_model.opening_width
    radius = height * np.sin(np.deg2rad(apex_angle/2.0))

    # randomize the points so we avoid sampling only a specific part of the object (if point cloud is sorted)
    point_indices = np.arange(len(point_cloud))
    r = np.random.RandomState(seed)
    r.shuffle(point_indices)

    # construct the cone template - its pointy end will be in the origin after the transform
    trans_height = np.eye(4)
    trans_height[2, 3] = -height
    cone_template = trimesh.creation.cone(radius, height, transform=trans_height)

    for idx in point_indices:
        ref_point = point_cloud[idx]

        # create cone, align it to point's normal, then translate to correct position
        tf_rot = np.eye(4)
        tf_rot[0:3, 0:3] = _rotation_to_align_vectors(np.array([0, 0, -1]), -ref_point[3:6])
        tf_trans = np.eye(4)
        tf_trans[0:3, 3] = ref_point[0:3]

        cone = cone_template.copy()
        cone.apply_transform(np.dot(tf_trans, tf_rot))

        # find points within the cone
        # it will probably be faster to construct a kd-tree for the point cloud and get candidates first
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(cone)
        bools = intersector.contains_points(point_cloud[:, 0:3])  # returns (n,) bool
        target_points = point_cloud[bools]

        print(len(target_points), 'target points found')

        cone_vis = o3d.geometry.TriangleMesh.create_cone(radius, height)
        cone_vis.translate(np.asarray([0, 0, -height]))
        cone_vis.transform(np.dot(tf_trans, tf_rot))

        sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere_vis.translate(ref_point[:3])

        if len(target_points) == 0:
            pc = visualization._numpy_pc_to_o3d(point_cloud)
            visualization.show_o3d_point_clouds([pc, cone_vis, sphere_vis])
            continue
        else:
            [pc, tp] = visualization._numpy_pc_to_o3d([point_cloud, target_points])
            visualization.show_o3d_point_clouds([pc, tp, cone_vis, sphere_vis])

        # target_points may or may not include the reference point (behaviour undefined according to trimesh docs)
        # compute all the required features
        d = (target_points[:, 0:3] - ref_point[0:3]).reshape(-1, 3)
        distance = np.linalg.norm(d, axis=-1)

        ang_n_ref_d = _angle(ref_point[3:6], d)
        ang_n_tar_d = _angle(target_points[:, 3:6], d)
        ang_n_ref_n_tar = _angle(ref_point[3:6], target_points[:, 3:6])

        # check shapes
        print('point_cloud', point_cloud.shape)
        print('target_points', target_points.shape)
        # print('distance', distance)
        print('ang_n_ref_d', ang_n_ref_d.shape)
        print('ang_n_tar_d', ang_n_tar_d.shape)
        print('ang_n_ref_n_tar', ang_n_ref_n_tar.shape)

        # if epsilon < distance < gripper_model.opening_width - epsilon:

        n_sampled += 1  # todo: adjust this to real number of sampled
        if n_sampled >= n:
            break

    return grasp.GraspSet()

