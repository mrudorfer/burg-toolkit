import numpy as np
import trimesh
import open3d as o3d

from . import grasp
from . import gripper
from . import util
from . import visualization


def sample_antipodal_grasps(point_cloud, gripper_model: gripper.ParallelJawGripper, n=10, apex_angle=30, seed=42,
                            epsilon=1e-05, visualize=False):
    """
    Sampling antipodal grasps from an object point cloud. Sampler looks for points which are opposing each other,
    i.e. have opposing surface normal orientations which are aligned with the vector connecting both points.

    :param visualize: if True, will show some pictures to visualize the process (default = False)
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

    for idx in point_indices:
        ref_point = point_cloud[idx]

        # create cone
        # translate so that point end is in origin, rotate it to point's normal, then translate to correct position
        # using trimesh here because they offer methods for checking if points are contained in a mesh
        tf_height = np.eye(4)
        tf_height[2, 3] = -height
        tf_rot = np.eye(4)
        tf_rot[0:3, 0:3] = util.rotation_to_align_vectors(np.array([0, 0, -1]), -ref_point[3:6])
        tf_trans = np.eye(4)
        tf_trans[0:3, 3] = ref_point[0:3]
        tf = np.dot(tf_trans, np.dot(tf_rot, tf_height))

        cone = trimesh.creation.cone(radius, height, transform=tf)

        # find points within the cone
        # it will probably be faster to construct a kd-tree for the point cloud and get candidates first
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(cone)
        bools = intersector.contains_points(point_cloud[:, 0:3])  # returns (n,) bool
        print('bools', bools.shape)
        print('any true?', np.any(bools))
        target_points = point_cloud[bools]
        print('target_points', target_points.shape)

        if visualize:
            cone_vis = o3d.geometry.TriangleMesh.create_cone(radius, height)
            # cone_vis.translate(np.asarray([0, 0, -height]))
            cone_vis.transform(tf)

            sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere_vis.translate(ref_point[:3])

            pc_list = [point_cloud, cone.vertices]
            if len(target_points) > 0:
                pc_list.append(target_points)

            o3d_obj_list = util.numpy_pc_to_o3d(pc_list)
            o3d_obj_list.extend([cone_vis, sphere_vis])
            visualization.show_o3d_point_clouds(o3d_obj_list)

        if len(target_points) == 0:
            continue

        # target_points may or may not include the reference point (behaviour undefined according to trimesh docs)
        # compute all the required features
        ppfs = np.empty((len(target_points), 5))

        d = (target_points[:, 0:3] - ref_point[0:3]).reshape(-1, 3)
        ppfs[:, 0] = np.linalg.norm(d, axis=-1)

        ppfs[:, 1] = util.angle(ref_point[3:6], d)
        ppfs[:, 2] = util.angle(target_points[:, 3:6], d)
        ppfs[:, 3] = util.angle(ref_point[3:6], target_points[:, 3:6])

        # check shapes
        print('point_cloud:', point_cloud.shape)
        print('target_points:', target_points.shape)
        print('ppfs:', ppfs.shape)

        print(len(ppfs[ppfs[:, 0] >= gripper_model.opening_width - epsilon]), 'points too far away')
        print(len(ppfs[ppfs[:, 0] <= epsilon]), 'points too close')

        ppfs[:, 4] = ppfs[:, 0:3].sum(axis=-1)

        print('min sum of angles:', np.amin(ppfs[:, 4]))
        print('max sum of angles:', np.amax(ppfs[:, 4]))

        n_sampled += 1  # todo: adjust this to real number of sampled
        if n_sampled >= n:
            break

    return grasp.GraspSet()

