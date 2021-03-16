import numpy as np
import trimesh
import open3d as o3d
from timeit import default_timer as timer

from . import grasp
from . import gripper
from . import util
from . import visualization


def sample_antipodal_grasps(point_cloud, gripper_model: gripper.ParallelJawGripper, n=10, apex_angle=30, seed=42,
                            epsilon=0.002, max_sum_of_angles=15, circle_discretisation_steps=12, visualize=False):
    """
    Sampling antipodal grasps from an object point cloud. Sampler looks for points which are opposing each other,
    i.e. have opposing surface normal orientations which are aligned with the vector connecting both points.

    :param circle_discretisation_steps: When a candidate point pair is found, grasps will be created on a circle
                                        in the middle between both points, around the connecting vector as circle
                                        centre. This parameter defines the number of grasps which will be equidistantly
                                        distributed on the circle's circumference.
    :param max_sum_of_angles: point pairs will be discarded whose sum of absolute PPF-angles is larger than this.
                              this parameter effectively puts a limit to the apex_angle, we might as well merge those
                              two to one parameter. there is only a difference when max_sum_of_angles would be larger
                              than apex_angle, but I am not sure if this would be a reasonable setting
    :param visualize: if True, will show some pictures to visualize the process (default = False)
    :param epsilon: only grasps are considered, whose points are between (epsilon, opening_width - epsilon) apart [m].
    :param apex_angle: opening angle of the cone in degree (full angle from side to side, not just to central axis)
    :param n: number of grasps to sample, set to `np.Inf` if you want all grasps
    :param gripper_model: the gripper model as instance of `gripper.ParallelJawGripper`
    :param point_cloud: (N, 6) numpy array with points and surface normals
    :param seed: Takes a seed to initialise the random number generator before sampling

    :return: a GraspSet
    """
    point_cloud = util.o3d_pc_to_numpy(point_cloud)

    n_sampled = 0
    gs = grasp.GraspSet()

    # determine the cone parameters
    height = gripper_model.opening_width - epsilon
    radius = height * np.sin(np.deg2rad(apex_angle/2.0))

    # randomize the points so we avoid sampling only a specific part of the object (if point cloud is sorted)
    point_indices = np.arange(len(point_cloud))
    r = np.random.RandomState(seed)
    r.shuffle(point_indices)

    t_find_target_pts = 0
    t_find_grasp_poses = 0
    t_check_collisions = 0

    t0 = timer()

    for idx in point_indices:
        ref_point = point_cloud[idx]

        t1 = timer()
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
        target_points = util.mesh_contains_points(cone, point_cloud)
        # target_points may or may not include the reference point (behaviour undefined according to trimesh docs)
        # so let's make sure to exclude the target_point
        target_points = target_points[(target_points != ref_point).any(axis=1)]
        print('found', len(target_points), 'target points in the cone')
        t2 = timer()
        t_find_target_pts += t2 - t1

        if len(target_points) == 0:
            if visualize:
                cone_vis = o3d.geometry.TriangleMesh.create_cone(radius, height)
                cone_vis.transform(tf)

                sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere_vis.translate(ref_point[:3])

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

                pc_list = [point_cloud, cone.vertices]

                o3d_obj_list = util.numpy_pc_to_o3d(pc_list)
                o3d_obj_list.extend([cone_vis, sphere_vis, frame])
                visualization.show_o3d_point_clouds(o3d_obj_list)

            continue

        # compute all the required features
        ppfs = np.empty((len(target_points), 5))

        d = (target_points[:, 0:3] - ref_point[0:3]).reshape(-1, 3)
        ppfs[:, 0] = np.linalg.norm(d, axis=-1)

        # compute angles (n_r, d), (n_t, d), (n_r, n_t)
        # also compute sum of absolute values
        ppfs[:, 1] = util.angle(ref_point[3:6], d)
        ppfs[:, 2] = util.angle(target_points[:, 3:6], d)
        signs = np.zeros(len(ppfs))
        ppfs[:, 3] = util.angle(ref_point[3:6], target_points[:, 3:6], sign_array=signs)
        ppfs[:, 4] = np.abs(ppfs[:, 0:3]).sum(axis=-1)

        # exclude target points which do not have opposing surface orientations
        # positive sign means vectors are facing into the same direction and have to be discarded
        print('*', len(signs[signs >= 0]), 'points are not on opposing surfaces')

        # in my example, there were no points too far or too close, but this may still be reasonable
        print('*', len(ppfs[ppfs[:, 0] > gripper_model.opening_width - epsilon]), 'points too far away')
        print('*', len(ppfs[ppfs[:, 0] < epsilon]), 'points too close')

        print('*', 'min sum of angles:', np.amin(ppfs[:, 4]))
        print('*', 'num of points below', max_sum_of_angles, 'degree as sum of angles:',
              len(ppfs[ppfs[:, 4] <= max_sum_of_angles]))

        # let's do the actual filtering
        # question being: do i really need a bunch of candidate points, or can i go with the best fit?
        mask = ((ppfs[:, 4] <= max_sum_of_angles)  # surface normals and connecting vector are all aligned
                & (ppfs[:, 0] <= gripper_model.opening_width - epsilon)  # target point not too far
                & (ppfs[:, 0] >= epsilon)  # target point not too close
                & (signs < 0)  # the surfaces have opposing normals
                )

        candidate_points = target_points[mask]
        candidate_ppf = ppfs[mask]
        candidate_d = d[mask]

        print('*', len(candidate_points), 'candidate points remaining')
        if len(candidate_points) == 0:
            continue

        best_idx = np.argmin(candidate_ppf[:, 4])
        # for each candidate point (or only the best one?)
        # get centre point p_c = p_r + 1/2 * d
        p_c = ref_point[:3] + 1/2 * candidate_d[best_idx]

        # create circle around p_c, with d/|d| as normal -> actually, this should be the average of the two surface
        # normals, so we align the finger tips as best as possible with the surfaces (reorienting n_r in the process)
        # get two vectors v_1, v_2 orthogonal to d (defining the circle plane)
        # then circle is p_c + r*cos(t)*v_1 + r*sin(t)*v_2, with t in k steps from 0 to 2pi, r defined by finger-length
        circle_normal = (candidate_points[best_idx, 3:6] - ref_point[3:6]) / 2
        circle_normal = circle_normal / np.linalg.norm(circle_normal)
        v1 = np.zeros(3)
        while np.linalg.norm(v1) == 0:
            tmp_vec = util.generate_random_unit_vector()
            v1 = np.cross(circle_normal, tmp_vec)

        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(circle_normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        t = np.arange(0, 2*np.pi, 2*np.pi / circle_discretisation_steps).reshape(-1, 1)
        r = gripper_model.finger_length  # todo: this actually rather depends on where the gripper origin is
        circle_points = p_c + r*np.cos(t)*v1.reshape(1, 3) + r*np.sin(t)*v2.reshape(1, 3)

        print('**', 'circle_points', circle_points.shape)
        # circle points are actually already our grasp points, but we still need the orientation:
        #   - gripper z-axis oriented towards p_c from each circle point
        #   - gripper x-axis is the circle normal
        #   - y correspondingly, sign of direction should not matter since most grippers are symmetric
        gripper_z = p_c - circle_points
        gripper_z = gripper_z / np.linalg.norm(gripper_z, axis=-1).reshape(-1, 1)

        gripper_x = circle_normal
        gripper_y = np.cross(gripper_z, gripper_x)

        # having the axes, we can build the rotation matrices, the translations and compute the final tfs
        tf_rot = np.zeros((circle_discretisation_steps, 4, 4))
        tf_rot[:, 3, 3] = 1
        tf_rot[:, :3, 0] = gripper_x  # should broadcast (gripper_x is shape (3,))
        tf_rot[:, :3, 1] = gripper_y
        tf_rot[:, :3, 2] = gripper_z

        tf_trans = np.zeros((circle_discretisation_steps, 4, 4))
        tf_trans[:] = np.eye(4)
        tf_trans[:, :3, 3] = circle_points

        tfs = np.matmul(tf_trans, tf_rot)

        t3 = timer()
        t_find_grasp_poses += t3 - t2

        # the tfs are our grasp poses
        # we do not need empty-grasped-volume check, since this should be satisfied as per definition of the grasps

        # let's now do the collision checks
        print('**', 'checking collisions...')
        valid_grasps = np.empty(shape=(len(tfs)), dtype=np.bool)
        for i in range(len(tfs)):
            gripper_mesh = o3d.geometry.TriangleMesh(gripper_model.mesh)
            gripper_mesh.transform(tfs[i])
            valid_grasps[i] = len(util.mesh_contains_points(gripper_mesh, point_cloud)) == 0

        print('**', '...done')
        t4 = timer()
        t_check_collisions += t4 - t3

        # after that, the only remaining degree of freedom is the standoff, for which there are various strategies:
        # - we could simply try to align finger tips with the ref/target points (what do we consider as the centre of
        #   the finger tips? it's a bit hard to say.
        # - we could move towards the centre of the circle for as long as possible without collision

        # finally, add to the result grasp set
        # (this might be inefficient, maybe we can initialise something with the number of desired grasps
        tfs = tfs[valid_grasps]
        gs.add(grasp.GraspSet.from_poses(tfs))

        if visualize:
            cone_vis = o3d.geometry.TriangleMesh.create_cone(radius, height)
            cone_vis.transform(tf)

            sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere_vis.translate(ref_point[:3])

            grippers = []
            for tf in tfs:
                gripper_vis = o3d.geometry.TriangleMesh(gripper_model.mesh)
                gripper_vis.transform(tf)
                grippers.append(gripper_vis)

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

            pc_list = [point_cloud, cone.vertices, target_points]
            if len(candidate_points) > 0:
                pc_list.append(candidate_points)
            if len(circle_points) > 0:
                pc_list.append(circle_points)

            o3d_obj_list = util.numpy_pc_to_o3d(pc_list)
            o3d_obj_list.extend([cone_vis, sphere_vis, frame])
            o3d_obj_list.extend(grippers)
            visualization.show_o3d_point_clouds(o3d_obj_list)

        n_sampled += 1  # todo: adjust this to real number of sampled (or leave it as n ref points?)
        if n_sampled >= n:
            break

    print('time to find target points', t_find_target_pts)
    print('time to compute the grasp poses', t_find_grasp_poses)
    print('time to check the collisions', t_check_collisions)
    print('total time', timer() - t0)

    return gs

