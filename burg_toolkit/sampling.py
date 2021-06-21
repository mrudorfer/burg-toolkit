import copy

import numpy as np
import trimesh
import open3d as o3d
from timeit import default_timer as timer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from . import grasp
from . import gripper
from . import util
from . import visualization
from . import mesh_processing


def rays_within_cone(axis, angle, n=10, uniform_on_plane=False):
    """
    Samples `n` rays originating from a cone apex.
    Cone is specified by it's `axis` vector from apex towards ground area and by the `angle` between axis and sides.
    We sample uniformly on the angle, causing the rays to be distributed more towards the center of the ground plane
    of the cone. If this is not desired, use `uniform_on_plane` parameter.

    :param axis: (3,) vector pointing from cone apex in the direction the rays will be cast
    :param angle: this is the max angle between axis and cast rays in [rad]
    :param n: number of rays to cast (defaults to 10)
    :param uniform_on_plane: if set to True, the rays will show uniform distribution on the ground plane of the cone.

    :return: (n, 3) numpy array with ray direction vectors (normalised)
    """
    # sample spherical coordinates: inclination in [0, angle], azimuth in [0, 2*pi]
    azimuth = np.random.uniform(0, 2*np.pi, n)
    if uniform_on_plane:
        inclination = np.arctan(np.random.uniform(0, np.tan(angle), n))
    else:
        inclination = np.random.uniform(0, angle, n)

    # convert from spherical to cartesian coordinates (radius = 1, i.e. vectors are normalized)
    cartesian = np.empty((n, 3, 1))
    cartesian[:, 0, 0] = np.sin(inclination)*np.cos(azimuth)
    cartesian[:, 1, 0] = np.sin(inclination)*np.sin(azimuth)
    cartesian[:, 2, 0] = np.cos(inclination)

    # transform so that z-axis aligns cone axis
    rot_mat = util.rotation_to_align_vectors([0, 0, 1], axis)
    ray_directions = np.matmul(rot_mat, cartesian)

    return ray_directions[:, :, 0]


class AntipodalGraspSampler:
    """
    A sampler for  antipodal grasps. Sampler looks for two contact points that satisfy the antipodal constraints
    for a given friction coefficient mu.
    """

    def __init__(self):
        self.gripper = None
        self.mu = 0.25
        self.n_orientations = 12
        self.n_rays = 100
        self.min_grasp_width = 0.002
        self.width_tolerance = 0.005
        self.max_targets_per_ref_point = 10
        self.no_contact_below_z = None
        self.verbose = True
        self.verbose_debug = False
        self.mesh = None
        self._trimesh = None

    def _interpolated_vertex_normals(self, points, triangle_indices):
        """
        computes the interpolated vertex normals for the given points in the associated triangles, corresponding
        to the mesh in self._trimesh.

        :param points: (n, 3)
        :param triangle_indices: (n,)

        :return: normals (n, 3)
        """

        normals = np.empty((len(points), 3))

        for idx, (point, triangle_idx) in enumerate(zip(points, triangle_indices)):
            faces = self._trimesh.faces[triangle_idx]  # contains idx of the three vertices
            vertices = self._trimesh.vertices[faces]
            vertex_normals = self._trimesh.vertex_normals[faces]

            # compute distance into some weighting factor
            distances = np.linalg.norm(vertices - point, axis=-1)
            # print('distances', distances)
            weights = 1/(distances**(1/2))  # gives a bit smoother result
            weights /= weights.sum()
            # print('weights:', weights, ' sum is ', weights.sum())
            normal = np.average(weights * vertex_normals, axis=0)
            normals[idx] = normal / np.linalg.norm(normal)

        return normals

    @staticmethod
    def construct_halfspace_grasp_set(reference_point, target_points, n_orientations):
        """
        For all pairs of reference point and target point grasps will be constructed at the center point.
        A grasp can be seen as a frame, the x-axis will point towards the target point and the z-axis will point
        in the direction from which the gripper will be approaching.
        This method only samples grasps from the halfspace above, the object will not be grasped from below.
        (Similar to GPNet data set.)

        :param reference_point: (3,) np array
        :param target_points: (n, 3) np array
        :param n_orientations: int, number of different orientations to use

        :return: grasp.GraspSet
        """
        reference_point = np.reshape(reference_point, (1, 3))
        target_points = np.reshape(target_points, (-1, 3))

        # center points in the middle between ref and target, x-axis pointing towards target point
        d = target_points - reference_point
        center_points = reference_point + 1/2 * d
        distances = np.linalg.norm(d, axis=-1)
        x_axes = d / distances[:, np.newaxis]

        # get unique x_axis representation = must only point upwards in z
        mask = x_axes[:, 2] < 0
        x_axes[mask] *= -1

        # y_tangent is constructed orthogonal to world z and gripper x
        y_tangent = -np.cross(x_axes, [0, 0, 1])
        y_tangent = y_tangent / np.linalg.norm(y_tangent, axis=-1)[:, np.newaxis]

        # the z-axes of the grasps are this y_tangent vector, but rotated around grasp's x-axis by theta
        # let's get the axis-angle representation to construct the rotations
        theta = np.linspace(0, np.pi, num=n_orientations)
        # multiply each of the x_axes with each of the thetas
        # [n, 3] * [m] --> [n, m, 3] --> [n*m, 3]
        # (first we have 1st x-axis with all thetas, then 2nd x-axis with all thetas, etc..)
        axis_angles = np.einsum('ik,j->ijk', x_axes, theta).reshape(-1, 3)
        rotations = R.from_rotvec(axis_angles)
        poses = np.empty(shape=(len(rotations), 4, 4))

        get_cosines_instead_of_width = False
        if get_cosines_instead_of_width:
            cosines = np.empty(shape=(len(rotations)))
            cosine_error = 0

        for i in range(len(x_axes)):
            for j in range(n_orientations):
                # apply the rotation to the y_tangent to get grasp z
                index = i*n_orientations + j
                rot_mat = rotations[index].as_matrix()
                z_axis = rot_mat @ y_tangent[i]

                # finally get y
                y_axis = np.cross(z_axis, x_axes[i])
                y_axis = y_axis / np.linalg.norm(y_axis)

                poses[index] = util.tf_from_xyz_pos(x_axes[i], y_axis, z_axis, center_points[i])

                if get_cosines_instead_of_width:
                    # let's confirm the cosine angle
                    cos = np.dot(z_axis, y_tangent[i])
                    cos_should_be = np.cos(theta[j])
                    cosines[index] = cos_should_be
                    cosine_error += np.square(cos - cos_should_be)

        gs = grasp.GraspSet.from_poses(poses)
        gs.widths = np.tile(distances, n_orientations).reshape(n_orientations, len(distances)).T.reshape(-1)

        if get_cosines_instead_of_width:
            print(f'cosine MSE is: {cosine_error}')
            gs.widths = cosines

        return gs

    @staticmethod
    def construct_grasp_set(reference_point, target_points, n_orientations):
        """
        For all pairs of reference point and target point grasps will be constructed at the center point.
        A grasp can be seen as a frame, the x-axis will point towards the target point and the z-axis will point
        in the direction from which the gripper will be approaching.

        :param reference_point: (3,) np array
        :param target_points: (n, 3) np array
        :param n_orientations: int, number of different orientations to use

        :return: grasp.GraspSet
        """
        reference_point = np.reshape(reference_point, (1, 3))
        target_points = np.reshape(target_points, (-1, 3))

        # center points in the middle between ref and target, x-axis pointing towards target point
        d = target_points - reference_point
        center_points = reference_point + 1/2 * d
        distances = np.linalg.norm(d, axis=-1)
        x_axis = d / distances[:, np.newaxis]

        # construct y-axis and z-axis orthogonal to x-axis
        y_axis = np.zeros(x_axis.shape)
        while (np.linalg.norm(y_axis, axis=-1) == 0).any():
            tmp_vec = util.generate_random_unit_vector()
            y_axis = np.cross(x_axis, tmp_vec)
            # todo: using the same random unit vec to construct all frames will lead to very similar orientations
            #       we might want to randomize this even more by using individual, random unit vectors
        y_axis = y_axis / np.linalg.norm(y_axis, axis=-1)[:, np.newaxis]
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1)[:, np.newaxis]

        # with all axes and the position, we can construct base frames
        tf_basis = util.tf_from_xyz_pos(x_axis, y_axis, z_axis, center_points).reshape(-1, 4, 4)

        # generate transforms for the n_orientations (rotation around x axis)
        theta = np.arange(0, 2*np.pi, 2*np.pi / n_orientations)
        tf_rot = np.tile(np.eye(4), (n_orientations, 1, 1))
        tf_rot[:, 1, 1] = np.cos(theta)
        tf_rot[:, 1, 2] = -np.sin(theta)
        tf_rot[:, 2, 1] = np.sin(theta)
        tf_rot[:, 2, 2] = np.cos(theta)

        # apply transforms
        tfs = np.matmul(tf_basis[np.newaxis, :, :, :], tf_rot[:, np.newaxis, :, :]).reshape(-1, 4, 4)
        gs = grasp.GraspSet.from_poses(tfs)

        # add distances as gripper widths (repeat n_orientation times)
        gs.widths = np.tile(distances, n_orientations).reshape(n_orientations, len(distances)).T.reshape(-1)

        return gs

    def sample(self, n=10):
        # probably do some checks before starting... is gripper None? is mesh None? ...
        if self.verbose:
            print('preparing to sample grasps...')
        # we need collision operations which are not available in o3d yet
        # hence convert the mesh to trimesh
        self._trimesh = util.o3d_mesh_to_trimesh(self.mesh)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self._trimesh)

        # we need to sample reference points from the mesh
        # since uniform sampling methods seem to go deterministically through the triangles and sample randomly within
        # triangles, we cannot sample individual points (as this would get very similar points all the time).
        # therefore, we first sample many points at once and then just use some of these at random
        # let's have a wild guess of how many are many ...
        n_sample = np.max([n, 5000, 3*len(self.mesh.triangles)])
        ref_points = util.o3d_pc_to_numpy(mesh_processing.poisson_disk_sampling(self.mesh, n_points=n_sample))
        np.random.shuffle(ref_points)

        if self.no_contact_below_z is not None:
            keep = ref_points[:, 2] > self.no_contact_below_z
            ref_points = ref_points[keep]

        if self.verbose:
            print(f'sampled {len(ref_points)} first contact point candidates, beginning to find grasps.')

        # determine some parameters for casting rays in a friction cone
        angle = np.arctan(self.mu)
        if self.verbose_debug:
            print('mu is', self.mu, 'hence angle of friction cone is', np.rad2deg(angle), '°')

        gs = grasp.GraspSet()
        gs_contacts = np.empty((0, 2, 3))
        i_ref_point = 0

        with tqdm(total=n, disable=not self.verbose) as progress_bar:
            while len(gs) < n:
                # todo check if ref point still in range
                p_r = ref_points[i_ref_point, 0:3]
                n_r = ref_points[i_ref_point, 3:6]
                if self.verbose_debug:
                    print(f'sampling ref point no {i_ref_point}: point {p_r}, normal {n_r}')
                i_ref_point = (i_ref_point + 1) % len(ref_points)

                # cast random rays from p_r within the friction cone to identify potential contact points
                ray_directions = rays_within_cone(-n_r, angle, self.n_rays)
                ray_origins = np.tile(p_r, (self.n_rays, 1))
                locations, _, index_tri = intersector.intersects_location(
                    ray_origins, ray_directions, multiple_hits=True)
                if self.verbose_debug:
                    print(f'* casting {self.n_rays} rays, leading to {len(locations)} intersection locations')
                if len(locations) == 0:
                    continue

                # eliminate intersections with origin
                mask_is_not_origin = ~np.isclose(locations, p_r, atol=1e-11).all(axis=-1)
                locations = locations[mask_is_not_origin]
                index_tri = index_tri[mask_is_not_origin]
                if self.verbose_debug:
                    print(f'* ... of which {len(locations)} are not with ref point')
                if len(locations) == 0:
                    continue

                # eliminate contact points too far or too close
                distances = np.linalg.norm(locations - p_r, axis=-1)
                mask_is_within_distance = \
                    (distances <= self.gripper.opening_width - self.width_tolerance)\
                    | (distances >= self.min_grasp_width)
                locations = locations[mask_is_within_distance]
                index_tri = index_tri[mask_is_within_distance]
                if self.verbose_debug:
                    print(f'* ... of which {len(locations)} are within gripper width constraints')
                if len(locations) == 0:
                    continue

                normals = self._interpolated_vertex_normals(locations, index_tri)

                if self.verbose_debug:
                    # visualize candidate points and normals

                    sphere_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    sphere_vis.translate(p_r)
                    sphere_vis.compute_vertex_normals()

                    o3d_pc = util.numpy_pc_to_o3d(np.concatenate([locations, normals], axis=1))
                    obj_list = [self.mesh, sphere_vis, o3d_pc]
                    arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=1 / 10000,
                        cone_radius=1.5 / 10000,
                        cylinder_height=5.0 / 1000,
                        cone_height=4.0 / 1000,
                        resolution=20,
                        cylinder_split=4,
                        cone_split=1)
                    arrow.compute_vertex_normals()
                    for point, normal in zip(locations, normals):
                        my_arrow = o3d.geometry.TriangleMesh(arrow)
                        my_arrow.rotate(util.rotation_to_align_vectors([0, 0, 1], normal), center=[0, 0, 0])
                        my_arrow.translate(point)
                        obj_list.append(my_arrow)

                    visualization.show_o3d_point_clouds(obj_list)
                    # o3d.visualization.draw_geometries(obj_list, point_show_normal=True)

                # compute angles to check antipodal constraints
                d = (locations - p_r).reshape(-1, 3)
                signs = np.zeros(len(d))
                angles = util.angle(d, normals, sign_array=signs, as_degree=False)

                # exclude target points which do not have opposing surface orientations
                # positive sign means vectors are facing into a similar direction as connecting vector, as expected
                mask_faces_correct_direction = signs > 0
                locations = locations[mask_faces_correct_direction]
                normals = normals[mask_faces_correct_direction]
                angles = angles[mask_faces_correct_direction]
                if self.verbose_debug:
                    print(f'* ... of which {len(locations)} are generally facing in opposing directions')
                if len(locations) == 0:
                    continue

                # check friction cone constraint
                mask_friction_cone = angles <= angle
                locations = locations[mask_friction_cone]
                normals = normals[mask_friction_cone]
                angles = angles[mask_friction_cone]
                if self.verbose_debug:
                    print(f'* ... of which {len(locations)} are satisfying the friction constraint')
                if len(locations) == 0:
                    continue

                # check below z contact
                if self.no_contact_below_z is not None:
                    mask_below_z = locations[:, 2] > self.no_contact_below_z
                    locations = locations[mask_below_z]
                    normals = normals[mask_below_z]
                    angles = angles[mask_below_z]
                    if self.verbose_debug:
                        print(f'* ... of which {len(locations)} are above the specified z value')
                    if len(locations) == 0:
                        continue

                # actually construct all the grasps (with n_orientations)
                # todo: maybe we can choose more intelligently here
                #       e.g. some farthest point sampling so grasps are likely to be more diverse
                if len(locations) > self.max_targets_per_ref_point:
                    indices = np.arange(len(locations))
                    np.random.shuffle(indices)
                    locations = locations[indices[:self.max_targets_per_ref_point]]
                if self.verbose_debug:
                    print(f'* ... of which we randomly choose {len(locations)} to construct grasps')

                # grasps = self.construct_grasp_set(p_r, locations, self.n_orientations)
                grasps = self.construct_halfspace_grasp_set(p_r, locations, self.n_orientations)

                # also compute the contact points
                contacts = np.empty((len(locations), 2, 3))
                contacts[:, 0] = p_r
                contacts[:, 1] = locations
                contacts = np.repeat(contacts, self.n_orientations, axis=0)

                gs_contacts = np.concatenate([gs_contacts, contacts], axis=0)
                gs.add(grasps)
                if self.verbose_debug:
                    print(f'* added {len(grasps)} grasps (with {self.n_orientations} orientations for each point pair)')
                progress_bar.update(len(grasps))

        return gs, gs_contacts

    def check_collisions(self, graspset, use_width=True, width_tolerance=0.01, additional_objects=None,
                         exclude_shape=False):
        """
        This will check collisions for the given graspset using the gripper mesh of the object's gripper.

        :param graspset: The n-elem grasp.GraspSet to check collisions for
        :param use_width: If True, will squeeze the gripper mesh to fit the opening width plus width_tolerance
        :param width_tolerance: As squeezing the gripper to the distance of the contact points will most certainly lead
                                to collisions, this tolerance is added to the opening width.
        :param additional_objects: list of o3d meshes that should be included in the collision manager (e.g. plane)
        :param exclude_shape: bool, if True will only check collisions with provided additional objects. Note that if
                              this is set to True additional objects must be provided.
        """
        if not additional_objects and exclude_shape:
            raise ValueError('no collision objects specified.')

        # we need collision operations which are not available in o3d yet
        # hence use trimesh
        manager = trimesh.collision.CollisionManager()
        if not exclude_shape:
            self._trimesh = util.o3d_mesh_to_trimesh(self.mesh)
            manager.add_object('shape', self._trimesh)

        # additional objects
        if additional_objects:
            for i, obj in enumerate(additional_objects):
                manager.add_object(f'add_obj_{i}', util.o3d_mesh_to_trimesh(obj))

        gripper_mesh = copy.deepcopy(self.gripper.mesh)
        tf = self.gripper.tf_base_to_TCP
        gripper_mesh.transform(tf)
        gripper_mesh = util.o3d_mesh_to_trimesh(gripper_mesh)

        collision_array = np.empty(len(graspset), dtype=np.bool)

        if self.verbose:
            print('checking collisions...')
        for i, g in tqdm(enumerate(graspset), disable=not self.verbose):
            tf_squeeze = np.eye(4)
            if use_width:
                tf_squeeze[0, 0] = (g.width + width_tolerance) / self.gripper.opening_width
            collision_array[i] = manager.in_collision_single(gripper_mesh, transform=g.pose @ tf_squeeze)

        return collision_array


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

        print('*', len(candidate_points), 'candidate points remaining')
        if len(candidate_points) == 0:
            continue

        best_idx = np.argmin(candidate_ppf[:, 4])

        tfs = AntipodalGraspSampler.construct_grasp_set(
            ref_point[:3], candidate_points[best_idx][:3], circle_discretisation_steps).poses

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

