import copy
import logging

import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from . import core
from . import grasp
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
        self.max_targets_per_ref_point = 1
        self.only_grasp_from_above = False
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

        gs = grasp.GraspSet.from_poses(poses)
        gs.widths = np.tile(distances, n_orientations).reshape(n_orientations, len(distances)).T.reshape(-1)

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
        self._trimesh = mesh_processing.as_trimesh(self.mesh)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self._trimesh)

        # we need to sample reference points from the mesh
        # since uniform sampling methods seem to go deterministically through the triangles and sample randomly within
        # triangles, we cannot sample individual points (as this would get very similar points all the time).
        # therefore, we first sample many points at once and then just use some of these at random
        # let's have a wild guess of how many are many ...
        n_sample = np.max([n, 1000, len(self.mesh.triangles)])
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

                if self.only_grasp_from_above:
                    grasps = self.construct_halfspace_grasp_set(p_r, locations, self.n_orientations)
                else:
                    grasps = self.construct_grasp_set(p_r, locations, self.n_orientations)

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
            self._trimesh = mesh_processing.as_trimesh(self.mesh)
            manager.add_object('shape', self._trimesh)

        # additional objects
        if additional_objects:
            for i, obj in enumerate(additional_objects):
                manager.add_object(f'add_obj_{i}', mesh_processing.as_trimesh(obj))

        gripper_mesh = copy.deepcopy(self.gripper.mesh)
        tf = self.gripper.tf_base_to_TCP
        gripper_mesh.transform(tf)
        gripper_mesh = mesh_processing.as_trimesh(gripper_mesh)

        collision_array = np.empty(len(graspset), dtype=np.bool)

        if self.verbose:
            print('checking collisions...')
        for i, g in tqdm(enumerate(graspset), disable=not self.verbose):
            tf_squeeze = np.eye(4)
            if use_width:
                tf_squeeze[0, 0] = (g.width + width_tolerance) / self.gripper.opening_width
            collision_array[i] = manager.in_collision_single(gripper_mesh, transform=g.pose @ tf_squeeze)

        return collision_array


def grasp_perturbations(grasps, radii=None, include_original_grasp=True):
    """
    Given a grasp g (or set of grasps), it will compile a grasp set with perturbed grasp poses.
    Poses will be sampled on 6d spheres (1mm translation = 1deg rotation), where each dimension will be set to
    positive and negative radius, i.e. for each sphere we get 12 perturbed grasp poses.

    :param grasps: a grasp.Grasp, or a grasp.GraspSet
    :param radii: a list with radii of the spheres. if None, defaults to [5, 10, 15]
    :param include_original_grasp: whether or not to include the original grasps in the return set

    :return: grasp.GraspSet with perturbed grasps. if a graspset has been provided, the returned set will have all
             perturbations in order, ie first all perturbations of the first grasp, then all perturbations of the
             second grasp, and so on.
    """
    if radii is None:
        radii = [5, 10, 15]
    elif not isinstance(radii, list):
        raise ValueError('radii must be a list (or None)')

    if not isinstance(grasps, grasp.GraspSet):
        if isinstance(grasps, grasp.Grasp):
            grasps = grasps.as_grasp_set()
        else:
            raise ValueError('g must be a grasp.Grasp or a grasp.GraspSet')

    n_grasps = len(grasps) * (len(radii) * 12 + int(include_original_grasp))
    gs = grasp.GraspSet(n=n_grasps)
    i = 0

    print(f'given {len(grasps)} grasps, we construct {n_grasps} perturbations in total.')
    for g in tqdm(grasps):
        if include_original_grasp:
            gs[i] = g
            i += 1

        for radius in radii:
            shift_mm = radius / 1000  # convert to mm
            for translation_idx in range(3):
                for sign in [1, -1]:
                    pose = copy.deepcopy(g.pose)
                    translation_axis = pose[0:3, translation_idx]
                    pose[0:3, 3] = pose[0:3, 3] + sign * shift_mm * translation_axis
                    gs[i].pose = pose
                    i += 1

            rot_rad = np.deg2rad(radius)
            for rotation_idx in range(3):
                for sign in [1, -1]:
                    pose = copy.deepcopy(g.pose)
                    rotation_axis = pose[0:3, rotation_idx]
                    pose[:3, :3] = R.from_rotvec(sign * rot_rad * rotation_axis).as_matrix() @ pose[:3, :3]
                    gs[i].pose = pose
                    i += 1

    return gs


def random_poses(n):
    """
    Samples random poses, i.e. random orientations with random positions in [0, 1].

    :param n: number of poses to return

    :return: numpy array with shape (n, 4, 4)
    """
    tfs = np.zeros((n, 4, 4))
    tfs[:, 3, 3] = 1
    tfs[:, 0:3, 0:3] = R.random(n).as_matrix().reshape(-1, 3, 3)
    tfs[:, 0:3, 3] = np.random.random((n, 3))
    return tfs


def sample_scene(object_library, ground_area, instances_per_scene, instances_per_object=1, max_tries=20):
    """
    Samples a physically plausible scene using the objects in the given object_library.

    :param object_library: core.ObjectLibrary, which objects to sample from
    :param ground_area: (l, w) length (x-axis) and width (y-axis) of the ground area
    :param instances_per_scene: number of desired object instances in the scene
    :param instances_per_object: number of allowed instances of each object type
    :param max_tries: tries to add an object at most `max_tries` times, if that fails it will return
                      a scene with fewer instances than have been asked for.

    :return: core.Scene
    """
    scene = core.Scene(ground_area=ground_area)
    rng = np.random.default_rng()

    # pick objects from object_library:
    population = [obj_type for obj_type in object_library.values() for _ in range(instances_per_object)]
    obj_types = rng.choice(population, instances_per_scene, replace=False)

    manager = trimesh.collision.CollisionManager()

    # try to add each object to the scene
    for i, object_type in enumerate(obj_types):
        success = False
        for n_tries in range(max_tries):
            n_tries += 1

            # choose random rotation around z-axis and random stable pose of the object
            angle = rng.random() * np.pi * 2
            tf_rot = np.eye(4)
            tf_rot[:3, :3] = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
            pose = tf_rot @ object_type.stable_poses.sample_pose(uniformly=True)

            # now sample some xy displacement on ground plane
            # to find the correct range for the offset, we need to account for the mesh bounds
            instance = core.ObjectInstance(object_type, pose)
            mesh = instance.get_mesh()
            min_x, min_y, _ = mesh.get_min_bound()
            max_x, max_y, _ = mesh.get_max_bound()
            range_x, range_y = ground_area[0] - (max_x - min_x), ground_area[1] - (max_y - min_y)
            x, y = rng.random() * range_x - min_x, rng.random() * range_y - min_y

            instance.pose[0, 3] = x + pose[0, 3]
            instance.pose[1, 3] = y + pose[1, 3]

            # check collision
            # note: trimesh docs say by using the same name for an object, the manager replaces the object if it has
            # been previously added, however, this does not seem to work properly, so we explicitly remove the object
            manager.add_object(f'obj{i}', mesh_processing.as_trimesh(instance.get_mesh()))
            if manager.in_collision_internal():
                manager.remove_object(f'obj{i}')
            else:
                # can add to scene and do next object
                scene.objects.append(instance)
                success = True
                break

        if not success:
            logging.warning(f'Could not add object to scene, exceeded number of max_tries ({max_tries}). Returning ' +
                            f'scene with fewer object instances than requested.')

    # todo: simulate scene to make sure it's stable
    # since objects are not touching, this should actually not be necessary.
    # however, just to be sure...
    # question is, do we do this in this function? it is actually separate from sampling, so potentially we should
    # do this somewhere else (the caller shall decide)
    return scene

