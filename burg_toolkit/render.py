import numpy as np
import trimesh

from . import util


class RandomCameraPoseGenerator:
    """
    A generator for certain amounts of random camera poses.
    This class can be used to generate an arbitrary number of camera poses. All poses look towards the origin. Their
    distance is chosen randomly with uniform distribution between ``cam_distance_min`` and ``cam_distance_max`` using
    the ``cam_distance_step``. ``rand_seed`` can be set to get reproducible results, ``number_of_poses`` defines the
    number of poses to be generated.

    :param number_of_poses: int, number of poses to generate
    :param cam_distance_min: float, minimum distance of camera poses (to origin)
    :param cam_distance_max: float, maximum distance of camera poses (to origin)
    :param upper_hemisphere: bool, whether to sample poses from upper hemisphere
    :param lower_hemisphere: bool, whether to sample poses from lower hemisphere
    :param rand_seed: int, provide seed for the rng for reproducible results, use None (default) for random seed
    """
    def __init__(self, number_of_poses=64, cam_distance_min=0.75, cam_distance_max=0.75, upper_hemisphere=True,
                 lower_hemisphere=True, rand_seed=None):
        self.number_of_poses = number_of_poses
        self.cam_distance_min = cam_distance_min
        self.cam_distance_max = cam_distance_max
        self.upper_hemisphere = upper_hemisphere
        self.lower_hemisphere = lower_hemisphere
        self.rand_seed = rand_seed

    def poses(self):
        """
        Generator function to get random poses as transform matrix, looking towards the origin.

        :return: 4x4 transform matrix as np array
        """
        rng = np.random.default_rng(self.rand_seed)

        if not (self.upper_hemisphere or self.lower_hemisphere):
            raise ValueError('bad configuration: need at least one hemisphere to generate camera poses.')
        # hemi factor will be 1 if only upper, and -1 if only sample from lower (and 0 if both)
        hemi_factor = 1*self.upper_hemisphere - 1*self.lower_hemisphere

        for _ in range(self.number_of_poses):
            unit_vec = util.generate_random_unit_vector()
            # account for hemisphere sampling and just project to other hemisphere if necessary
            if unit_vec[2] * hemi_factor < 0:
                unit_vec[2] *= -1

            # apply distance
            distance = rng.random() * (self.cam_distance_max - self.cam_distance_min) + self.cam_distance_min
            cam_pos = unit_vec * distance

            # convert to matrix oriented towards origin
            pose = util.look_at(cam_pos, target=[0, 0, 0], up=[0, 0, 1])
            yield pose


class IcoSphereCameraPoseGenerator:
    """
    The vertices of an icosphere will be used as camera positions, oriented such that they look towards the origin.
    In-plane rotations and distances will be applied accordingly. This is similar to what has been used e.g. for
    LINEMOD template matching (Hinterstoisser et al., ACCV, 2012).

    :param cam_distance_min: float, minimum distance of camera poses (to origin)
    :param cam_distance_max: float, maximum distance of camera poses (to origin)
    :param upper_hemisphere: bool, whether to use poses from upper hemisphere
    :param lower_hemisphere: bool, whether to use poses from lower hemisphere
    :param subdivisions: int, number of subdivisions of the icosphere, producing more positions (careful, <=5)
    :param in_plane_rotations: int, number of in-plane rotations for each position, will be evenly distributed over
                               the full 360 degree.
    :param scales: int, number of distances to use per position and in-plane rotation
    :param random_distances: bool, whether to use random distance values. If False, distances will be evenly spaced
                             between `cam_distance_min` and `cam_distance_max`.
    :param rand_seed: int, provide seed for the rng for reproducible results, use None (default) for random seed
                      (this only affects the distances and only if `random_distances` is set to True)
    """
    # Hinterstoisser says that they subdivide the icosahedron by recursive decomposition, iterating a few times.
    # So basically it is an icosphere, and in Hinterstoisser2012 it has 162 vertices at the upper hemisphere.
    # Different scales are used with step size of 10cm (size of polyhedron)
    # For each vertex of the polyhedron they use 36 angles (step size 10°)
    # that'd make 5832 views for the upper hemisphere only, with only one scale... wow

    def __init__(self, cam_distance_min=0.5, cam_distance_max=0.90, upper_hemisphere=True,
                 lower_hemisphere=False, subdivisions=2, in_plane_rotations=12, scales=2, random_distances=False,
                 rand_seed=None):
        self.cam_distance_min = cam_distance_min
        self.cam_distance_max = cam_distance_max
        self.upper_hemisphere = upper_hemisphere
        self.lower_hemisphere = lower_hemisphere
        self.subdivisions = subdivisions
        self.in_plane_rotations = in_plane_rotations
        self.scales = scales
        self.random_distances = random_distances
        self.rand_seed = rand_seed

    @property
    def number_of_poses(self):
        """
        :return: determines the number of poses this generator will produce, based on current configuration
        """
        n_vertices = 4**self.subdivisions * 10 + 2

        if self.upper_hemisphere + self.lower_hemisphere == 1:
            # only one of both
            n_ground = 4 * 2**self.subdivisions
            n_vertices = (n_vertices - n_ground) / 2
        elif not (self.upper_hemisphere or self.lower_hemisphere):
            # none of both (configuration not useful but possible)
            return 0

        return int(n_vertices * self.scales * self.in_plane_rotations)

    def poses(self):
        """
        Generator function to get the camera poses as transform matrix, looking towards the origin.

        :return: 4x4 transform matrix as np array
        """
        # the vertices of the icosphere are our camera positions
        cam_positions = trimesh.creation.icosphere(subdivisions=self.subdivisions, radius=1).vertices

        # only keep positions of appropriate hemisphere
        # it is debatable whether we want to keep the points on xy plane or not...
        if not self.lower_hemisphere:
            cam_positions = cam_positions[cam_positions[:, 2] > 0]
        if not self.upper_hemisphere:
            cam_positions = cam_positions[cam_positions[:, 2] < 0]

        if not (self.upper_hemisphere or self.lower_hemisphere):
            raise ValueError('bad configuration: need at least one hemisphere to generate camera poses.')

        rng = np.random.default_rng(self.rand_seed)

        # prepare distances (only used if not random)
        scale_distances = np.linspace(self.cam_distance_min, self.cam_distance_max, num=self.scales)
        # todo: having logarithmic instead of linspace could be slightly better, as changes in the rendered images
        #       get smaller for larger distances (also applies to randomly generated distances)

        # prepare in-plane rotations
        # this matrix will need to be applied in each step to rotate a bit more
        in_plane_rotation_matrix = np.eye(4)
        if self.in_plane_rotations > 1:
            angle = 2*np.pi / self.in_plane_rotations
            in_plane_rotation_matrix[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)],
                                                        [np.sin(angle), np.cos(angle)]])

        for cam_pos in cam_positions.reshape(-1, 3):
            # get the pose of the camera
            base_pose = util.look_at(cam_pos, target=[0, 0, 0], up=[0, 0, 1])

            for i in range(self.in_plane_rotations):

                for s in range(self.scales):
                    # scale position using distances (orientation not affected)
                    if self.random_distances:
                        d = rng.random() * (self.cam_distance_max - self.cam_distance_min) + self.cam_distance_min
                    else:
                        d = scale_distances[s]
                    pose = np.array(base_pose, copy=True)
                    pose[0:3, 3] = pose[0:3, 3] * d
                    yield pose

                # now apply the in-plane rotation
                base_pose = base_pose @ in_plane_rotation_matrix
