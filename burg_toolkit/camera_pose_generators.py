from abc import ABC, abstractmethod
import random
import math

import numpy as np
import trimesh

from . import util


class AbstractCameraPoseGenerator(ABC):
    """An abstract base class for camera pose generators."""
    def __init__(self):
        self.distance_factor = 1    # can be used to convert distances (e.g. from mm to m)
        self._current_pose = 0
        self.number_of_poses = 0

    @abstractmethod
    def _generate_pose(self):
        """
        This method generates the next pose of the generator and must be implemented by the derived classes.

        :return: 4x4 transform matrix as np array with shape=(4, 4)
        """
        pass

    def has_next_pose(self):
        """
        Checks if further poses are available.

        :return: True, if less than `number_of_poses` have been retrieved yet, else returns False.
        """
        return self._current_pose < self.number_of_poses

    def get_next_pose(self):
        """
        Retrieves the next camera pose as 4x4 transform matrix.

        :return: 4x4 transform matrix as np array with shape=(4, 4), or None if `number_of_poses` has been exceeded.
        """
        if not self.has_next_pose():
            return None

        pose = self._generate_pose()
        self._current_pose += 1

        return pose

    def reset(self):
        """
        Resets the CameraPoseGenerator.
        CameraPoseGenerator will generate the same series of poses again after a reset.
        """
        # reset counter
        self._current_pose = 0

    @staticmethod
    def look_at(vertex, target=[0, 0, 0]):
        """Returns a 4x4 matrix directed towards the target, with random in-plane rotation"""
        # todo:
        #   move this to util
        #   also check if we can make canonical instead of random in-plane rotation
        #   ideally we want to specify the "upright" direction of the camera in world frame
        # rotate the coordinate system to look at the target
        # get base vectors of the frame, use a random in-plane rotation
        # inspired by https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        z_vec = vertex - target
        z_vec /= np.linalg.norm(z_vec)

        unit_vec = util.generate_random_unit_vector()
        x_vec = np.cross(unit_vec, z_vec)

        # if cross product failed (x_vec should be all zeros), just repeat
        while np.array_equal(x_vec, np.zeros(3)):
            unit_vec = util.generate_random_unit_vector()
            x_vec = np.cross(unit_vec, z_vec)

        y_vec = np.cross(z_vec, x_vec)

        # now let's normalize the vectors to not skew anything
        x_vec /= np.linalg.norm(x_vec)
        y_vec /= np.linalg.norm(y_vec)

        # convert to transform matrix
        matrix = np.identity(4)
        matrix[0:3, 0] = x_vec
        matrix[0:3, 1] = y_vec
        matrix[0:3, 2] = z_vec
        matrix[0:3, 3] = vertex

        return matrix


class RandomCameraPoseGenerator(AbstractCameraPoseGenerator):
    """
    A generator for certain amounts of random camera poses.
    This class can be used to generate an arbitrary number of camera poses. All poses look towards the origin. Their
    distance is chosen randomly with uniform distribution between ``cam_distance_min`` and ``cam_distance_max`` using
    the ``cam_distance_step``. ``rand_seed`` can be set to get reproducible results, ``number_of_poses`` defines the
    number of poses to be generated.
    """
    def __init__(self, *args, **kwargs):
        # call the super constructor
        super().__init__()

        # todo:
        #   this is a constructor from config file
        #   let's make this a proper constructor instead, with a static-func .from_config_file()?

        # first, set default values for member variables
        self.cam_distance_min = 0.75
        self.cam_distance_max = 0.75
        self.cam_distance_step = 0.1

        # number of poses to generate
        self.number_of_poses = 10
        self.rand_seed = 12  # todo: check if this is really reproducible or if we broke it

        # now, let's interpret the given config values (this is some fancy magic)
        # also see https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # now we have to check types, as everything probably has been read as a string
        self.distance_factor = float(self.distance_factor)
        self.cam_distance_min = int(self.cam_distance_min)
        self.cam_distance_max = int(self.cam_distance_max)
        self.cam_distance_step = int(self.cam_distance_step)

        self.number_of_poses = int(self.number_of_poses)
        self.rand_seed = int(self.rand_seed)

        self.reset()

    def _generate_pose(self):
        """
        Generates a random pose as transform matrix, that looks towards the origin.

        :return: 4x4 transform matrix as np array
        """
        # get some unit vector
        unit_vec = util.generate_random_unit_vector()

        # scale with random distance (max is excluded, therefore we add 1)
        distance = random.randrange(self.cam_distance_min, self.cam_distance_max + 1, self.cam_distance_step)
        distance *= self.distance_factor  # unit conversion
        vec = unit_vec * distance

        # convert to matrix oriented towards origin
        matrix = AbstractCameraPoseGenerator.look_at(vec)
        return matrix

    def reset(self):
        super().reset()

        # also reset the random seed
        random.seed(self.rand_seed)


class IcoSphereCameraPoseGenerator(AbstractCameraPoseGenerator):
    """
    A camera pose generator that gives out-of-plane rotations on an icosphere and adds in-plane rotations to it.
    """
    # seek help for construction of ico spheres
    # http://www.songho.ca/opengl/gl_sphere.html
    # https://stackoverflow.com/questions/43107006/faces-missing-when-drawing-icosahedron-in-opengl-following-code-in-redbook
    # Hinterstoisser says that they subdivide the icosahedron by recursive decomposition, iterating a few times.
    # So basically it is an icosphere, and in Hinterstoisser2012 it has 162 vertices at the upper hemisphere.
    # Different scales are used with step size of 10cm (size of polyhedron)
    # For each vertex of the polyhedron they use 36 angles (step size 10°)
    # that'd make 5832 views for the upper hemisphere only, with only one scale... wow

    def __init__(self, *args, **kwargs):
        # call the super constructor
        super().__init__()

        # todo:
        #   again, make this a proper constructor...
        #   we could also introduce random distances

        # first, set default values for member variables
        self.cam_distance_min = 90  # could be cm, will be multiplied with distance_factor
        self.cam_distance_max = 75
        self.scales = 2  # 1

        self.upper_hemisphere = True
        self.lower_hemisphere = True
        self.in_plane_rotations = 18  # 18
        self.subdivisions = 1  # 2

        # now, let's interpret the given values (this is some fancy magic)
        # also see https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # now we have to check types, as everything has been read as a string
        self.distance_factor = float(self.distance_factor)
        self.cam_distance_min = int(self.cam_distance_min)
        self.cam_distance_max = int(self.cam_distance_max)
        self.scales = int(self.scales)

        self.upper_hemisphere = str(self.upper_hemisphere).lower() in ['true', '1', 'yes', 'y']
        self.lower_hemisphere = str(self.lower_hemisphere).lower() in ['true', '1', 'yes', 'y']
        self.in_plane_rotations = int(self.in_plane_rotations)
        self.subdivisions = int(self.subdivisions)

        # this is the actual list of vertices we will use to generate the poses, along with some indices
        self._vertices = []
        self._previous_mat = IcoSphereCameraPoseGenerator._get_xy_transform(0)  # store previous mat
        self.reset()

    def _build_icosphere_vertices(self):
        """Builds a list of vertices for an icosphere with specified subdivisions

        Returns
            List of list of all vertices belonging to the icosphere.
        """

        # get the vertices from an icosphere created in trimesh
        sphere = trimesh.creation.icosphere(subdivisions=self.subdivisions, radius=1)
        self._vertices = sphere.vertices
        print(f"created icosphere with {self.subdivisions:d} subdivisions ({self._vertices.shape[0]:d} vertices).")

        # sort out points if certain hemisphere is unwanted
        if not self.lower_hemisphere:
            print("removing vertices on lower hemisphere...")
            self._vertices = self._vertices[self._vertices[:, 2] >= 0]
        if not self.upper_hemisphere:
            print("removing vertices on upper hemisphere...")
            self._vertices = self._vertices[self._vertices[:, 2] <= 0]

        print(f'using {self._vertices.shape[0]:d} vertices.')

    @staticmethod
    def _get_xy_transform(angle):
        """Returns a transformation matrix to rotate in xy plane by specified angle in degree"""

        angle = angle / 180 * math.pi
        matrix = np.identity(4)
        rot_2d = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        matrix[:2, :2] = rot_2d
        return matrix

    def _generate_pose(self):
        # identify indicies with floor division
        vertex_idx = self._current_pose // (self.in_plane_rotations * self.scales)
        scale_idx = self._current_pose // self.in_plane_rotations % self.scales
        ipr_idx = self._current_pose % self.in_plane_rotations  # must be fastest changing index

        # scale the vector with appropriate scale
        # todo: using logarithmic scales instead of evenly distributed might be beneficial
        # todo: also, having slight aberrations in distance might also help
        distance_step = (self.cam_distance_max - self.cam_distance_min) / max(self.scales - 1, 1)
        distance = self.cam_distance_min + scale_idx * distance_step
        distance *= self.distance_factor

        vertex = self._vertices[vertex_idx]
        vertex *= distance / np.linalg.norm(vertex)

        # find appropriate in-plane rotation, or generate new one
        if ipr_idx == 0:
            matrix = AbstractCameraPoseGenerator.look_at(vertex)
        else:
            angle = 360 / self.in_plane_rotations
            rot_xy = IcoSphereCameraPoseGenerator._get_xy_transform(angle)
            matrix = np.dot(self._previous_mat, rot_xy)

        self._previous_mat = matrix
        return matrix

    def reset(self):
        super().reset()
        self._build_icosphere_vertices()
        self.number_of_poses = len(self._vertices) * self.in_plane_rotations * self.scales
        print(f"number of poses: {self.number_of_poses:d}")
