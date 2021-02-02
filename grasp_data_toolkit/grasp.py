import numpy as np
import quaternion


class Grasp:
    """
    a grasp.
    for efficiency, all information will be stored in an internal numpy array and can be retrieved via
    get and set functions
    """
    ARRAY_LEN = 13

    def __init__(self, np_array=None):
        """
        initialises a grasp
        :param np_array: the internal numpy array, which is structured as follows:
            [translation(3), rotation_matrix(3x3), score], i.e. length = 13
        """
        if np_array is None:
            np_array = np.zeros(self.ARRAY_LEN)

        assert(len(np_array) == self.ARRAY_LEN), 'provided np_array has wrong length.'

        self._grasp_array = np_array.astype(np.float32)

    @property
    def internal_array(self):
        """
        :return: the internal ndarray representation. only use when you know what you're doing.
        """
        return self._grasp_array

    @internal_array.setter
    def internal_array(self, np_array):
        """
        :param np_array: the new internal array, must be of length Grasp.ARRAY_LEN. use with caution.
        """
        assert(len(np_array) == self.ARRAY_LEN), 'provided np_array has wrong length.'
        self._grasp_array = np_array.astype(np.float32)

    @property
    def translation(self):
        """
        :return: translation as np-array with length 3
        """
        return self._grasp_array[0:3]

    @translation.setter
    def translation(self, translation):
        """
        :param translation: np-array with length 3
        """
        self._grasp_array[0:3] = np.asarray(translation).astype(np.float32)

    @property
    def rotation_matrix(self):
        """
        :return: rotation matrix as np array 3x3
        """
        return self._grasp_array[3:12].reshape((3, 3))

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix):
        """
        :param rotation_matrix: the rotation matrix, 3x3 numpy array
        """
        self._grasp_array[3:12] = rotation_matrix.reshape(9).astype(np.float32)

    @property
    def pose(self):
        """
        :return: the 6d pose as homogenous transformation matrix 4x4 np array
        """
        pose = np.eye(4)
        pose[0:3, 0:3] = self.rotation_matrix
        pose[0:3, 3] = self.translation
        return pose

    @pose.setter
    def pose(self, pose):
        """
        :param pose: the 6d pose as homogenous transformation matrix 4x4 np array
        """
        self.translation = pose[0:3, 3]
        self.rotation_matrix = pose[0:3, 0:3]

    @property
    def score(self):
        """
        :return: the score of this grasp as float value
        """
        return self._grasp_array[12]

    @score.setter
    def score(self, score):
        """
        :param score: a float value as the score
        """
        self._grasp_array[13] = float(score)

    def distance_to(self, other_grasp):
        """
        computes the distance of this grasp to the other_grasp according to Eppner et al., 2019
        translation and rotation terms are weighted so that a distance of 1 equals to either 1 mm or 1 degree
        :param other_grasp: the other grasp of type Grasp
        :return: the distance between this grasp and the other grasp as float value
        """
        q1 = quaternion.from_rotation_matrix(self.rotation_matrix)
        q2 = quaternion.from_rotation_matrix(other_grasp.rotation_matrix)

        q1 = quaternion.as_float_array(q1)
        q2 = quaternion.as_float_array(q2)

        t1 = self.translation
        t2 = other_grasp.translation

        # scale up so that 1mm -> distance=1
        translation_dist = 1000 * np.linalg.norm(t1 - t2)
        # scale so that 1deg -> distance=1 (I'm not exactly sure why I need to multiply with 360 instead of 180, but
        # else the values will be too small. Also the dot product does not conjugate q2, but when I did this it
        # gave unreasonable values. This version here seems to work reasonably well now.
        rotation_dist = np.arccos(np.abs(np.dot(q1, q2)))/np.pi*360

        return translation_dist + rotation_dist


class GraspSet:
    """
    this is a collection that holds a number of grasps
    """

    def __init__(self, np_array=None):
        """
        initialises a grasp set
        :param np_array: the internal numpy array, which is of shape (n, Grasp.ARRAY_LEN) and each row is a Grasp
        """
        if np_array is None:
            np_array = np.zeros((0, Grasp.ARRAY_LEN), dtype=np.float32)

        assert(np_array.shape[1] == Grasp.ARRAY_LEN), 'provided np_array has wrong shape.'

        self._gs_array = np_array.astype(np.float32)

    @classmethod
    def from_translations_and_quaternions(cls, poses):
        """
        creates a grasp set from poses specified with translation (3) and quaternion (4)
        :param poses: (n, 7) np array with position (0:3) and quaternion (3:7)
        :return: grasp set with corresponding poses, all other fields are zero-initialised
        """
        gs = cls(np.zeros((poses.shape[0], Grasp.ARRAY_LEN), dtype=np.float32))
        # get translations
        gs.translations = poses[:, 0:3]

        # get rotation matrices (using the numpy-quaternion package, which offers vectorized implementations)
        quaternions = quaternion.as_quat_array(poses[:, 3:7])
        rotation_matrices = quaternion.as_rotation_matrix(quaternions)
        gs.rotation_matrices = rotation_matrices

        return gs

    @classmethod
    def from_translations(cls, translations):
        """
        creates a grasp set from translations (x, y, z) only - rotation matrices will be eye(3)
        :param translations: (n, 3) np array with position
        :return: grasp set with corresponding grasping points with default orientation, other fields zero-initialised
        """
        np_array = np.zeros((translations.shape[0], Grasp.ARRAY_LEN), dtype=np.float32)
        np_array[:, 0:3] = translations

        # set canonical orientations (= np.eye(3) for each grasp)
        np_array[:, 3] = 1.0
        np_array[:, 7] = 1.0
        np_array[:, 11] = 1.0

        return cls(np_array)

    def __len__(self):
        return self._gs_array.shape[0]

    def __getitem__(self, item):
        """
        :param item: can be index, slice or array
        :return: if single index, then grasp object, else grasp set object - note that these will be shallow copies
        """
        if type(item) == int:
            return Grasp(self._gs_array[item])
        elif (type(item) == slice) or (type(item) == list) or (type(item) == np.ndarray):
            return GraspSet(self._gs_array[item])
        else:
            raise TypeError('unknown index type calling GraspSet.__getitem__')

    def __setitem__(self, key, value):
        """
        :param key: can be index, slice or array
        :param value: if single index: Grasp object or GraspSet, else: GraspSet object
        """
        if type(key) == int:
            if type(value) is GraspSet:
                if len(value) != 1:
                    raise ValueError('If type(index) is int, value needs to be Grasp or GraspSet of length 1.')
                value = Grasp(value.internal_array)
            if type(value) is not Grasp:
                raise TypeError('Provided value has wrong type. Expected Grasp or GraspSet of length 1.')
            self._gs_array[key] = value.internal_array

        elif (type(key) == slice) or (type(key) == list) or (type(key) == np.ndarray):
            if type(value) is not GraspSet:
                raise TypeError('Provided value has wrong type. Expected GraspSet.')
            self._gs_array[key] = value.internal_array
        else:
            raise TypeError('unknown index type calling GraspSet.__setitem__')

    @property
    def internal_array(self):
        """
        :return: gives the internal numpy array representation of shape (N, Grasp.ARRAY_LEN)
        """
        return self._gs_array

    @internal_array.setter
    def internal_array(self, np_array):
        """
        :param np_array: sets the new internal np array, must be of dim (n, Grasp.ARRAY_LEN). use with caution.
        """
        assert(np_array.shape[1] == Grasp.ARRAY_LEN), 'provided np_array has wrong shape.'
        self._gs_array = np_array.astype(np.float32)

    @property
    def translations(self):
        """
        :return: (n, 3) np array with translations
        """
        return self._gs_array[:, 0:3]

    @translations.setter
    def translations(self, translations):
        """
        :param translations: an (n, 3) np array
        :return:
        """
        assert(translations.shape == (len(self), 3)), "provided translations have wrong shape"
        self._gs_array[:, 0:3] = translations

    @property
    def rotation_matrices(self):
        """
        :return: (n, 3, 3) np array with rotation matrices
        """
        return self._gs_array[:, 3:12].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices):
        """
        :param rotation_matrices:  (n, 3, 3) np array with rotation matrices
        """
        assert(rotation_matrices.shape == (len(self), 3, 3)), "provided rotation matrices have wrong shape"
        self._gs_array[:, 3:12] = rotation_matrices.reshape((-1, 9))

    @property
    def poses(self):
        """
        :return: (n, 4, 4) np array with poses (homogenous tf matrices)
        """
        poses = np.zeros((len(self), 4, 4), dtype=np.float32)
        poses[:, 3, 3] = 1
        poses[:, 0:3, 3] = self.translations
        poses[:, 0:3, 0:3] = self.rotation_matrices
        return poses

    @poses.setter
    def poses(self, poses):
        """
        :param poses: (n, 4, 4) np array with poses (homogenous tf matrices)
        """
        assert(poses.shape == (len(self), 4, 4)), "provided poses have wrong shape"
        self._gs_array[:, 0:3] = poses[:, 0:3, 3]
        self._gs_array[:, 3:12] = poses[:, 0:3, 0:3].reshape((-1, 9))
