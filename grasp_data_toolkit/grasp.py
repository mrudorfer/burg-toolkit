import numpy as np
from scipy import spatial
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
        # compute using module function
        # could also use quaternion computation:
        # rotation_dist = 2*np.arccos(np.abs(np.dot(q1, q2)))/np.pi*180

        return pairwise_distances(self, other_grasp)[0, 0]


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


def pairwise_distances(graspset1, graspset2, print_timings=False):
    """
    computes the pairwisse distances between the provided grasps from set1 and set2
    this is a vectorized implementation, but should not be used if both sets are extremely large
    (i.e. it is not suitable for nearest-neighbor search in coverage computation)
    :param graspset1: GraspSet of length N (or single Grasp)
    :param graspset2: GraspSet of length M (or single Grasp)
    :param print_timings: whether or not to print the computation time
    :return: (N, M) matrix of distances (1 = 1mm or 1deg)
    """
    # convert provided arguments to GraspSets if necessary
    if type(graspset1) is Grasp:
        graspset1 = GraspSet(graspset1.internal_array[np.newaxis, :])
    if type(graspset2) is Grasp:
        graspset2 = GraspSet(graspset2.internal_array[np.newaxis, :])

    if (type(graspset1) is not GraspSet) or (type(graspset2) is not GraspSet):
        raise TypeError('Arguments have wrong type. Expected Grasp or GraspSet.')

    if print_timings:
        from timeit import default_timer as timer
        t1 = timer()

    # let's compute translation distance first
    # shape: (N, M, 3) = (N, 2, 3) - (1, M, 3)
    distances = np.linalg.norm(
        graspset1.translations[:, np.newaxis, :] - graspset2.translations[np.newaxis, :, :],
        axis=-1) * 1000  # also scale so that 1 = 1mm

    if print_timings:
        t2 = timer()
        print('TIME: computing distances took', t2-t1, 'seconds')

    # see http://boris-belousov.net/2016/12/01/quat-dist/ for explanation of basic formula
    # basic formula: arccos((tr(R1*R2.T)-1)/2)
    # we just have to vectorize it to compute pairwise distances
    distances += np.rad2deg(
        np.arccos(
            (
                np.trace(
                  np.matmul(
                      # extend the dimensions to (N, 1, 3, 3) and (1, M, 3, 3) for the pairwise computations
                      graspset1.rotation_matrices[:, np.newaxis, ...],
                      np.transpose(graspset2.rotation_matrices, axes=(0, 2, 1))[np.newaxis, ...]
                  ),
                  axis1=-2, axis2=-1  # compute trace over last two dims
                ) - 1.0
            ) / 2.0
        )
    )

    if print_timings:
        t3 = timer()
        print('TIME: computing angles took', t3-t2, 'seconds')

    return distances


def coverage_brute_force(reference_grasp_set, query_grasp_set, epsilon=15.0):
    """
    computes coverage, i.e. fraction of grasps from the reference grasp set which are covered by a grasp of query set
    corresponds to coverage_1 from Eppner et al., 2019
    this brute force variant computes the full distance matrix and then checks for the threshold. the vectorized
    implementation is relatively fast but it quickly runs into memory limitations when sets become larger
    :param reference_grasp_set: the grasp set to be covered
    :param query_grasp_set: the grasp set which shall cover the reference grasp set
    :param epsilon: the tolerance threshold used in the distance function - a grasp from the reference set will be
    considered as covered, if its distance to the closest query grasp does not exceed epsilon
    :return: float in [0, 1] corresponding to the fraction of grasps in the reference grasp set which are covered
    by grasps from the query grasp set
    """
    cov = np.any(pairwise_distances(reference_grasp_set, query_grasp_set) <= epsilon, axis=1)
    return np.count_nonzero(cov) / len(cov)


def coverage(reference_grasp_set: GraspSet, query_grasp_set: GraspSet, epsilon=15.0, print_timings=False):
    """
    computes coverage, i.e. fraction of grasps from the reference grasp set which are covered by a grasp of query set
    corresponds to coverage_1 from Eppner et al., 2019
    this implementation uses kd-trees and can thus handle larger grasp sets, although it may take some time.
    :param reference_grasp_set: the grasp set to be covered
    :param query_grasp_set: the grasp set which shall cover the reference grasp set
    :param epsilon: the tolerance threshold used in the distance function - a grasp from the reference set will be
    considered as covered, if its distance to the closest query grasp does not exceed epsilon
    :param print_timings: does print timing information if set to True
    :return: float in [0, 1] corresponding to the fraction of grasps in the reference grasp set which are covered
    by grasps from the query grasp set
    """
    # so we have to consider that both grasp sets can be very large, computing a brute-force distance matrix
    # can quickly exceed the available RAM
    # Eppner et al. used an algorithm by Ichnowski et al. which performs nearest neighbor search in SE(3) with
    # quaternions.. see the paper here: http://dx.doi.org/10.1007/978-3-319-16595-0_12
    # however, i could not find any code available for this. so i went with scipy's kd tree methods

    # variant 1:
    # 1) construct kdtree for query grasp set translations
    # 2) query ball point with reference grasp translations --> this step requires 99.99% of the time
    # 3) go through resulting array and check actual distances

    # variant 2:
    # 1) construct kdtree for query grasp set translations
    # 1a) construct kdtree for ref grasp set translations
    # 2) query ball tree
    # 3) go through resulting list and check actual distances

    # i tested both variants with 2.6M ref grasp set and 500 query grasp set
    # var 1 took 380-450 seconds, with 2) being the most time-consuming step
    # var 2 took 270 seconds (only one run), with 1a) being most time-consuming step and greatly accelerating 2)
    # so i stick with variant 2

    if print_timings:
        from timeit import default_timer as timer
        t1 = timer()

    ref_tree = spatial.KDTree(reference_grasp_set.translations)
    query_tree = spatial.KDTree(query_grasp_set.translations)

    if print_timings:
        t2 = timer()

    list_of_indices = ref_tree.query_ball_tree(query_tree, r=epsilon/1000)

    if print_timings:
        t3 = timer()

    # although we have to loop through the whole list, this step is fairly quick because each iteration only
    # considers very few candidates
    covered = 0
    for i, indices in enumerate(list_of_indices):
        if len(indices) > 0:
            if np.any(pairwise_distances(reference_grasp_set[i], query_grasp_set[indices]) <= epsilon):
                covered += 1

    if print_timings:
        t4 = timer()
        print('it took:')
        print(t2-t1, 'seconds to construct the trees')
        print(t3-t2, 'seconds to query the tree and get the array of indices based on translations')
        print(t4-t3, 'seconds to loop through the remaining candidates')

    return covered / len(reference_grasp_set)
