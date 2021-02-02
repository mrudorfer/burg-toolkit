import numpy as np
import quaternion
import open3d as o3d


class ObjectType:
    """
    describes an object type
    """

    def __init__(self, obj_info, displacement=None, point_cloud=None):
        """
        initialises the object type from a objectInformation dict parsed from a .mat file
        :param obj_info: one entry of the dict parsed from the render_data.mat file
        :param displacement: additional info about translation displacement of object centre
        """
        self.name = obj_info['name']
        self.mass = obj_info['mass']
        self.friction_coeff = obj_info['coefficientOfFriction']

        if displacement is None:
            displacement = [0, 0, 0]
        self.displacement = np.asarray(displacement)

        if point_cloud is None:
            self.point_cloud = []
        else:
            self.point_cloud = point_cloud

        # this data is also available in the dict, we can add it if needed
        # coefficientOfRestitution
        # shadingType
        # diffuseReflectionConstant
        # specularReflectionConstant
        # shininessConstant
        # ambientReflectionConstant


class ObjectInstance:
    """
    describes one instance of an object in a scene
    """

    def __init__(self, obj_instance):
        """
        initialises an object instance in the scene, has library index and a pose
        :param obj_instance: dict from heap, describing one object instance
        """
        self.library_index = obj_instance['objectLibraryIndex']
        translation = obj_instance['translationVector']
        rotation = obj_instance['rotationMatrix']
        tf = np.eye(4)
        tf[0:3, 0:3] = rotation
        tf[0:3, 3] = translation
        self.pose = tf


class BackgroundObject:
    """
    describes the table, or other possible background objects
    """

    def __init__(self, name="", pose=None):
        self.name = name
        self.pose = pose if pose is not None else np.eye(4)
        self.point_cloud = np.asarray([])

    @classmethod
    def from_translation_rotation(cls, name, translation, rotation):
        """
        create instance of BackgroundObject from translation and rotation
        :param name: string with object name
        :param translation: [x, y, z]
        :param rotation: 3x3 rotation matrix
        :return: instance of BackgroundObject with name and corresponding pose
        """
        pose = np.eye(4)
        pose[0:3, 0:3] = rotation
        pose[0:3, 3] = translation
        return cls(name, pose)


class Camera:
    """
    holds intrinsic and extrinsic parameters
    """

    def __init__(self):
        self.resolution = [0, 0]  # w x h
        self.intrinsic_parameters = {
            'fx': 0.0,
            'fy': 0.0,
            'cx': 0.0,
            'cy': 0.0
        }
        self.pose = np.eye(4)

    def set_resolution(self, width: int, height: int):
        self.resolution = [width, height]

    def set_intrinsic_parameters(self, fx=None, fy=None, cx=None, cy=None):
        """
        overwrites only the given parameters, the others stay the same
        :param fx: focal length x
        :param fy: focal length y
        :param cx: principal point x
        :param cy: principal point y
        :return: nothing
        """
        if fx is not None:
            self.intrinsic_parameters['fx'] = fx
        if fy is not None:
            self.intrinsic_parameters['fy'] = fy
        if cx is not None:
            self.intrinsic_parameters['cx'] = cx
        if cy is not None:
            self.intrinsic_parameters['cy'] = cy

    def get_o3d_intrinsics(self):
        """
        :return: intrinsic parameters (incl. resolution) as instance of o3d.camera.PinholeCameraIntrinsic()
        """
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.resolution[0]),
            height=int(self.resolution[1]),
            fx=self.intrinsic_parameters['fx'],
            fy=self.intrinsic_parameters['fy'],
            cx=self.intrinsic_parameters['cx'],
            cy=self.intrinsic_parameters['cy']
        )
        return o3d_intrinsics

    def set_extrinsic_parameters(self, camera_pose):
        """
        sets the pose of the camera
        :param camera_pose: np 4x4 homogenous tf matrix
        :return:
        """
        self.pose = camera_pose


class CameraView:
    """
    all images from one camera view, including intrinsics, extrinsics and resolution
    """

    def __init__(self, img_data):
        """
        creates a CameraView object from one given img_data dictionary (as read from MATLAB file)
        :param img_data: one instance of imageData from the .mat file
        """
        self.camera = Camera()
        self.camera.set_resolution(img_data['cameraResolution'][0], img_data['cameraResolution'][1])
        self.camera.set_intrinsic_parameters(
            fx=float(img_data['cameraIntrinsicParameters']['focalLengthValue'][0]),
            fy=float(img_data['cameraIntrinsicParameters']['focalLengthValue'][1]),
            cx=float(img_data['cameraIntrinsicParameters']['principalPointValue'][0]),
            cy=float(img_data['cameraIntrinsicParameters']['principalPointValue'][1])
        )

        translation = img_data['cameraExtrinsicParameters']['translationVectorValue']
        rotation = img_data['cameraExtrinsicParameters']['rotationMatrix']
        tf = np.eye(4)
        tf[0:3, 0:3] = rotation
        tf[0:3, 3] = translation
        self.camera.set_extrinsic_parameters(tf)
        self.rgb_image = img_data['heapRGBImage']
        self.depth_image = img_data['heapDepthImage']
        self.class_label_image = img_data['heapClassLabelImage']
        self.instance_label_image = img_data['heapInstanceLabelImage']


class Scene:
    """
    contains all information about a scene
    """

    def __init__(self, objects=None, bg_objects=None, views=None):
        self.objects = objects or []
        self.views = views or []
        self.bg_objects = bg_objects or []


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
        self._grasp_array[3:12] = rotation_matrix[:].astype(np.float32)

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
