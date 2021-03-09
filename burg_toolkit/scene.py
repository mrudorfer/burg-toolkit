import numpy as np
import open3d as o3d


class ObjectType:
    """
    Describes an Object Type (i.e. the model).

    Initialises the object type from a objectInformation dict parsed from a .mat file

    :param obj_info: one entry of the dict parsed from the render_data.mat file
    :param displacement: additional info about translation displacement of object centre
    """

    def __init__(self, obj_info, displacement=None, point_cloud=None):

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
    Describes an instance of an object (in a scene).

    Initialises an object instance in the scene, has library index and a pose.

    :param obj_instance: dict from heap, describing one object instance
    """

    def __init__(self, obj_instance):
        self.library_index = obj_instance['objectLibraryIndex']
        translation = obj_instance['translationVector']
        rotation = obj_instance['rotationMatrix']
        tf = np.eye(4)
        tf[0:3, 0:3] = rotation
        tf[0:3, 3] = translation
        self.pose = tf


class BackgroundObject:
    """
    Describes the table, or other possible background objects (is both instance and type, so may want to change that
    in the future).
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
        """
        self.pose = camera_pose


class CameraView:
    """
    All images from one camera view, including camera parameters.

    Creates a CameraView object from one given img_data dictionary (as read from MATLAB file).

    :param img_data: one instance of imageData from the .mat file
    """

    def __init__(self, img_data):
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
