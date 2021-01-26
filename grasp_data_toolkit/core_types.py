import numpy as np
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

    def __init__(self, name, pose):
        self.name = name or ""
        self.pose = pose or np.eye(4)
        self.point_cloud = np.asarray([])

    def __init__(self, name, translation, rotation):
        self.name = name or ""
        pose = np.eye(4)
        pose[0:3, 0:3] = rotation
        pose[0:3, 3] = translation
        self.pose = pose
        self.point_cloud = np.asarray([])


class CameraView:
    """
    all images from one camera view, including intrinsics, extrinsics and resolution
    """

    def __init__(self, img_data):
        """
        creates a CameraView object from one given img_data dictionary (as read from MATLAB file)
        :param img_data: one instance of imageData from the .mat file
        """
        self.cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.cam_intrinsics.set_intrinsics(
            width=int(img_data['cameraResolution'][0]),
            height=int(img_data['cameraResolution'][1]),
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
        self.cam_pose = tf
        self.cam_resolution = img_data['cameraResolution']
        self.rgb_image = img_data['heapRGBImage']
        self.depth_image = img_data['heapDepthImage']
        self.class_label_image = img_data['heapClassLabelImage']
        self.instance_label_image = img_data['heapInstanceLabelImage']
        self.scene_coordinate_image = img_data['heapSceneCoordinateImage']


class Scene:
    """
    contains all information about a scene
    """

    def __init__(self, objects=None, bg_objects=None, views=None):
        self.objects = objects or []
        self.views = views or []
        self.bg_objects = bg_objects or []
