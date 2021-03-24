import numpy as np
import open3d as o3d


class ObjectType:
    """
    Describes an Object Type.

    :param identifier: object identifier as string
    :param mesh: open3d.geometry.TriangleMesh associated with the object
    :param mass: mass of object in kg (defaults to 0, which means immovable, used for background objects)
    :param friction_coeff: friction coefficient, defaults to 0.24
    """
    def __init__(self, identifier=None, mesh=None, mass=None, friction_coeff=None):
        self.identifier = identifier or ''
        self.mesh = mesh
        self.mass = mass or 0
        self.friction_coeff = friction_coeff or 0.24


class ObjectInstance:
    """
    Describes an instance of an object type in the object library and a pose.

    :param object_type: an ObjectType referring to the type of this object instance
    :param pose: (4, 4) np array - homogenous transformation matrix
    """

    def __init__(self, object_type, pose=None):
        self.object_type = object_type
        if pose is None:
            self.pose = np.eye(4)
        else:
            self.pose = pose


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

    def __init__(self, camera_intrinsics=None, camera_pose=None, depth_image=None, rgb_image=None,
                 class_label_image=None, instance_label_image=None):
        self.camera_intrinsics = camera_intrinsics
        self.camera_pose = camera_pose
        self.depth_image = depth_image
        self.rgb_image = rgb_image
        self.class_label_image = class_label_image
        self.instance_label_image = instance_label_image

    def to_point_cloud(self, stride=2):
        """
        creates a partial point cloud from the depth image considering intrinsic/extrinsic parameters

        :param stride: the stride with which pixels will be converted to points, use 1 for dense conversion (default 2)

        :return: an o3d point cloud
        """

        # create point cloud from depth
        pc = o3d.geometry.PointCloud.create_from_depth_image(
            depth=self.depth_image,
            intrinsic=self.camera_intrinsics,
            extrinsic=self.camera_pose,
            depth_scale=1.0,
            depth_trunc=1.0,
            stride=stride,
            project_valid_depth_only=True
        )

        return pc


class Scene:
    """
    contains all information about a scene
    """

    def __init__(self, objects=None, bg_objects=None, views=None):
        self.objects = objects or []
        self.bg_objects = bg_objects or []
        self.views = views or []
