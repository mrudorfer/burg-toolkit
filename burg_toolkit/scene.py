import os
from collections import UserDict
import copy

import numpy as np
import open3d as o3d

from . import mesh_processing


class ObjectType:
    """
    Describes an Object Type.

    :param identifier: object identifier as string
    :param mesh: open3d.geometry.TriangleMesh associated with the object
    :param mass: mass of object in kg (defaults to 0, which means immovable, used for background objects)
    :param friction_coeff: friction coefficient, defaults to 0.24
    """
    def __init__(self, identifier=None, mesh=None, mass=None, friction_coeff=None, restitution_coeff=None):
        self.identifier = identifier or ''
        self.mesh = mesh
        self.mass = mass or 0
        self.friction_coeff = friction_coeff or 0.24
        self.restitution_coeff = restitution_coeff or 0.1
        self.urdf_fn = None

    def __str__(self):
        return f'ObjectType: {self.identifier}\n\thas mesh: {self.mesh is not None}\n\tmass: {self.mass}\n\t' + \
            f'friction: {self.friction_coeff}\n\trestitution: {self.restitution_coeff}\n\turdf: {self.urdf_fn}'

    def make_urdf_file(self, directory, overwrite_existing=False, default_inertia=None, mass_factor=1.0):
        """
        This method will produce a temporary .obj file which stores the current status of the mesh, so we don't have
        to bother about scaling or displacement applied during loading the mesh.
        Based on that, we will create a urdf file in the given directory.
        The directory will then contain an `object_name.obj` and an `object_name.urdf` file.

        :param directory: string with directory in which the files shall be stored.
        :param overwrite_existing: bool that indicates whether to overwrite existing files or not.
        :param default_inertia: (3, 3) ndarray, provide an inertia matrix to override the mesh's actual properties
        :param mass_factor: actual mass of object will be multiplied by this factor (note that original mass will be
                            used to compute the inertia)
        """
        mesh_fn = self.identifier + '.obj'
        mesh_path = os.path.join(directory, mesh_fn)
        self.urdf_fn = os.path.join(directory, self.identifier + '.urdf')

        # handle existing files
        if (os.path.exists(mesh_path) or os.path.exists(self.urdf_fn)) and not overwrite_existing:
            print(f'make_urdf_file(): file already exists and will not be re-created for object: {self.identifier}')
            return

        # make sure directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # cannot write vertex normals in obj file, but we don't really need them in urdf
        # this does produce warnings that it can't write triangle normals to obj file. don't know how to suppress.
        o3d.io.write_triangle_mesh(mesh_path, self.mesh, write_vertex_normals=False)

        if default_inertia is not None:
            inertia = default_inertia
            com = self.mesh.get_center()
        else:
            inertia, com = mesh_processing.compute_mesh_inertia(self.mesh, self.mass)

        origin = [0, 0, 0]  # meshes are already saved as is, so we have no displacement
        with open(self.urdf_fn, 'w') as urdf:
            urdf.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
            urdf.write(f'<robot name="{self.identifier}">\n')
            urdf.write(f'\t<link name="base">\n')

            # collision
            urdf.write(f'\t\t<collision>\n')
            urdf.write(f'\t\t\t<geometry>\n')
            urdf.write(f'\t\t\t\t<mesh filename="{mesh_fn}"/>\n')
            urdf.write(f'\t\t\t</geometry>\n')
            urdf.write(f'\t\t\t<origin xyz="{" ".join(map(str, origin))}"/>\n')
            urdf.write(f'\t\t</collision>\n')

            # visual
            urdf.write(f'\t\t<visual>\n')
            urdf.write(f'\t\t\t<geometry>\n')
            urdf.write(f'\t\t\t\t<mesh filename="{mesh_fn}"/>\n')
            urdf.write(f'\t\t\t</geometry>\n')
            urdf.write(f'\t\t\t<origin xyz="{" ".join(map(str, origin))}"/>\n')
            urdf.write(f'\t\t</visual>\n')

            # physics
            urdf.write(f'\t\t<inertial>\n')
            urdf.write(f'\t\t\t<mass value="{self.mass * mass_factor}"/>\n')
            urdf.write(f'\t\t\t<inertia ixx="{inertia[0, 0]}" ixy="{inertia[0, 1]}" ixz="{inertia[0, 2]}"' +
                       f' iyy="{inertia[1, 1]}" iyz="{inertia[1, 2]}" izz="{inertia[2, 2]}" />\n')
            urdf.write(f'\t\t\t<origin xyz="{" ".join(map(str, com))}"/>\n')
            urdf.write(f'\t\t</inertial>\n')

            urdf.write(f'\t</link>\n')
            urdf.write(f'</robot>')


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

    def __str__(self):
        return f'instance of {self.object_type.identifier} object type. pose:\n{self.pose}'

    def get_mesh(self):
        """
        Returns a copy of the mesh of the object type in the pose of the instance.

        :return: open3d.geometry.TriangleMesh
        """
        if self.object_type.mesh is None:
            raise ValueError('no mesh associated with this object type')
        mesh = copy.deepcopy(self.object_type.mesh)
        mesh.transform(self.pose)
        return mesh


class ObjectLibrary(UserDict):
    """
    Contains a library of ObjectType objects and adds some convenience methods to it.
    Acts like a regular python dict.
    """

    def yell(self):
        print('OH MY GOSH I AM AN OBJECT LIBRARY!!!! Look at all my objects:')
        print([key for key in self.data.keys()])

    def generate_urdf_files(self, directory, overwrite_existing=False):
        for name, obj in self.data.items():
            obj.make_urdf_file(directory, overwrite_existing=overwrite_existing)

    def __len__(self):
        return len(self.data.keys())


class Camera:
    """
    holds intrinsic and extrinsic parameters, initialises with some Kinect-like intrinsics.
    """
    def __init__(self):
        self.resolution = [640, 480]  # w x h
        self.intrinsic_parameters = {
            'fx': 572.41140,
            'fy': 573.57043,
            'cx': 325.26110,
            'cy': 242.04899
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
