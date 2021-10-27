from collections import UserDict
import copy

import numpy as np

from . import io


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

    def __str__(self):
        return f'ObjectType: {self.identifier}\n\thas mesh: {self.mesh is not None}\n\tmass: {self.mass}\n\t' + \
            f'friction: {self.friction_coeff}\n\trestitution: {self.restitution_coeff}'


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
            io.save_mesh_and_urdf(obj, directory, overwrite_existing=overwrite_existing)

    def __len__(self):
        return len(self.data.keys())


class Scene:
    """
    contains all information about a scene
    """

    def __init__(self, objects=None, bg_objects=None, views=None):
        self.objects = objects or []
        self.bg_objects = bg_objects or []
        self.views = views or []
