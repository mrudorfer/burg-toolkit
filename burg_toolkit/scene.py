import os
from collections import UserDict
import copy

import numpy as np
import open3d as o3d

from . import mesh_processing
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
        raise DeprecationWarning('This method is obsolete and will be removed.'
                                 'Please use burg.io.save_mesh_and_urdf() instead.')

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
