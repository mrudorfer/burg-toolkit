import copy
import glob
import os
import pathlib

import numpy as np
import mat73
import scipy.io as spio
import h5py
import open3d as o3d

from . import util
from . import scene
from . import grasp
from . import mesh_processing


def load_mesh(mesh_fn, texture_fn=None):
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    if texture_fn is not None:
        mesh.textures = [o3d.io.read_image(texture_fn)]

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def save_mesh(fn, mesh_obj, overwrite_existing=True):
    """
    Saves a mesh to file.

    :param fn: string, filename
    :param mesh_obj: one of o3d.geometry.TriangleMesh, burg.scene.ObjectType, burg.scene.ObjectInstance, in case of the
                     latter, the mesh will be saved in the transformed pose
    :param overwrite_existing: bool, indicating whether to overwrite an existing file (defaults to True)
    """
    if not overwrite_existing and os.path.exists(fn):
        return

    if isinstance(mesh_obj, o3d.geometry.TriangleMesh):
        mesh = mesh_obj
    elif isinstance(mesh_obj, scene.ObjectType):
        mesh = mesh_obj.mesh
    elif isinstance(mesh_obj, scene.ObjectInstance):
        mesh = copy.deepcopy(mesh_obj.object_type.mesh)
        mesh.transform(mesh_obj.pose)
    else:
        return ValueError('unrecognised mesh_obj type, must be one of:' +
                          'o3d.geometry.TriangleMesh, burg.scene.ObjectType, burg.scene.ObjectInstance')

    # this does produce warnings that it can't write triangle normals to obj file. don't know how to suppress.
    o3d.io.write_triangle_mesh(fn, mesh, write_vertex_normals=False)


def save_urdf(fn, mesh_fn, name, origin=None, inertia=None, com=None, mass=0, overwrite_existing=True):
    """
    Creates a urdf file with given parameters.

    :param fn: filename of the urdf file
    :param mesh_fn: filename of the mesh to be referenced as visual and collision model
    :param name: name of the object/robot as string
    :param origin: [x,y,z] of mesh origin, if None then [0, 0, 0] will be used
    :param inertia: (3, 3) ndarray, an inertia matrix of the object, defaults to np.eye(3)*0.001
    :param com: [x,y,z] center of mass of mesh, if None then [0, 0, 0] will be used
    :param mass: float, mass of object (default 0, which means object is fixed in space)
    :param overwrite_existing: bool, indicating whether to overwrite existing file (defaults to True)
    """
    if not overwrite_existing and os.path.exists(fn):
        return

    if origin is None:
        origin = [0, 0, 0]
    if inertia is None:
        inertia = np.eye(3) * 0.001
    if com is None:
        com = [0, 0, 0]

    with open(fn, 'w') as urdf:
        urdf.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
        urdf.write(f'<robot name="{name}">\n')
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
        urdf.write(f'\t\t\t<mass value="{mass}"/>\n')
        urdf.write(f'\t\t\t<inertia ixx="{inertia[0, 0]}" ixy="{inertia[0, 1]}" ixz="{inertia[0, 2]}"' +
                   f' iyy="{inertia[1, 1]}" iyz="{inertia[1, 2]}" izz="{inertia[2, 2]}" />\n')
        urdf.write(f'\t\t\t<origin xyz="{" ".join(map(str, com))}"/>\n')
        urdf.write(f'\t\t</inertial>\n')

        urdf.write(f'\t</link>\n')
        urdf.write(f'</robot>')


def save_mesh_and_urdf(mesh_obj, directory, name=None, default_inertia=None, mass_factor=1.0, overwrite_existing=True):
    """
    This method will produce an .obj file which stores the current status of the mesh. It will create a urdf file
    in the same directory, which references the .obj file.
    The directory will then contain an `name.obj` and an `name.urdf` file.
    All physics properties are populated as best as possible with the given data.

    :param mesh_obj: can be o3d.geometry.TriangleMesh, burg.scene.ObjectType or burg.scene.ObjectInstance, in the case
                     of the former, some physical properties cannot be determined, in case of the latter two, the
                     properties are taken from the ObjectType info
    :param directory: string with directory in which the files shall be stored.
    :param name: string, name of the object, is required if a pure mesh is provided, optional if burg ObjectType or
                 ObjectInstance, in that case it defaults to the ObjectType.identifier
    :param default_inertia: (3, 3) ndarray, provide an inertia matrix to override the mesh's actual properties
    :param mass_factor: actual mass of object will be multiplied by this factor (note that original mass will be
                        used to compute the inertia)
    :param overwrite_existing: bool that indicates whether to overwrite existing files, defaults to True.
    """
    # check the different types and populate some info about mass and name
    if isinstance(mesh_obj, o3d.geometry.TriangleMesh):
        mesh = mesh_obj
        mass = 0
        if name is None:
            raise ValueError('name is a mandatory parameter if providing an o3d.geometry.TriangleMesh')
    elif isinstance(mesh_obj, scene.ObjectType):
        mesh = mesh_obj.mesh
        mass = mesh_obj.mass * mass_factor
        if name is None:
            name = mesh_obj.identifier
    elif isinstance(mesh_obj, scene.ObjectInstance):
        mesh = copy.deepcopy(mesh_obj.object_type.mesh)
        mesh.transform(mesh_obj.pose)
        mass = mesh_obj.object_type.mass * mass_factor
        if name is None:
            name = mesh_obj.object_type.identifier
    else:
        return ValueError('unrecognised mesh_obj type, must be one of:' +
                          'o3d.geometry.TriangleMesh, burg.scene.ObjectType, burg.scene.ObjectInstance')

    make_sure_directory_exists(directory)
    mesh_fn = name + '.obj'
    mesh_path = os.path.join(directory, mesh_fn)
    urdf_fn = os.path.join(directory, name + '.urdf')

    save_mesh(mesh_path, mesh, overwrite_existing)

    if default_inertia is not None:
        inertia = default_inertia
        com = mesh.get_center()
    else:
        try:
            inertia, com = mesh_processing.compute_mesh_inertia(mesh, mass)
        except ValueError:
            # if mesh is not watertight we cannot compute inertia
            inertia = None
            com = mesh.get_center()

    origin = [0, 0, 0]

    save_urdf(urdf_fn, mesh_fn, name, origin, inertia, com, mass, overwrite_existing)


class BaseviMatlabScenesReader:
    """
    Reader for files that are related to the MATLAB scene generation pipeline of Basevi (unpublished).

    :param path_config: the part of the config file related to MATLAB scene generation, containing all relevant
                        paths to read required information
    """
    def __init__(self, path_config):
        self.object_library_fn = path_config['object_lib_fn']
        self.obj_models_dir = path_config['models_dir']
        self.mesh_fn_ext = path_config['mesh_fn_ext']
        self.table_fn = os.path.join(path_config['bg_models_dir'], path_config['table_fn'])
        self.scenes_dir = path_config['scenes_dir']
        self.table_id = 'table'
        self.table_scale_factor = float(path_config['table_scale_factor'])
        self.library_index_to_name = {}
        self.object_library = scene.ObjectLibrary()

    def get_scene_filenames(self, directory=None):
        """finds heap and image data files in the given directory

        :param directory: Directory containing the files. If none given, the `scenes_dir` path from the config file
                          will be used.

        :return: list of dicts with 'heap_fn' and 'image_data_fn' keys which both include the full path
        """
        if directory is None:
            directory = self.scenes_dir

        heap_files = glob.glob(directory + "Data*ObjectHeap*.mat")
        image_files = glob.glob(directory + "Images*ObjectHeap*.mat")
        if len(heap_files) != len(image_files):
            print("warning: different number of heap and image files found")
            return {}

        # todo: we could try to make sure the file names match...
        filenames = []
        for heap_fn, image_fn in zip(heap_files, image_files):
            filenames.append({
                'heap_fn': heap_fn,
                'image_data_fn': image_fn
            })

        return filenames

    @staticmethod
    def read_view(view_dict):
        """converts a view into a scene.CameraView object

        :param view_dict: an element in image_data_mat['imageData']

        :return: scene.CameraView"""
        # get camera info
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(view_dict['cameraResolution'][0]),
            height=int(view_dict['cameraResolution'][1]),
            fx=float(view_dict['cameraIntrinsicParameters']['focalLengthValue'][0]),
            fy=float(view_dict['cameraIntrinsicParameters']['focalLengthValue'][1]),
            cx=float(view_dict['cameraIntrinsicParameters']['principalPointValue'][0]),
            cy=float(view_dict['cameraIntrinsicParameters']['principalPointValue'][1])
        )

        tf = np.eye(4)
        tf[0:3, 0:3] = view_dict['cameraExtrinsicParameters']['rotationMatrix']
        tf[0:3, 3] = view_dict['cameraExtrinsicParameters']['translationVectorValue']

        rgb_image = util.o3d_image_from_numpy(view_dict['heapRGBImage'])

        # o3d can't handle the inf values, so set them to zero
        depth_image = util.o3d_image_from_numpy(view_dict['heapDepthImage'], replace_inf_by=0)

        class_label_image = util.o3d_image_from_numpy(view_dict['heapClassLabelImage'])
        instance_label_image = util.o3d_image_from_numpy(view_dict['heapInstanceLabelImage'])

        view = scene.CameraView(camera_intrinsics=o3d_intrinsics, camera_pose=tf, depth_image=depth_image,
                                rgb_image=rgb_image, class_label_image=class_label_image,
                                instance_label_image=instance_label_image)
        return view

    def read_scene_files(self, filenames):
        """reads scene data

        :param filenames: dict with 'heap_fn' and 'image_data_fn' (which include full path). E.g. an element of the list
                          that has been output by `get_scene_filenames()`

        :return: scene.Scene object
        """
        heap_mat = spio.loadmat(filenames['heap_fn'], simplify_cells=True)
        # image_data is a v7.3 mat stored in hdf5 format, thus needs different reader
        image_data_mat = mat73.loadmat(filenames['image_data_fn'])

        object_instances = []
        for obj in heap_mat['heap']:
            # library index must be mapped to name
            index = obj['objectLibraryIndex']
            if index not in self.library_index_to_name.keys():
                raise ValueError(f'object library index {index} not found in object library')
            identifier = self.library_index_to_name[index]
            tf = np.eye(4)
            tf[0:3, 0:3] = obj['rotationMatrix']
            tf[0:3, 3] = obj['translationVector']
            object_instance = scene.ObjectInstance(self.object_library[identifier], tf)
            object_instances.append(object_instance)

        table_pose = np.eye(4)
        table_pose[0:3, 0:3] = heap_mat['backgroundInformation']['tableBasis']
        table_pose[0:3, 3] = heap_mat['backgroundInformation']['tableCentre']
        table_instance = scene.ObjectInstance(self.object_library[self.table_id], table_pose)
        bg_objects = [table_instance]

        views = [self.read_view(v) for v in image_data_mat['imageData']]

        return scene.Scene(object_instances, bg_objects, views)

    def read_object_library(self):
        """
        Reads the object info from the render_data.mat file, using the paths provided to constructor.
        The object library will be stored internally, but also returned.
        It already reads the provided meshes as well, but does not create point clouds from those.

        :return: (object_library, library_index_to_name)
                 object_library is a dictionary with object identifiers as keys and scene.ObjectType as values.
                 library_index_to_name maps the numeric library index to the object identifier.

        """

        input_dict = spio.loadmat(self.object_library_fn, simplify_cells=True)
        self.object_library = scene.ObjectLibrary()
        self.library_index_to_name = {}

        for i, (displacement, obj_dict) in enumerate(zip(input_dict['objectCentres'], input_dict['objectInformation'])):
            name = obj_dict['name']
            mass = obj_dict['mass']
            friction_coeff = obj_dict['coefficientOfFriction']
            resitution_coeff = obj_dict['coefficientOfRestitution']

            mesh_fn = os.path.join(self.obj_models_dir, name + self.mesh_fn_ext)
            mesh = load_mesh(mesh_fn)  # todo: could also load textures here
            mesh.translate(-np.asarray(displacement))

            obj = scene.ObjectType(identifier=name, mesh=mesh, mass=mass, friction_coeff=friction_coeff,
                                   restitution_coeff=resitution_coeff)
            self.object_library[name] = obj

            self.library_index_to_name[i+1] = name

        # also add the table to the object library
        table_mesh = load_mesh(self.table_fn)
        table_mesh.scale(self.table_scale_factor, np.array([0, 0, 0]))
        self.object_library[self.table_id] = scene.ObjectType(identifier=self.table_id, mesh=table_mesh, mass=None)

        return self.object_library, self.library_index_to_name


def read_grasp_file_eppner2019(grasp_fn):
    """
    Reads grasps from the grasp file of dataset provided with publication of Eppner et al. 2019.

    It should contain densely sampled, successful grasps (verified in their simulation).

    :param grasp_fn: the filename

    :return: a core_types.GraspSet, and center of mass (as np array with length 3)
    """
    print('read_grasp_file_eppner2019')
    hf = h5py.File(grasp_fn, 'r')
    print('just keys:', list(hf.keys()))

    print_keys = ['gripper', 'object', 'object_class', 'object_dataset']
    for key in print_keys:
        print(key, hf[key][()].decode())

    print('object_scale:', hf['object_scale'][()])
    print('poses.shape:', hf['poses'].shape)
    print('com', hf['object_com'][:])

    print('creating grasp set...')
    gs = grasp.GraspSet.from_translations_and_quaternions(translations=hf['poses'][:, 0:3],
                                                          quaternions=hf['poses'][:, 3:7])
    print('done')

    return gs, hf['object_com'][:]


def make_sure_directory_exists(directory):
    """
    Creates directory including all the parent directories if they do not already exist.

    :param directory: string with path, or list of paths
    """
    if not isinstance(directory, list):
        directory = [directory]

    for path in directory:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
