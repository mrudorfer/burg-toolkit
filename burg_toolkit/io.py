import glob
import os
import pathlib
import logging
from abc import ABC, abstractmethod

import numpy as np
import yaml
import mat73
import scipy.io as spio
import h5py
import open3d as o3d
import imageio
try:
    import pyexr
except ImportError:
    pyexr = None
    logging.info('Could not import pyexr, loading burg toolkit without OpenEXR support.')

from . import core
from . import util
from . import grasp


class YAMLFileTypeException(Exception):
    pass


class YAMLObject(ABC):
    YAML_FILE_TYPE = 'yaml_file_type'
    YAML_FILE_VERSION = 'yaml_file_version'

    @classmethod
    @abstractmethod
    def yaml_version(cls):
        """ must be implemented to instantiate - provides a version number for the files """
        pass

    @abstractmethod
    def to_yaml(self, yaml_fn):
        """ push the object to a yaml file """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, yaml_fn):
        """ create an object instance based on a yaml file """
        pass

    @classmethod
    def get_yaml_data(cls, yaml_fn):
        with open(yaml_fn, 'r') as stream:
            data = yaml.safe_load(stream)

        # check correct file type
        if cls.YAML_FILE_TYPE not in data.keys():
            raise YAMLFileTypeException(f'YAML file must have a {cls.YAML_FILE_TYPE} field. '
                                        f'Your file may be too old if it does not have it. '
                                        f'File: {yaml_fn}')
        if data[cls.YAML_FILE_TYPE] != cls.__name__:
            raise YAMLFileTypeException(f'YAML file has wrong file type. Expected {cls.__name__} but got '
                                        f'{data[cls.YAML_FILE_TYPE]} instead. File: {yaml_fn}')

        # check correct version
        if cls.YAML_FILE_VERSION not in data.keys():
            raise YAMLFileTypeException(f'YAML file must have a {cls.YAML_FILE_VERSION} field. '
                                        f'Your file may be too old if it does not have it. '
                                        f'File: {yaml_fn}')
        if data[cls.YAML_FILE_VERSION] != cls.yaml_version():
            raise YAMLFileTypeException(f'YAML file has wrong version. Expected {cls.yaml_version()} but got '
                                        f'{data[cls.YAML_FILE_VERSION]} instead. File: {yaml_fn}')

        # all ok, remove those fields from the dict
        data.pop(cls.YAML_FILE_TYPE)
        data.pop(cls.YAML_FILE_VERSION)

        return data

    def dump_yaml_data(self, yaml_fn, data):
        if self.YAML_FILE_TYPE in data.keys() or self.YAML_FILE_VERSION in data.keys():
            raise ValueError(f'Given dict must not contain keys like {self.YAML_FILE_TYPE} or {self.YAML_FILE_VERSION}')

        # add those attributes
        data[self.YAML_FILE_VERSION] = self.yaml_version()
        data[self.YAML_FILE_TYPE] = type(self).__name__

        with open(yaml_fn, 'w') as file:
            yaml.dump(data, file)


def load_mesh(mesh_fn, texture_fn=None):
    mesh = o3d.io.read_triangle_mesh(mesh_fn, enable_post_processing=True)
    if texture_fn is not None:
        mesh.textures = [o3d.io.read_image(texture_fn)]

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def save_mesh(fn, mesh_obj, overwrite_existing=True):
    """
    Saves a mesh to file.

    :param fn: string, filename
    :param mesh_obj: one of o3d.geometry.TriangleMesh, burg.core.ObjectType, burg.core.ObjectInstance, in case of the
                     latter, the mesh will be saved in the transformed pose
    :param overwrite_existing: bool, indicating whether to overwrite an existing file (defaults to True)
    """
    if not overwrite_existing and os.path.exists(fn):
        return

    if isinstance(mesh_obj, o3d.geometry.TriangleMesh):
        mesh = mesh_obj
    elif isinstance(mesh_obj, core.ObjectType):
        mesh = mesh_obj.mesh
    elif isinstance(mesh_obj, core.ObjectInstance):
        mesh = mesh_obj.get_mesh()
    else:
        return ValueError('unrecognised mesh_obj type, must be one of:' +
                          'o3d.geometry.TriangleMesh, burg.core.ObjectType, burg.core.ObjectInstance')

    # this does produce warnings that it can't write triangle normals to obj file. don't know how to suppress.
    o3d.io.write_triangle_mesh(fn, mesh, write_vertex_normals=False, write_vertex_colors=False)


def save_urdf(fn, mesh_fn, name, origin=None, inertia=None, com=None, mass=0, friction=0.24, overwrite_existing=True):
    """
    Creates a urdf file with given parameters.

    :param fn: filename of the urdf file
    :param mesh_fn: filename of the mesh to be referenced as visual and collision model
    :param name: name of the object/robot as string
    :param origin: [x,y,z] of mesh origin, if None then [0, 0, 0] will be used
    :param inertia: (3, 3) ndarray, an inertia matrix of the object, defaults to np.eye(3)*0.001
    :param com: [x,y,z] center of mass of mesh, if None then [0, 0, 0] will be used
    :param mass: float, mass of object (default 0, which means object is fixed in space)
    :param friction: float, friction coefficient (defaults to 0.24)
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
        urdf.write(f'\t\t\t<contact_coefficients mu="{friction}" />\n')
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


def save_depth_image(filename, depth, filetype=None):
    """
    Saves a depth image (e.g. generated by render module) to an image file.
    We support several file types:
    'tum', saving as '.png', depth values are multiplied with 5000 and stored as uint16 (see
    https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats).
    'exr', saving as '.exr', depth values are stored as float in a single-channel EXR image.
    'exr3', saving as '.exr', depth values are stored as float in a three-channel (RGB) EXR image -- this option adds
    overhead and should not be used unless required by dependencies postprocessing the images.
    Note that EXR options are only available if `pyexr` can be loaded, which requires OpenEXR to be installed on the
    system. See https://stackoverflow.com/a/68102521/1264582 for further information.

    :param filename: Name of the file. Need not contain the ending, unless no filetype specified, in which case we try
                      to guess the filetype based on the ending.
    :param depth: (h, w) ndarray with float values in meter.
    :param filetype: string, optional, one of 'tum', 'exr', 'exr3'.
    """
    # guess filetype from filename if not provided
    if filetype is None:
        if filename.endswith('.exr'):
            filetype = 'exr'
        elif filename.endswith('.png'):
            filetype = 'tum'
        else:
            raise ValueError(f'Could not guess filetype from filename {filename}.')

    # check if we support the requested filetype
    supported_file_types = ['tum']
    if pyexr is not None:
        supported_file_types.extend(['exr', 'exr3'])
    if filetype not in supported_file_types:
        raise ValueError(f'given filetype {filetype} not supported. mut be one of {supported_file_types}.')

    if filetype == 'tum':
        if not filename.endswith('.png'):
            filename += '.png'
        # use tum file format (which is actually a scaled 16bit png)
        # https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
        imageio.imwrite(filename, (depth * 5000).astype(np.uint16))

    elif filetype == 'exr':
        if not filename.endswith('.exr'):
            filename += '.exr'
        pyexr.write(filename, depth, channel_names=['Z'], precision=pyexr.FLOAT)

    elif filetype == 'exr3':
        if not filename.endswith('.exr'):
            filename += '.exr'
        # store images to file (extend to three channels and store in exr)
        # this is for compatibility with GPNet dataset, although it bloats the file size
        img = np.repeat(depth, 3).reshape(depth.shape[0], depth.shape[1], 3)
        pyexr.write(filename, img, channel_names=['R', 'G', 'B'], precision=pyexr.FLOAT)


class YCBObjectLibraryReader:
    """
    Class to read the YCB objects from a directory into an object library.
    Assumes directory structure:
    - base_path
        - shape_name_1
            - model_type
                - model_fn
        - shape_name_2
        - ...
    """
    def __init__(self, base_path, model_type='google_16k', model_fn='nontextured.ply'):
        self.base_path = base_path
        self.model_type = model_type
        self.model_fn = model_fn

    def read_object_library(self):
        shape_names = [x for x in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, x))]
        object_library = core.ObjectLibrary()

        for shape_name in shape_names:
            # this assumes the directory structure
            model_path = os.path.join(self.base_path, shape_name, self.model_type, self.model_fn)
            obj_type = core.ObjectType(identifier=shape_name, mesh_fn=model_path)
            object_library[shape_name] = obj_type

        # this is a bloody mass hack
        object_library['003_cracker_box'].mass = 0.411
        object_library['005_tomato_soup_can'].mass = 0.349
        object_library['006_mustard_bottle'].mass = 0.603
        object_library['010_potted_meat_can'].mass = 0.370
        object_library['025_mug'].mass = 0.118
        object_library['044_flat_screwdriver'].mass = 0.0984
        object_library['051_large_clamp'].mass = 0.125
        object_library['056_tennis_ball'].mass = 0.058

        return object_library


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
        self.object_library = core.ObjectLibrary()

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
        """converts a view into a core.CameraView object

        :param view_dict: an element in image_data_mat['imageData']

        :return: core.CameraView"""
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

        view = core.CameraView(camera_intrinsics=o3d_intrinsics, camera_pose=tf, depth_image=depth_image,
                                rgb_image=rgb_image, class_label_image=class_label_image,
                                instance_label_image=instance_label_image)
        return view

    def read_scene_files(self, filenames):
        """reads scene data

        :param filenames: dict with 'heap_fn' and 'image_data_fn' (which include full path). E.g. an element of the list
                          that has been output by `get_scene_filenames()`

        :return: core.Scene object
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
            object_instance = core.ObjectInstance(self.object_library[identifier], tf)
            object_instances.append(object_instance)

        table_pose = np.eye(4)
        table_pose[0:3, 0:3] = heap_mat['backgroundInformation']['tableBasis']
        table_pose[0:3, 3] = heap_mat['backgroundInformation']['tableCentre']
        table_instance = core.ObjectInstance(self.object_library[self.table_id], table_pose)
        bg_objects = [table_instance]

        views = [self.read_view(v) for v in image_data_mat['imageData']]

        return core.Scene(object_instances, bg_objects, views)

    def read_object_library(self):
        """
        Reads the object info from the render_data.mat file, using the paths provided to constructor.
        The object library will be stored internally, but also returned.
        It already reads the provided meshes as well, but does not create point clouds from those.

        :return: (object_library, library_index_to_name)
                 object_library is a dictionary with object identifiers as keys and core.ObjectType as values.
                 library_index_to_name maps the numeric library index to the object identifier.

        """

        input_dict = spio.loadmat(self.object_library_fn, simplify_cells=True)
        self.object_library = core.ObjectLibrary()
        self.library_index_to_name = {}

        for i, (displacement, obj_dict) in enumerate(zip(input_dict['objectCentres'], input_dict['objectInformation'])):
            name = obj_dict['name']
            mass = obj_dict['mass']
            friction_coeff = obj_dict['coefficientOfFriction']
            resitution_coeff = obj_dict['coefficientOfRestitution']

            mesh_fn = os.path.join(self.obj_models_dir, name + self.mesh_fn_ext)
            mesh = load_mesh(mesh_fn)  # todo: could also load textures here
            mesh.translate(-np.asarray(displacement))

            obj = core.ObjectType(identifier=name, mesh=mesh, mass=mass, friction_coeff=friction_coeff,
                                   restitution_coeff=resitution_coeff)
            self.object_library[name] = obj

            self.library_index_to_name[i+1] = name

        # also add the table to the object library
        table_mesh = load_mesh(self.table_fn)
        table_mesh.scale(self.table_scale_factor, np.array([0, 0, 0]))
        self.object_library[self.table_id] = core.ObjectType(identifier=self.table_id, mesh=table_mesh, mass=None)

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


def get_abs_path(fn, base_dir):
    """
    Will return the absolute path by joining base_dir and fn. If fn is None, will return None.
    """
    if fn is None:
        return None
    return os.path.join(base_dir, fn)


def get_rel_path(fn, base_dir):
    """
    Will return the relative path of fn as seen from base_dir. If fn is None, will return None.
    """
    if fn is None:
        return None
    return os.path.relpath(fn, base_dir)
