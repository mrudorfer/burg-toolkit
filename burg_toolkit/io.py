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


def load_mesh(mesh_fn, texture_fn=None):
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    if texture_fn is not None:
        mesh.textures = [o3d.io.read_image(texture_fn)]

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


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
