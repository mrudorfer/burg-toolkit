import glob

import mat73
import scipy.io as spio
import h5py
import open3d as o3d

from . import scene
from . import grasp


def load_mesh(mesh_fn, texture_fn=None):
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    if texture_fn is not None:
        mesh.textures = [o3d.io.read_image(texture_fn)]

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# get scene file names
def get_scene_filenames(directory):
    """finds heap and imagedata files in the given directory

    :param directory: directory containing the files

    :return: list of dicts with heap_fn and image_data_fn which both include the full path
    """

    heap_files = glob.glob(directory + "Data*ObjectHeap*.mat")
    image_files = glob.glob(directory + "Images*ObjectHeap*.mat")
    if len(heap_files) != len(image_files):
        print("warning: different number of heap and image files found")
        return {}

    # make sure the file names match as well?

    filenames = []
    for heap_fn, image_fn in zip(heap_files, image_files):
        filenames.append({
            'heap_fn': heap_fn,
            'image_data_fn': image_fn
        })

    return filenames


def read_scene_files(filenames):
    """reads scene data

    :param filenames: dict with 'heap_fn' and 'image_data_fn' (which include full path)

    :return: core_types.Scene object
    """
    heap_mat = spio.loadmat(filenames['heap_fn'], simplify_cells=True)
    # image_data is a v7.3 mat stored in hdf5 format, thus needs different reader
    image_data_mat = mat73.loadmat(filenames['image_data_fn'])

    objects = [scene.ObjectInstance(obj) for obj in heap_mat['heap']]
    views = [scene.CameraView(v) for v in image_data_mat['imageData']]

    table = scene.BackgroundObject.from_translation_rotation(
        name='table',
        translation=heap_mat['backgroundInformation']['tableCentre'],
        rotation=heap_mat['backgroundInformation']['tableBasis']
    )
    bg_objects = [table]

    return scene.Scene(objects, bg_objects, views)


def read_object_library(object_lib_path):
    """
    reads the object info

    :param object_lib_path: path of render_data.mat file

    :return: list of core_types.ObjectType
    """

    input_dict = spio.loadmat(object_lib_path, simplify_cells=True)

    object_library = [scene.ObjectType(obj_dict, displacement)
                      for displacement, obj_dict
                      in zip(input_dict['objectCentres'], input_dict['objectInformation'])]

    return object_library


def read_grasp_file_eppner2019(grasp_fn):
    """
    Reads grasps from the grasp file of dataset provided with publication of Eppner et al. 2019.

    It should contain densely sampled, successful grasps (verified in their simulation).

    :param grasp_fn: the filename

    :return: a core_types.GraspSet, and center of mass (as np array with length 3)
    """

    hf = h5py.File(grasp_fn, 'r')
    print('just keys:', list(hf.keys()))

    print_keys = ['gripper', 'object', 'object_class', 'object_dataset']
    for key in print_keys:
        print(key, hf[key][()].decode())

    print('object_scale:', hf['object_scale'][()])
    print('poses.shape:', hf['poses'].shape)
    print('com', hf['object_com'][:])

    print('creating grasp set...')
    gs = grasp.GraspSet.from_translations_and_quaternions(hf['poses'])
    print('done')

    return gs, hf['object_com'][:]
