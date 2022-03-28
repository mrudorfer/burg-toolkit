"""
This script translates the ShapeNetSem models used in ACRONYM into an ObjectLibrary, so it can be used within
the BURG toolkit.
"""

import argparse
import os
from functools import partial

import h5py
import numpy as np
import burg_toolkit as burg
from tqdm.contrib.concurrent import process_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acronym_dir', type=str, default='/home/rudorfem/datasets/acronym/')
    return parser.parse_args()


def create_object_type(grasp_filename, acronym_dir):
    # Each shape must be translated into a burg.ObjectType. All ObjectTypes within one ObjectLibrary must have a
    # unique identifier, which is a bit troublesome as there are several duplicate shapes which only distinguish
    # themselves in the scale factor. They must be different ObjectTypes.

    cat, shape, _ = grasp_filename[:-len('.h5')].split('_')
    grasps = h5py.File(os.path.join(acronym_dir, 'grasps', grasp_filename), 'r')
    mesh_fn = os.path.join(acronym_dir, grasps['object/file'][()].decode('utf-8'))
    scale = float(grasps['object/scale'][()])  # rather read scale directly than from grasp filename

    # create object type: identifier=category_shape_scale
    # set all parameters
    obj = burg.ObjectType(
        identifier=f'{cat}_{shape}_{scale}',
        mesh_fn=mesh_fn,
        mass=float(grasps['object/mass'][()]),
        friction_coeff=float(grasps['object/friction'][()]),
        scale=scale
    )

    # create vhacd
    # duplicate shapes with different scales will have same VHACD file, as it is scaled during loading
    vhacd_dir = os.path.join(acronym_dir, 'vhacd', cat)
    burg.io.make_sure_directory_exists(vhacd_dir)
    vhacd_fn = os.path.join(vhacd_dir, f'{shape}.obj')
    if not os.path.isfile(vhacd_fn):
        obj.generate_vhacd(vhacd_fn)
    obj.vhacd_fn = vhacd_fn  # make sure object type has this property even if it did not generate vhacd itself

    # however, each object type has its own urdf file, as it is scale-specific
    # some parameters (inertia/com) are given in ACRONYM annotations, so let's use those directly
    urdf_dir = os.path.join(acronym_dir, 'urdf', cat)
    burg.io.make_sure_directory_exists(urdf_dir)
    urdf_fn = os.path.join(urdf_dir, f'{shape}_{scale}.urdf')
    rel_mesh_fn = os.path.relpath(vhacd_fn, os.path.dirname(urdf_fn))

    burg.io.save_urdf(urdf_fn, mesh_fn=rel_mesh_fn, name=obj.identifier, origin=[0, 0, 0],
                      inertia=np.array(grasps['object/inertia']), com=np.array(grasps['object/com']),
                      mass=obj.mass, friction=obj.friction_coeff, scale=obj.scale,
                      overwrite_existing=True)
    obj.urdf_fn = urdf_fn
    return obj


def main(args):
    acronym_dir = args.acronym_dir
    lib = burg.ObjectLibrary(name='ACRONYM objects',
                             description='objects from ShapeNetSem, used for ACRONYM')
    lib.to_yaml(os.path.join(acronym_dir, 'object_library.yml'))

    grasp_annotation_dir = os.path.join(acronym_dir, 'grasps')
    filenames = os.listdir(grasp_annotation_dir)

    # execute concurrently, will take a while...
    object_types = process_map(partial(create_object_type, acronym_dir=acronym_dir), filenames, chunksize=10)
    for obj in object_types:
        lib[obj.identifier] = obj
    lib.to_yaml()


if __name__ == '__main__':
    main(parse_args())
