"""
This script translates the ShapeNetSem models used in ACRONYM into an ObjectLibrary, so it can be used within
the BURG toolkit.
"""

import argparse
import os
import h5py

import numpy as np
import burg_toolkit as burg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acronym_dir', type=str, default='/home/rudorfem/datasets/acronym/')
    return parser.parse_args()


def main(args):
    acronym_dir = args.acronym_dir
    lib = burg.ObjectLibrary(name='ACRONYM objects',
                             description='objects from ShapeNetSem, used for ACRONYM')
    lib.to_yaml(os.path.join(acronym_dir, 'object_library.yml'))

    # Each shape must be translated into a burg.ObjectType. All ObjectTypes within one ObjectLibrary must have a
    # unique identifier, which is a bit troublesome as there are several duplicate shapes which only distinguish
    # themselves in the scale factor.

    # create object type: identifier=category/shape (this may cause trouble when automatically generating folders)
    # set mesh_fn
    # create vhacd (create all the category folders first)
    # create urdf WITH scale, i.e. ObjectTypes need to be WITH scale
    grasp_annotation_dir = os.path.join(acronym_dir, 'grasps')
    filenames = os.listdir(grasp_annotation_dir)
    for filename in filenames:
        cat, shape, _ = filename[:-len('.h5')].split('_')
        grasps = h5py.File(os.path.join(grasp_annotation_dir, filename), 'r')
        mesh_fn = os.path.join(acronym_dir, grasps['object/file'][()].decode('utf-8'))
        scale = float(grasps['object/scale'][()])  # rather read scale directly than from grasp filename

        obj = burg.ObjectType(
            identifier=f'{cat}_{shape}_{scale}',
            mesh_fn=mesh_fn,
            mass=float(grasps['object/mass'][()]),
            friction_coeff=float(grasps['object/friction'][()]),
            scale=scale
        )

        vhacd_dir = os.path.join(acronym_dir, 'vhacd', cat)
        burg.io.make_sure_directory_exists(vhacd_dir)
        vhacd_fn = os.path.join(vhacd_dir, f'{shape}.obj')
        # obj.generate_vhacd(vhacd_fn)

        # we have to create urdf customly to fill in more parameters... this is probs not ideal, but ok for now
        urdf_dir = os.path.join(acronym_dir, 'urdf', cat)
        burg.io.make_sure_directory_exists(urdf_dir)
        urdf_fn = os.path.join(urdf_dir, f'{shape}_{scale}.urdf')
        rel_mesh_fn = os.path.relpath(vhacd_fn, os.path.dirname(urdf_fn))

        burg.io.save_urdf(urdf_fn, rel_mesh_fn, name=obj.identifier,
                          inertia=np.array(grasps['object/inertia']),
                          com=np.array(grasps['object/com']),
                          mass=obj.mass,
                          friction=obj.friction_coeff
                          # todo: scale
                          )
        obj.urdf_fn = urdf_fn
        lib[obj.identifier] = obj
        break
    lib.to_yaml()


if __name__ == '__main__':
    main(parse_args())
