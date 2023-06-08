"""
===================
Setup Scene Example
===================

This script provides examples for loading, using and saving an object library based on a YAML file.
We will step through the individual commands, generate thumbnails, VHACD meshes, URDF files for all objects.
After that, we will compute the stable poses of each object.
We randomly sample scenes exploiting the stable poses of the objects.
We can interact with the object instances and move them, and put them into a simulator so they attain a resting
pose again.
"""

import argparse
import os

import burg_toolkit as burg


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', type=str,
                        default='/home/rudorfem/datasets/object_libraries/test_library/test_library.yaml',
                        help='path to object library file')
    parser.add_argument('--skip', action='store_true', default=False,
                        help='use this option to skip through the user interactions')
    parser.add_argument('--override', action='store_true', default=False,
                        help='if activated, all object data will be regenerated even if it already exists')
    return parser.parse_args()


def wait_for_user(skip=False):
    if skip:
        return
    print('press any key to continue')
    input()


def main(object_library_fn, skip, override):
    object_library = burg.ObjectLibrary.from_yaml(object_library_fn)
    library_dir = os.path.dirname(object_library_fn)

    print(object_library)  # prints short version
    object_library.print_details()  # gives more detailed output about contained objects

    print('*************************')
    print('next action: generate thumbnail files for all objects')
    wait_for_user(skip)
    thumbnail_dir = os.path.join(library_dir, 'thumbnails')
    object_library.generate_thumbnails(thumbnail_dir, override=override)
    print(f'thumbnails created in {thumbnail_dir}')

    print('*************************')
    print('next action: generate vhacd meshes')
    wait_for_user(skip)
    vhacd_dir = os.path.join(library_dir, 'vhacd')
    object_library.generate_vhacd_files(vhacd_dir, override=override)
    print(f'vhacd files created in {vhacd_dir}')

    print('*************************')
    print('next action: generate urdf files')
    wait_for_user(skip)
    urdf_dir = os.path.join(library_dir, 'urdf')
    object_library.generate_urdf_files(urdf_dir, use_vhacd=True, override=override)
    print(f'urdf files created in {urdf_dir}')

    print('*************************')
    print('next action: compute stable poses for objects and verify with vhacd in simulation')
    wait_for_user(skip)
    object_library.compute_stable_poses(verify_in_sim=True, override=override)
    print('stable poses computed.')

    print('*************************')
    print('all information in object library should be completed now:')
    object_library.print_details()

    print('*************************')
    new_lib_fn = f'{object_library_fn[:-5]}_roundtrip.yaml'
    print(f'next action: save object library to {new_lib_fn}')
    wait_for_user(skip)
    object_library.to_yaml(new_lib_fn)
    print('object library saved.')

    print('*************************')
    print('next action: sampling scenes with object instances in stable poses, and visualise.')
    print('note: you need to close the open3d window to continue. (not the simulation window later on, though!)')
    dim = (1, 0.5)
    n_instances = min(5, len(object_library))
    print(f'{n_instances} instances will be placed in ground area of {dim}')
    wait_for_user(skip)
    scene = burg.sampling.sample_scene(
        object_library,
        ground_area=dim,
        instances_per_scene=n_instances,
        instances_per_object=1
    )
    burg.visualization.show_geometries([scene])

    print('*************************')
    print('next action: simulate this scene to make sure it is at rest, then visualise again.')
    wait_for_user(skip)
    sim = burg.scene_sim.SceneSimulator(verbose=True)  # verbose shows the simulator GUI, slower than real-time
    sim.simulate_scene(scene)  # the poses of all instances in the scene are automatically updated by the simulator
    sim.dismiss()  # can also reuse, then the window stays open
    burg.visualization.show_geometries([scene])

    print('*************************')
    print('next action: manually change the pose of an object instance, visualise, simulate, visualise.')
    wait_for_user(skip)
    instance = scene.objects[0]
    # we lift it up a bit to avoid any collisions with other objects
    instance.pose[2, 3] = instance.pose[2, 3] + 0.2
    burg.visualization.show_geometries([scene])
    sim = burg.scene_sim.SceneSimulator(verbose=True)
    sim.simulate_scene(scene)
    burg.visualization.show_geometries([scene])
    sim.dismiss()

    print('*************************')
    print('that was all, thank you and good bye.')


if __name__ == "__main__":
    args = parse_args()
    main(args.lib, args.skip, args.override)
