import os.path
from time import time

import numpy as np
from matplotlib import pyplot as plt

import burg_toolkit as burg


def loading_saving_lib():
    fn = '/home/rudorfem/datasets/object_libraries/test_library/test_library_def.yaml'
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_def.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    lib.print_details()
    lib.to_yaml(fn2)


def generate_urdfs():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    lib.generate_urdf_files('/home/rudorfem/datasets/object_libraries/test_library/urdf/')
    lib.to_yaml(fn2)


def making_thumbnail():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib_dir = os.path.dirname(fn2)
    lib = burg.ObjectLibrary.from_yaml(fn2)
    print(lib)
    obj = lib['006_mustard_bottle']
    obj.generate_thumbnail(os.path.join(lib_dir, 'thumbnails', f'{obj.identifier}.png'))
    lib.to_yaml(fn2)


def render_pc():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib_dir = os.path.dirname(fn2)
    lib = burg.ObjectLibrary.from_yaml(fn2)
    obj = lib['006_mustard_bottle']

    r = burg.render.MeshRenderer(output_dir=os.path.join(lib_dir, obj.identifier))
    poses = burg.render.CameraPoseGenerator().icosphere(subdivisions=1, in_plane_rotations=1, scales=1)
    scene = burg.Scene(burg.ObjectInstance(object_type=obj))
    r.render_depth(scene, poses, depth_fn_type='npy-pc', depth_fn_func=lambda i: f'pc{i}')

    pcs = [burg.visualization.create_plane(), obj.mesh]
    for i in range(len(poses)):
        pc = np.load(os.path.join(lib_dir, obj.identifier, f'pc{i}.npy'))
        pcs.append(burg.util.numpy_pc_to_o3d(pc))

    burg.visualization.show_geometries(pcs)


def pybullet_kram():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    scene, lib, printout = burg.Scene.from_yaml('/home/rudorfem/Downloads/moving_box.yaml', object_library=lib)

    target = '003_cracker_box'
    idx = 0  # object to inspect
    for i, instance in enumerate(scene.objects):
        print(instance.object_type.identifier)
        if instance.object_type.identifier == target:
            idx = i
    print(f'tracking position of target object: {idx} - {scene.objects[idx].object_type.identifier}')
    burg.visualization.show_geometries([scene])

    pose = scene.objects[idx].pose.copy()
    gs = burg.GraspSet.from_poses(pose.reshape(1, 4, 4))
    gs_prev = gs

    sim = burg.scene_sim.SceneSimulator(verbose=False, timeout=10)
    for i in range(30):
        print(f'{i}: simulator found rest after {sim.simulate_scene(scene)} seconds')
        gs_new = burg.GraspSet.from_poses(scene.objects[idx].pose.reshape(1, 4, 4))
        print(f'* init pose dist: pos {burg.metrics.euclidean_distances(gs, gs_new)[0, 0]:.7f}; '
              f'ang {np.rad2deg(burg.metrics.angular_distances(gs, gs_new))[0, 0]:.7f} degree')
        print(f'* prev pose dist: pos {burg.metrics.euclidean_distances(gs_prev, gs_new)[0, 0]:.7f}; '
              f'ang {np.rad2deg(burg.metrics.angular_distances(gs_prev, gs_new))[0, 0]:.7f} degree')
        gs_prev = gs_new
    sim.dismiss()

    print('initial rotation')
    print(gs.rotation_matrices)
    print('final rotation')
    print(gs_new.rotation_matrices)

    pose1 = burg.GraspSet.from_poses(burg.sampling.random_poses(1))
    pose2 = burg.GraspSet.from_poses(burg.sampling.random_poses(1))
    print(f'* sanity check: pos {burg.metrics.euclidean_distances(pose1, pose2)[0, 0]:.7f}; '
          f'ang {np.rad2deg(burg.metrics.angular_distances(pose1, pose2))[0, 0]:.7f} degree')


def compute_stable_poses():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    obj = lib['051_large_clamp']

    #stable_poses = burg.mesh_processing.compute_stable_poses(obj, verify_in_sim=False)
    #print(stable_poses)
    for prob, pose in obj.stable_poses:
        instance = burg.ObjectInstance(obj, pose)
        burg.visualization.show_geometries([burg.visualization.create_plane(), instance.get_mesh()])

    #lib.compute_stable_poses()
    #lib.to_yaml(fn2)


def check_stable_pose():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    obj = lib['051_large_clamp']

    orig_pose = obj.stable_poses.sample_pose()
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng()
    for i in range(5):
        angle = rng.random() * np.pi * 2
        tf_rot = np.eye(4)
        tf_rot[:3, :3] = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
        pose = tf_rot @ orig_pose
        pose[:2, 3] = rng.random(2) * (0.5, 0.5)

        instance = burg.ObjectInstance(obj, pose)
        burg.visualization.show_geometries([burg.visualization.create_plane(), instance.get_mesh()])


def check_scene_sampling():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_def.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    object_names = list(lib.keys())
    print('all attribs?', lib.objects_have_all_attributes())

    # scene = burg.sampling.sample_scene(lib, burg.Scene.size_A3, instances_per_scene=5, instances_per_object=1)
    # burg.visualization.show_geometries([scene])
    # img = scene.render_printout()  # uses pyrender

    scene_size = burg.constants.SIZE_A4
    scene = burg.sampling.sample_scene(lib, scene_size, instances_per_scene=1, instances_per_object=1)
    burg.visualization.show_geometries([scene])

    printout = burg.printout.Printout(scene_size)
    printout.add_scene(scene)
    start_time = time()
    img = printout.get_image()
    elapsed = time() - start_time
    print(f'producing image took {elapsed*1000:.4f} ms')
    plt.imshow(img, cmap='gray')
    plt.show()
    print('marker info:\n', printout.marker_info)
    start_time = time()
    printout.save_pdf(os.path.join(os.path.dirname(fn2), 'printout.pdf'), page_size=burg.constants.SIZE_A4)
    elapsed = time() - start_time
    print(f'producing image, pdf and saving took {elapsed*1000:.4f} ms')
    printout.save_image(os.path.join(os.path.dirname(fn2), 'printout.png'))

    yaml_fn = os.path.join(os.path.dirname(fn2), 'scene.yaml')
    scene.to_yaml(yaml_fn, object_library=lib, printout=printout)
    print(scene)

    scene, library, printout = burg.Scene.from_yaml(yaml_fn, lib)
    print(scene)
    print(printout.to_dict())
    frame = printout.get_marker_frame()
    print(frame)
    frame = burg.visualization.create_frame(pose=frame)
    burg.visualization.show_geometries([scene, frame])


def check_rendering():
    fn2 = '/home/rudorfem/datasets/object_libraries/test_library/test_library_roundtrip.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn2)
    obj = lib['051_large_clamp']

    engine = burg.render.RenderEngineFactory.create('pybullet')
    print('engine:', engine._p)
    renderer = burg.render.ThumbnailRenderer(engine, size=128)
    engine.dismiss()
    print('engine:', engine._p)
    engine2 = burg.render.RenderEngineFactory.create('pybullet')
    print('engine2:', engine2._p)
    #img = renderer.render(obj)
    renderer = burg.render.ThumbnailRenderer(engine2, size=128)
    print('engine2:', engine2._p)
    img2 = renderer.render(obj)
    img = renderer.render(obj)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(img2)
    plt.show()


def show_off():
    fn = '/home/rudorfem/datasets/l2g-ycb-test-set/object_library.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn)
    print(lib)

    while True:
        scene = burg.sampling.sample_scene(lib, burg.constants.SIZE_A4, instances_per_scene=5)
        printout = burg.printout.Printout(burg.constants.SIZE_A4)
        printout.add_scene(scene)

        lib_dir = os.path.dirname(fn)
        printout_fn = os.path.join(lib_dir, 'printout.pdf')
        printout.save_pdf(printout_fn)

        burg.visualization.show_geometries([scene])


def check_z_values():
    fn = '/home/rudorfem/datasets/l2g-ycb-test-set/object_library.yaml'
    lib = burg.ObjectLibrary.from_yaml(fn)

    while True:
        scene = burg.sampling.sample_scene(lib, burg.constants.SIZE_A4, instances_per_scene=5)
        printout = burg.printout.Printout(burg.constants.SIZE_A4)
        printout.add_scene(scene)

        tmp_dir = '/home/rudorfem/tmp'
        scene.to_yaml(os.path.join(tmp_dir, 'scene.yaml'), lib, printout)
        printout_fn = os.path.join(tmp_dir, 'printout.pdf')
        printout.save_pdf(printout_fn)

        burg.visualization.show_geometries([scene])


def io_scene():
    fn = '/home/rudorfem/tmp/scene.yaml'
    scene, lib, template = burg.Scene.from_yaml(fn)
    scene.to_yaml(fn, lib, template)


def recreate_urdf():
    from xml.etree import ElementTree

    path = '/home/rudorfem/datasets/l2g-ycb-test-set/urdf/'
    fns = os.listdir(path)
    for fn in fns:
        datafile = os.path.join(path, fn)
        tree = ElementTree.parse(datafile)
        node = tree.find('.//mass')
        mass = float(node.get('value'))
        node.set('value', str(mass/1000))
        tree.write(datafile)


if __name__ == "__main__":
    # loading_saving_lib()
    # compute_stable_poses()
    # check_scene_sampling()
    # check_stable_pose()
    # pybullet_kram()
    # check_rendering()
    # show_off()
    # check_z_values()
    # io_scene()
    recreate_urdf()
