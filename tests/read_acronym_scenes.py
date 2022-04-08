import burg_toolkit as burg


def main():
    print('loading object library...')
    lib = burg.ObjectLibrary.from_yaml('/home/martin/datasets/acronym/object_library.yml')
    print(f'loaded lib with {len(lib)} objects')

    scene_idx = 15
    while True:
        print(f'reading scene {scene_idx}')
        acronym = burg.io.ACRONYMSceneReader(acronym_dir='/home/martin/datasets/acronym', object_library=lib)
        scene, grasp_sets = acronym.get_scene(scene_idx)

        # visualise scene
        burg.visualization.show_geometries([scene])

        # simulate scene
        sim = burg.scene_sim.SceneSimulator(verbose=True)
        sim.simulate_scene(scene)

        # for instance in scene.objects:
        #     print('target object:', instance.object_type.identifier)
        #     burg.visualization.show_grasp_set([scene], grasp_sets[instance],
        #                                       gripper=burg.gripper.TwoFingerGripperVisualisation(),
        #                                       n=500)
        scene_idx += 1


if __name__ == '__main__':
    main()
