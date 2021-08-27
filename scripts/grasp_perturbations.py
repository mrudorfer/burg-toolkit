import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import burg_toolkit as burg

try:
    import gpnet_sim
except ImportError:
    gpnet_sim = None
    print('Warning: package gpnet_sim not found. Please install from https://github.com/mrudorfer/GPNet-simulator')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--gpnet_data', type=str, default='/home/rudorfem/datasets/GPNet_release_data_fixed/',
                        help='path to GPNet dataset')
    parser.add_argument('-md', '--model_dir', type=str,
                        default='/home/rudorfem/dev/GPNet-simulator/gpnet_data/processed/',
                        help='path to the GPNet models (obj files)')
    parser.add_argument('-s', '--shape', type=str, default='c34718bd10e378186c6c61abcbd83e5a',
                        help='name of shape to process')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/rudorfem/datasets/YCB_grasp_tmp/',
                        help='where to put generated dataset files')
    parser.add_argument('-n', '--num_grasps', type=int, default=None,
                        help='number of grasps to process. up to first n successful grasps used. None processes all.')
    parser.add_argument('--simulate', action='store_true', default=False,
                        help='activate to perturb grasps and simulate all of them (takes time)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='use this option to visualise perturbed grasps (taken from output_dir)')
    return parser.parse_args()


# function to load data of a given GPNet object
def load_successful_grasps(gpnet_object_id, gpnet_data_path, gpnet_model_dir):
    print(f'loading grasps for object {gpnet_object_id}')

    center_fn = os.path.join(gpnet_data_path, 'annotations/candidate/', gpnet_object_id + '_c.npy')
    quat_fn = os.path.join(gpnet_data_path, 'annotations/candidate/', gpnet_object_id + '_q.npy')
    success_fn = os.path.join(gpnet_data_path, 'annotations/simulateResult/', gpnet_object_id + '.npy')

    centers = np.load(center_fn)
    quats = np.load(quat_fn)
    success = np.load(success_fn)

    print(f'total grasps: {len(centers)}')
    print(f'successful:   {np.count_nonzero(success)}')
    print(f'ratio:        {np.mean(success)}')

    centers = centers[np.nonzero(success)]
    quats = quats[np.nonzero(success)]
    gs = burg.grasp.GraspSet.from_translations_and_quaternions(centers, quats)

    # model_fn = os.path.join(gpnet_model_dir, gpnet_object_id + '.obj')
    # mesh = burg.io.load_mesh(model_fn)
    # burg.visualization.show_grasp_set([mesh], gs, gripper=burg.gripper.ParallelJawGripper(), with_plane=True)

    return gs


def perturb_and_simulate(grasps, gpnet_object_id):
    print(f'going through {len(grasps)} grasps to perturb and simulate.')

    n_perturbations = 37

    scores = np.zeros(len(grasps))
    overall_success = np.zeros(n_perturbations)  # so we can reason about certain perturbations

    perturbations = burg.sampling.grasp_perturbations(grasps, radii=[5, 10, 15])
    conf = gpnet_sim.default_conf()
    conf.z_move = True
    success, outcomes = gpnet_sim.simulate_direct(
        conf, gpnet_object_id, perturbations.translations, perturbations.quaternions)

    success = success.reshape(len(grasps), n_perturbations)
    scores = np.mean(success, axis=-1)
    overall_success = np.mean(success, axis=0)

    grasps.scores = scores
    print('***** SUMMARY *****')
    print('average robustness score:', np.mean(scores))
    print(f'max/min robustness score: {np.max(scores)}/{np.min(scores)}')
    print('overall outcomes,\tavg per grasp,\tavg per perturbation:')
    for outcome, count in outcomes.items():
        print(f'\t{outcome}\t{count}\t{count/len(grasps)}\t{count/len(grasps)/n_perturbations} ')
    print('overall success per perturbation:')
    for i, success in enumerate(overall_success):
        print(f'\t{i}\t{success}')

    return success, outcomes


def analyse_perturbation_success(success):
    # we've got 37 fields in the array
    print('*** perturbation analysis ***')
    print('original grasps:', success[0])

    # the loop goes like this:
    # - radius (5, 10, 15)
    #   - axis (trans xyz then rot xyz)
    #       - plus, then minus
    for i, label in enumerate(['x_trans', 'y_trans', 'z_trans', 'x_rot', 'y_rot', 'z_rot']):
        sum = 0
        for r, rad in enumerate(['5', '10', '15']):
            idx_pos = 1 + 2*i + 12*r
            idx_neg = idx_pos + 1
            sum += success[idx_pos] + success[idx_neg]
            print(f'{label}_{rad}:\t{(success[idx_pos] + success[idx_neg])/2}')
            if label == 'z_trans':
                print(f'\tfarther: \t{success[idx_pos]}')
                print(f'\tcloser: \t{success[idx_neg]}')
        print(f'{label}_avg:\t{sum/6}')


def visualise_perturbation_success(success):
    pose = np.eye(4)
    pose[0:3, 1] = [0, 0, -1]
    pose[0:3, 2] = [0, 1, 0]
    gs = burg.grasp.GraspSet(1)
    gs[0].pose = pose
    gs_perturbed = burg.sampling.grasp_perturbations(gs[0], radii=[5, 10, 15])
    gs_perturbed.scores = success
    gripper = burg.gripper.ParallelJawGripper()
    burg.visualization.show_grasp_set([], gs_perturbed, gripper=gripper,
                                      score_color_func=lambda s: [1-s, s, 0])


if __name__ == "__main__":
    print('hi')
    print('args:')
    args = parse_args()
    for key, value in vars(args).items():
        print(f'\t{key}:\t{value}')
    print('****')

    grasps = load_successful_grasps(args.shape, args.gpnet_data, args.model_dir)
    if args.num_grasps is not None:
        grasps = grasps[:args.num_grasps]

    if args.simulate is True:
        success, outcomes = perturb_and_simulate(grasps, args.shape)
        robustness_scores = np.mean(success, axis=-1)
        perturbation_success = np.mean(success, axis=0)

        burg.io.make_sure_directory_exists(args.output_dir)
        np.save(os.path.join(args.output_dir, args.shape + '_success.npy'), success)
        np.save(os.path.join(args.output_dir, args.shape + '_outcomes.npy'), outcomes)

    if args.show is True:
        # we changed file output at some point, so first check if we have the newer version
        success_fn = os.path.join(args.output_dir, args.shape + '_success.npy')
        if os.path.isfile(success_fn):
            success = np.load(success_fn)
            robustness_scores = np.mean(success, axis=-1)
            perturbation_success = np.mean(success, axis=0)
        else:
            # this is the legacy version which does not retain binary success values for all perturbations
            success = None
            robustness_scores = np.load(os.path.join(args.output_dir, args.shape + '_scores.npy'))
            perturbation_success = np.load(os.path.join(args.output_dir, args.shape + '_success_p.npy'))

        analyse_perturbation_success(perturbation_success)
        visualise_perturbation_success(perturbation_success)

        print(f'loaded robustness scores for {len(robustness_scores)} grasps, having {len(grasps)} grasps.')
        print('\tavg:', np.mean(robustness_scores))
        print('\tmin:', np.min(robustness_scores))
        print('\tmax:', np.max(robustness_scores))
        print('\tmed:', np.median(robustness_scores))
        print(f'\t>0.5: {np.mean(robustness_scores > 0.5)} ({np.count_nonzero(robustness_scores > 0.5)})')
        print(f'\t>0.75: {np.mean(robustness_scores > 0.75)} ({np.count_nonzero(robustness_scores > 0.75)})')
        print(f'\t>0.9: {np.mean(robustness_scores > 0.9)} ({np.count_nonzero(robustness_scores > 0.9)})')

        plt.hist(robustness_scores, bins=np.linspace(0, 1, 21))
        plt.title(f'score distribution {args.shape}')
        plt.show()

        robustness_scores = robustness_scores[:len(grasps)]
        grasps = grasps[:len(robustness_scores)]
        grasps.scores = robustness_scores

        model_fn = os.path.join(args.model_dir, args.shape + '.obj')
        mesh = burg.io.load_mesh(model_fn)
        # first just visualise the best 10 and worst 10 grasps
        top_n = 15
        best_indices = np.argpartition(robustness_scores, -top_n)[-top_n:]
        worst_indices = np.argpartition(robustness_scores, top_n)[:top_n]
        indices = np.concatenate([worst_indices, best_indices])
        burg.visualization.show_grasp_set([mesh], grasps[indices], gripper=burg.gripper.ParallelJawGripper(),
                                          with_plane=True, score_color_func=lambda s: [1-s, s, 0])

        #burg.visualization.show_grasp_set([mesh], grasps, gripper=burg.gripper.ParallelJawGripper(),
        #                                  with_plane=True, score_color_func=lambda s: [1-s, s, 0], n=3000)

    print('bye')
