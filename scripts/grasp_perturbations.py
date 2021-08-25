import os
import argparse
from collections import Counter

import numpy as np
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
    overall_outcomes = Counter()
    overall_success = np.zeros(n_perturbations)  # so we can reason about certain perturbations

    for i, grasp in enumerate(grasps):
        perturbations = burg.sampling.grasp_perturbations(grasp, radii=[5, 10, 15])

        conf = gpnet_sim.default_conf()
        conf.z_move = True
        success, outcomes = gpnet_sim.simulate_direct(
            conf, gpnet_object_id, perturbations.translations, perturbations.quaternions)
        scores[i] = np.mean(success)
        overall_success += success
        overall_outcomes += Counter(outcomes)

    grasps.scores = scores
    print('***** SUMMARY *****')
    print('average robustness score:', np.mean(scores))
    print(f'max/min robustness score: {np.max(scores)}/{np.min(scores)}')
    print('overall outcomes,\tavg per grasp,\tavg per perturbation:')
    for outcome, count in overall_outcomes.items():
        print(f'\t{outcome}\t{count}\t{count/len(grasps)}\t{count/len(grasps)/n_perturbations} ')
    print('overall success per perturbation (abs vs rel):')
    for i, success in enumerate(overall_success):
        print(f'\t{i}\t{success}\t{success/len(grasps)}')

    return scores, overall_success, overall_outcomes


if __name__ == "__main__":
    print('hi')
    args = parse_args()
    grasps = load_successful_grasps(args.shape, args.gpnet_data, args.model_dir)
    if args.num_grasps is not None:
        grasps = grasps[:args.num_grasps]
    robustness_scores, perturbation_success, outcomes = perturb_and_simulate(grasps, args.shape)

    burg.io.make_sure_directory_exists(args.output_dir)

    np.save(os.path.join(args.output_dir, args.shape + '_scores.npy'), robustness_scores)
    np.save(os.path.join(args.output_dir, args.shape + '_success_p.npy'), perturbation_success)
    np.save(os.path.join(args.output_dir, args.shape + '_outcomes.npy'), outcomes)

    print('bye')
