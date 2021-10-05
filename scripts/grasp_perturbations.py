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
    parser.add_argument('-o', '--output_dir', type=str,
                        default='/home/rudorfem/datasets/GPNet_release_data_fixed/perturbations/',
                        help='where to put generated dataset files')
    parser.add_argument('-n', '--num_grasps', type=int, default=None,
                        help='number of grasps to process. up to first n successful grasps used. None processes all.')
    parser.add_argument('--simulate', action='store_true', default=False,
                        help='activate to perturb grasps and simulate all of them (takes time)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='use this option to visualise perturbed grasps (taken from output_dir)')
    parser.add_argument('--negatives', action='store_true', help='use negative instead of positive annotations')
    return parser.parse_args()


def get_objects_and_grasp_numbers(gpnet_data_path):
    img_dir = os.path.join(gpnet_data_path, 'images/')
    shape_names = [x for x in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, x))]

    grasp_num_dict = {}

    for shape_name in shape_names:
        if shape_name == '4eefe941048189bdb8046e84ebdc62d2':  # no GT data for this shape
            continue
        grasps = load_gt_grasps(shape_name, gpnet_data_path)
        grasp_num_dict[shape_name] = len(grasps)

    for key, item in grasp_num_dict.items():
        print(f'{key}: {item}')

    print('total grasps:\t', sum(grasp_num_dict.values()))
    values = np.fromiter(grasp_num_dict.values(), dtype=int)
    print('values shape', values.shape)
    print('max num', np.max(values))
    print('min num', np.min(values))
    plt.hist(values)
    plt.title(f'distribution of number of GT grasps')
    plt.show()


# function to load data of a given GPNet object
def load_gt_grasps(gpnet_object_id, gpnet_data_path, gpnet_model_dir=None, type='positives'):
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

    if type == 'positives':
        centers = centers[np.nonzero(success)]
        quats = quats[np.nonzero(success)]
    elif type == 'negatives':
        centers = centers[np.nonzero(1 - success)]
        quats = quats[np.nonzero(1 - success)]
    elif type != 'all':
        raise ValueError(f'provided type unrecognised, got {type} but expected one of [positives, negatives, all]')

    gs = burg.grasp.GraspSet.from_translations_and_quaternions(centers, quats)

    # if gpnet_model_dir is not None:
    #   model_fn = os.path.join(gpnet_model_dir, gpnet_object_id + '.obj')
    #   mesh = burg.io.load_mesh(model_fn)
    #   burg.visualization.show_grasp_set([mesh], gs, gripper=burg.gripper.ParallelJawGripper(), with_plane=True)

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
    for i, s in enumerate(overall_success):
        print(f'\t{i}\t{s}')

    return success, outcomes


def analyse_perturbation_success(success):
    # we've got 37 fields in the array
    print('*** perturbation analysis ***')
    print('original grasps:', success[0])

    # the loop goes like this:
    # - radius (5, 10, 15)
    #   - axis (trans xyz then rot xyz)
    #       - plus, then minus
    rads = ['5', '10', '15']
    for i, label in enumerate(['x_{trans}', 'y_{trans}', 'z_{trans}', 'x_{rot}', 'y_{rot}', 'z_{rot}']):
        sum = 0
        for r, rad in enumerate(rads):
            idx_pos = 1 + 2*i + 12*r
            idx_neg = idx_pos + 1
            sum += success[idx_pos] + success[idx_neg]
            print(f'{label}_{rad}:\t{(success[idx_pos] + success[idx_neg])/2}')
            if label == 'z_{trans}':
                print(f'\tfarther: \t{success[idx_pos]}')
                print(f'\tcloser: \t{success[idx_neg]}')
        print(f'{label}_avg:\t{sum/6}')

    print('** tex table **')
    print('\\begin{tabular}{ |l||c|c|c||c| }')
    print('\t\\hline')
    print('\t& \\multicolumn{4}{c|}{success rate at radius}\\\\')
    print('\tperturbation & 5 & 10 & 15 & avg \\\\ ')
    print('\t\\hline')
    print(f'\tnone & \\multicolumn{{3}}{{c||}}{{{success[0]:.2f}}} & \\\\')
    print('\t\\hline')
    avg = np.zeros(len(rads)+1)
    labels = ['x_{trans}', 'y_{trans}', 'z_{trans}', 'x_{rot}', 'y_{rot}', 'z_{rot}']
    for i, label in enumerate(labels):
        sum = 0
        line = f'\t$\\pm {label}$\t'
        for r, rad in enumerate(rads):
            idx_pos = 1 + 2 * i + 12 * r
            idx_neg = idx_pos + 1
            val = (success[idx_pos] + success[idx_neg]) / 2
            sum += val
            avg[r] += val
            line += f'& {val:.2f}\t'
        line += f'& {sum / len(rads):.2f} \\\\'
        avg[-1] += sum / len(rads)
        print(line)
    print('\t\\hline')
    line = '\taverage'
    for i in range(len(rads)+1):
        line += f'\t& {avg[i]/len(labels):.2f}'
    line += ' \\\\'
    print(line)
    print('\t\\hline')
    print('\\end{tabular}')
    print('** end table **')


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

    # uncomment to see statistics about number of annotations
    # get_objects_and_grasp_numbers(args.gpnet_data)

    grasps = None
    neg_str = ''
    if args.negatives:
        grasps = load_gt_grasps(args.shape, args.gpnet_data, args.model_dir, type='negatives')
        neg_str = '_neg'
    else:
        grasps = load_gt_grasps(args.shape, args.gpnet_data, args.model_dir, type='positives')
    if args.num_grasps is not None:
        if args.num_grasps < len(grasps):
            indices = np.random.choice(len(grasps), args.num_grasps, replace=False)
            grasps = grasps[indices]

    if args.simulate is True:
        success, outcomes = perturb_and_simulate(grasps, args.shape)
        print('success shape', success.shape)
        robustness_scores = np.mean(success, axis=-1)
        perturbation_success = np.mean(success, axis=0)

        burg.io.make_sure_directory_exists(args.output_dir)
        np.save(os.path.join(args.output_dir, args.shape + neg_str + '_success.npy'), success)
        np.save(os.path.join(args.output_dir, args.shape + neg_str + '_outcomes.npy'), outcomes)

    if args.show is True:
        # we changed file output at some point, so first check if we have the newer version
        success_fn = os.path.join(args.output_dir, args.shape + neg_str + '_success.npy')
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
        score_thresholds = [0.5, 0.75, 0.9]
        for th in score_thresholds:
            print(f'\t>{th}: {np.mean(robustness_scores > th)} ({np.count_nonzero(robustness_scores > th)})')

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

        burg.visualization.show_grasp_set([mesh], grasps, gripper=burg.gripper.ParallelJawGripper(),
                                          with_plane=True, score_color_func=lambda s: [1-s, s, 0], n=3000)

    print('bye')
