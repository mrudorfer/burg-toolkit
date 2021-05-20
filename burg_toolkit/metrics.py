import numpy as np
from scipy import spatial

from . import grasp


def pairwise_euclidean_distances(graspset1, graspset2):
    """
    Computes the pairwise euclidean distances of the positions of the provided grasps from set1 and set2.
    This is a vectorized implementation, but should be used in chunks if both sets are extremely large.

    :param graspset1: GraspSet of length N (or single Grasp)
    :param graspset2: GraspSet of length M (or single Grasp)

    :return: (N, M) ndarray of euclidean distances
    """
    if type(graspset1) is grasp.Grasp:
        graspset1 = graspset1.as_grasp_set()
    if type(graspset2) is grasp.Grasp:
        graspset2 = graspset2.as_grasp_set()

    # shape: (N, M, 3) = (N, 1, 3) - (1, M, 3)
    distances = np.linalg.norm(
        graspset1.translations[:, np.newaxis, :] - graspset2.translations[:, 0:3][np.newaxis, :, :], axis=-1)
    return distances


def pairwise_angular_distances(graspset1, graspset2):
    """
    Computes the pairwise angular distances in rad of the orientations of the provided grasps from set1 and set2.
    This is a vectorized implementation, but should be used in chunks if both sets are extremely large.

    :param graspset1: GraspSet of length N (or single Grasp)
    :param graspset2: GraspSet of length M (or single Grasp)

    :return: (N, M) ndarray of angular distances
    """
    if type(graspset1) is grasp.Grasp:
        graspset1 = graspset1.as_grasp_set()
    if type(graspset2) is grasp.Grasp:
        graspset2 = graspset2.as_grasp_set()

    # see http://boris-belousov.net/2016/12/01/quat-dist/ for explanation of basic formula
    # basic formula: arccos((tr(R1*R2.T)-1)/2)
    # we just have to vectorize it to compute pairwise distances
    distances = np.arccos(
            (np.trace(
                        np.matmul(
                            # extend the dimensions to (N, 1, 3, 3) and (1, M, 3, 3) for the pairwise computations
                            graspset1.rotation_matrices[:, np.newaxis, ...],
                            np.transpose(graspset2.rotation_matrices, axes=(0, 2, 1))[np.newaxis, ...]
                        ),
                        axis1=-2, axis2=-1  # compute trace over last two dims
                    ) - 1.0) / 2.0)
    return distances


def pairwise_combined_distances(graspset1, graspset2, weight=1000.0):
    """
    Computes the pairwisse combined distances between the provided grasps from set1 and set2.
    The combined distance is `weight * euclidean distance [m] + angular_distance [deg]`.

    :param graspset1: GraspSet of length N (or single Grasp)
    :param graspset2: GraspSet of length M (or single Grasp)
    :param weight: weight factor for the euclidean distance, defaults to 1000 so 1mm=1degree

    :return: (N, M) matrix of combined distances
    """
    combined_distances = weight * pairwise_euclidean_distances(graspset1, graspset2) + \
        np.rad2deg(pairwise_angular_distances(graspset1, graspset2))

    return combined_distances


def coverage_brute_force(reference_grasp_set, query_grasp_set, epsilon=15.0):
    """
    Computes coverage, i.e. fraction of grasps from the reference grasp set which are covered by a grasp of query set
    corresponds to coverage_1 from Eppner et al., 2019.

    This brute force variant computes the full distance matrix and then checks for the threshold. the vectorized
    implementation is relatively fast but it quickly runs into memory limitations when sets become larger.

    :param reference_grasp_set: the grasp set to be covered
    :param query_grasp_set: the grasp set which shall cover the reference grasp set
    :param epsilon: the tolerance threshold used in the distance function - a grasp from the reference set will be
                    considered as covered, if its distance to the closest query grasp does not exceed epsilon

    :return: float in [0, 1] corresponding to the fraction of grasps in the reference grasp set which are covered by
             grasps from the query grasp set
    """
    cov = np.any(pairwise_combined_distances(reference_grasp_set, query_grasp_set) <= epsilon, axis=1)
    return np.count_nonzero(cov) / len(cov)


def coverage(reference_grasp_set, query_grasp_set, epsilon=15.0, print_timings=False):
    """
    Computes coverage, i.e. fraction of grasps from the reference grasp set which are covered by a grasp of query set
    corresponds to coverage_1 from Eppner et al., 2019.

    This implementation uses kd-trees and can thus handle larger grasp sets, although it may take some time.

    :param reference_grasp_set: the grasp set to be covered
    :param query_grasp_set: the grasp set which shall cover the reference grasp set
    :param epsilon: the tolerance threshold used in the distance function - a grasp from the reference set will be
                    considered as covered, if its distance to the closest query grasp does not exceed epsilon
    :param print_timings: does print timing information if set to True

    :return: float in [0, 1] corresponding to the fraction of grasps in the reference grasp set which are covered
             by grasps from the query grasp set
    """
    # so we have to consider that both grasp sets can be very large, computing a brute-force distance matrix
    # can quickly exceed the available RAM
    # Eppner et al. used an algorithm by Ichnowski et al. which performs nearest neighbor search in SE(3) with
    # quaternions.. see the paper here: http://dx.doi.org/10.1007/978-3-319-16595-0_12
    # however, i could not find any code available for this. so i went with scipy's kd tree methods

    # variant 1:
    # 1) construct kdtree for query grasp set translations
    # 2) query ball point with reference grasp translations --> this step requires 99.99% of the time
    # 3) go through resulting array and check actual distances

    # variant 2:
    # 1) construct kdtree for query grasp set translations
    # 1a) construct kdtree for ref grasp set translations
    # 2) query ball tree
    # 3) go through resulting list and check actual distances

    # i tested both variants with 2.6M ref grasp set and 500 query grasp set
    # var 1 took 380-450 seconds, with 2) being the most time-consuming step
    # var 2 took 270 seconds (only one run), with 1a) being most time-consuming step and greatly accelerating 2)
    # so i stick with variant 2

    if print_timings:
        from timeit import default_timer as timer
        t1 = timer()

    ref_tree = spatial.KDTree(reference_grasp_set.translations)
    query_tree = spatial.KDTree(query_grasp_set.translations)

    if print_timings:
        t2 = timer()

    list_of_indices = ref_tree.query_ball_tree(query_tree, r=epsilon/1000)

    if print_timings:
        t3 = timer()

    # although we have to loop through the whole list, this step is fairly quick because each iteration only
    # considers very few candidates
    covered = 0
    for i, indices in enumerate(list_of_indices):
        if len(indices) > 0:
            if np.any(pairwise_combined_distances(reference_grasp_set[i], query_grasp_set[indices]) <= epsilon):
                covered += 1

    if print_timings:
        t4 = timer()
        print('it took:')
        print(t2-t1, 'seconds to construct the trees')
        print(t3-t2, 'seconds to query the tree and get the array of indices based on translations')
        print(t4-t3, 'seconds to loop through the remaining candidates')

    return covered / len(reference_grasp_set)
