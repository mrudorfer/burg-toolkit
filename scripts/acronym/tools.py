import os
from collections import defaultdict


def identify_shapes_and_categories(acronym_dir):
    """
    In acronym/grasps folder, there are .h5 files with nomenclature: `category_shape_scale.h5`.
    We simply parse the categories and shape ids from those filenames.

    :param acronym_dir: base acronym directory

    :return: dict with categories as keys and list of shape ids as values
    """
    shapes_by_category = defaultdict(list)
    grasp_annotation_dir = os.path.join(acronym_dir, 'grasps')
    filenames = os.listdir(grasp_annotation_dir)
    for filename in filenames:
        elems = filename.split('_')
        category = elems[0]
        shape = elems[1]
        shapes_by_category[category].append(shape)

    return shapes_by_category


def print_shapes_by_categories(shapes_by_categories, with_values=False):
    """
    Use this function to print the `shapes_by_category` dict returned from `identify_shapes_and_categories()`.
    """
    if len(shapes_by_categories) == 0:
        print('\tnone')
        return

    total = 0
    for cat, value in shapes_by_categories.items():
        print(f'\t{cat}: {len(value)} {value if with_values else ""}')
        total += len(value)
    print(f'\ttotal: {total} (from {len(shapes_by_categories.keys())} categories)')

