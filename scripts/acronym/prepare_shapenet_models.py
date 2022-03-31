"""
This script prepares the ShapeNetSem models (https://shapenet.org/) for using them within in the
ACRONYM dataset (https://github.com/NVlabs/acronym) for robotic grasping.
We assume you have downloaded both ACRONYM and ShapeNetSem and extracted ACRONYM.
Furthermore, you must have manifold and simplify downloaded from https://github.com/hjwdzh/Manifold and built.

The script will look in the acronym/grasps directory to identify categories and shapes, extract the required shapes
from ShapeNetSem, perform manifold and simplify and then put each shape to acronym/meshes/category/shape.obj.
It creates a temporary directory in acronym/tmp which should be empty after successful completion of the script.
In total, it took almost 10hrs on my machine to run through all the steps.
"""

import argparse
import zipfile
import os
import pathlib
import shutil
import subprocess
from functools import partial
from collections import defaultdict

import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map

from tools import identify_shapes_and_categories, print_shapes_by_categories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapenet_core_zipfile', type=str,
                        default='/home/rudorfem/datasets/ShapeNet/ShapeNetCore.v2.zip',
                        help='path to shapenet core zipfile')
    parser.add_argument('--shapenet_sem_zipfile', type=str,
                        default='/home/rudorfem/datasets/ShapeNet/ShapeNetSem-models-OBJ.zip',
                        help='path to shapenet sem zipfile')
    parser.add_argument('--acronym_dir', type=str, default='/home/rudorfem/datasets/acronym/')
    parser.add_argument('--manifold_dir', type=str, default='/home/rudorfem/dev/Manifold/build')
    return parser.parse_args()


def extract_shapes(shapes, shapenet_zipfile, target_dir):
    """
    Looks in shapenet_zipfile for the identified shapes. Will unzip the obj file and put it into target_dir.
    Returns a list with remaining shapes.
    It has a bit of overhead, but works as well with ShapeNetCore (if necessary).
    """
    remaining_shapes = []
    archive = zipfile.ZipFile(shapenet_zipfile)
    filenames = archive.namelist()
    filenames = [name for name in filenames if name.endswith('.obj')]
    for shape in tqdm.tqdm(shapes):
        files = [name for name in filenames if shape in name]
        if len(files) == 0:
            # nothing found, note shape
            remaining_shapes.append(shape)
        else:
            # ShapeNetSem has unique identifiers, in ShapeNetCore they can occur multiple times
            # multiples seem identical though, just happen to be in different categories (based on n=1 lol)
            # just use first file and extract contents to the target dir
            with archive.open(files[0]) as file:
                contents = file.read()
            target_fn = os.path.join(target_dir, f'{shape}.obj')
            with open(target_fn, 'wb') as file:
                file.write(contents)
    return remaining_shapes


def filter_shapes_and_categories(shapes_by_category, cats_to_delete):
    """
    use this to remove categories from the dict
    """
    for cat in cats_to_delete:
        r = shapes_by_category.pop(cat)
        if r is not None:
            print(f'\tremoved {len(r)} {cat} objects')
    print('\tdone.')
    return shapes_by_category


def process_manifold(filename, manifold_bin, simplify_bin, source_dir, target_dir, use_s_flag=True):
    """
    This performs the actual manifold and simplify operations:
        `manifold source_dir/model.obj target_dir/temp.model.obj -s`
        `simplify -i target_dir/temp.model.obj -o target_dir/model.obj -m -r 0.02`

    However, manifold fails for some shapes. I could find reports of this issue, but no reported solutions. I found that
    omitting the -s flag gives a plausible result. This is some `g_sharp` parameter in the implementation, but I could
    not find any mention of what exactly it is doing.

    Anyhow, this method will first try using the -s option and if it fails it proceeds without, and only reports
    an error if that fails as well.

    The method cleans up after itself, i.e. it deletes both source_dir/model.obj and target_dir/temp.model.obj (but
    only if successful).

    :param filename: Filename (no path) of the model.obj
    :param manifold_bin: Path to manifold bin
    :param simplify_bin: Path to simplify bin
    :param source_dir: Path where to find model.obj
    :param target_dir: Path where to put temp.model.obj and final model.obj
    :param use_s_flag: Whether to use the flag. If True and does not work, will fall back to False.

    :return: Returns 0 if all went well, returns 1 if failed.
    """
    # manifold
    fn = os.path.join(source_dir, filename)
    tmp_fn = os.path.join(target_dir, 'temp.' + filename)
    command = [manifold_bin, fn, tmp_fn]
    if use_s_flag:
        command.append('-s')
    cp_m = subprocess.run(command, capture_output=True)
    if cp_m.returncode != 0:
        if use_s_flag:
            # print(f'manifold failed in 1st attempt for {fn}')
            return process_manifold(filename, manifold_bin, simplify_bin, source_dir, target_dir, use_s_flag=False)
        else:
            print(f'manifold failed in 2nd attempt for {fn}')
            # print('manifold stdout', cp_m.stdout)
            # print('manifold stderr', cp_m.stderr)
            return 1

    # simplify
    target_fn = os.path.join(target_dir, filename)
    cp_s = subprocess.run([simplify_bin, '-i', tmp_fn, '-o', target_fn, '-m', '-r', '0.02'], capture_output=True)
    if cp_s.returncode != 0:
        if use_s_flag:
            # print(f'simplify failed in 1st attempt for {fn}')
            return process_manifold(filename, manifold_bin, simplify_bin, source_dir, target_dir, use_s_flag=False)
        else:
            print(f'simplify failed in 2nd attempt for {fn}')
            # print('manifold stdout', cp_m.stdout)
            # print('manifold stderr', cp_m.stderr)
            # print('simplify stdout', cp_s.stdout)
            # print('simplify stderr', cp_s.stderr)
            return 1

    # successful, so clean up
    pathlib.Path(fn).unlink(missing_ok=True)
    pathlib.Path(tmp_fn).unlink(missing_ok=True)
    return 0


def manifold_and_simplify(manifold_dir, source_dir, target_dir):
    """
    This method will invoke processing of all files from source_dir with manifold and simplify.

    :param manifold_dir: directory where to find the manifold/simplify binaries
    :param source_dir: directory with all the .obj files (and only those), will be emptied
    :param target_dir: directory where to put the final processed obj files
    """
    manifold_bin = os.path.join(manifold_dir, 'manifold')
    simplify_bin = os.path.join(manifold_dir, 'simplify')
    if not os.path.isfile(manifold_bin):
        raise FileNotFoundError('manifold bin is not a file: ' + manifold_bin)
    if not os.path.isfile(simplify_bin):
        raise FileNotFoundError('simplify bin is not a file: ' + simplify_bin)

    # process all files concurrently (from tqdm.contrib.concurrent, see https://tqdm.github.io/docs/contrib.concurrent/)
    files = os.listdir(source_dir)
    result = process_map(
        partial(process_manifold, manifold_bin=manifold_bin, simplify_bin=simplify_bin, source_dir=source_dir,
                target_dir=target_dir),
        files, chunksize=10)

    errored = []
    for ret_val, file in zip(result, files):
        if ret_val != 0:
            errored.append(file)
    return errored


def populate_target_dir(shapes_by_category, source_dir, acronym_dir):
    """
    This function will move files from source dir to target dir, creating subdirectories in target dir according to
    the shape categories.
    """
    # some objects may belong to multiple categories, hence we first copy all and only after we have gathered all
    # files we will remove the files from source_dir
    missing = []
    for cat, shapes in shapes_by_category.items():
        cat_dir = os.path.join(acronym_dir, 'meshes', cat)
        pathlib.Path(cat_dir).mkdir(parents=True, exist_ok=True)

        for shape in shapes:
            fn = os.path.join(source_dir, f'{shape}.obj')
            if not os.path.isfile(fn):
                missing.append(shape)
                continue

            shutil.copy(fn, os.path.join(cat_dir, f'{shape}.obj'))

    # now clean up
    for cat, shapes in shapes_by_category.items():
        for shape in shapes:
            fn = os.path.join(source_dir, f'{shape}.obj')
            pathlib.Path(fn).unlink(missing_ok=True)  # may have been deleted previously, that's ok
    return missing


def filter_extracted_shapes(shapes_by_cat, acronym_dir):
    """
    for each shape in shapes_by_cat, this method looks in the acronym_dir/meshes folder and check if a mesh is present.
    returns a dict of same structure as shapes_by_cat, containing only those shapes for which no mesh could be found.
    """
    missing_shapes_by_cat = defaultdict(list)
    mesh_dir = os.path.join(acronym_dir, 'meshes')
    for category, shapes in shapes_by_cat.items():
        cat_dir = os.path.join(mesh_dir, category)
        for shape in shapes:
            mesh_fn = os.path.join(cat_dir, f'{shape}.obj')
            if not os.path.isfile(mesh_fn):
                missing_shapes_by_cat[category].append(shape)

    return missing_shapes_by_cat


def main(args):
    shapes_by_cat = identify_shapes_and_categories(args.acronym_dir)
    print('based on target data structure, looking for following shapes by categories:')
    print_shapes_by_categories(shapes_by_cat)

    shapes_by_cat = filter_extracted_shapes(shapes_by_cat, args.acronym_dir)
    print('-'*10)
    print('comparing with the shapes that are already extracted, you are missing the following:')
    print_shapes_by_categories(shapes_by_cat)
    # for cat, shapes in shapes_by_cat.items():
    #     for shape in shapes:
    #         print(f'{cat}_{shape}')

    all_shapes = np.array([item for values in list(shapes_by_cat.values()) for item in values])
    unique_shapes, frequencies = np.unique(all_shapes, return_counts=True)
    print('-'*10)
    print('total number of annotated shapes:', all_shapes.shape[0])
    print('unique shapes', unique_shapes.shape[0])  # around 1000 non-unique ones!
    freqs, count = np.unique(frequencies, return_counts=True)
    for f, c in zip(freqs, count):
        print(f'\t{c} shapes occur {f} time(s)')
    print('multiple occurrences might be due to inclusion in multiple categories or due to multiple scales')

    # create temporary dir where to put the files
    unzip_dir = os.path.join(args.acronym_dir, 'tmp', 'unzipped')
    pathlib.Path(unzip_dir).mkdir(parents=True, exist_ok=True)
    print(f'created tmp dir for extracting shapes: {unzip_dir}')

    print('extracting shapes from ShapeNetSem')
    remaining = extract_shapes(unique_shapes, args.shapenet_sem_zipfile, unzip_dir)
    if remaining:
        print(f'could not find {len(remaining)} shapes in ShapeNetSem... something seems wrong')
    print('done.')

    tmp_dir = os.path.join(args.acronym_dir, 'tmp', 'processed')
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    print(f'created tmp dir for processing shapes: {tmp_dir}')

    print('using manifold and simplify...')
    print('in case the progress bar does not get updated, you can check whether tmp folders are being populated')
    print('files should move from tmp/unzipped to tmp/processed')
    failing = manifold_and_simplify(args.manifold_dir, unzip_dir, tmp_dir)
    if failing:
        print(f'could not process {len(failing)} shapes. failed ones are the following:')
        print(failing)

    print('populating target directories with processed meshes')
    populate_target_dir(shapes_by_cat, tmp_dir, args.acronym_dir)

    print('done. if all went well, the tmp directories should be empty. you can delete them.')
    print('manifold/simplify fails silently in rare cases. running this script again will show you if there are any '
          'shapes missing.')
    print('-' * 10)

    # the following only applied for the dataset of 6dof GraspNet, perhaps not for ACRONYM:
    # print('remember that the copied files are .obj files, whereas some annotations refer to .stl files.')
    # print('which one of the two is present should be checked during dataset loading.')


if __name__ == '__main__':
    main(parse_args())
