import sys
import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
python_versions = '>=3.6, <3.9'  # restricted by availability of open3d 0.12.0

requirements_default = [
    'numpy',       # for all datastructures
    'scipy',       # kdtrees (might check out those from open3d instead)
    'matplotlib',  # vis
    'h5py',        # dataset io
    'mat73',       # interface to matlab scene generation tool
    'numba',       # speedup for numpy-quaternion
    'numpy-quaternion',  # numpy integration for quaternions
    'configparser',  # parsing configuration files
    'tqdm',         # progress bars
    'open3d==0.12.0',      # point clouds and processing
    'trimesh[easy]',  # this works on windows and linux, as opposed to trimesh[all]
    'pybullet'        # for the simulation module
]

if sys.platform == 'linux':
    requirements_default.append('python-fcl')  # for collision checks with trimesh, linux only
else:
    # there are some efforts to bring python-fcl to Windows, e.g. there are packages like python-fcl-win32 and
    # python-fcl-win32-nr, but there seem to be issues installing those, see:
    # https://github.com/BerkeleyAutomation/python-fcl/issues/17
    # (i have also not been able to install either of those packages with pip)
    print('Platform does not support collision checks with python-fcl')

requirements_docs = [
    'Sphinx',      # tool for creating docs
    'm2r2'         # for automatically parsing the python modules
]

# merge requirements and remove duplicates
reqs_all = list(set(requirements_default + requirements_docs))

setuptools.setup(
    name='BURG-toolkit',
    version='0.1',
    python_requires=python_versions,
    install_requires=reqs_all,
    packages=setuptools.find_packages(),
    url='',
    license='',
    author='Martin Rudorfer',
    author_email='m.rudorfer@bham.ac.uk',
    description='toolkit for benchmarking and understanding robotic grasping',
    long_description=long_description
)
