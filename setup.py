import setuptools
import itertools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
python_versions = '>=3.6, <3.9'  # restricted by availability of open3d 0.12.0

requirements_default = [
    'numpy',       # for all datastructures
    'scipy',       # kdtrees (might check out those from open3d instead), rotations
    'matplotlib',  # vis
    'h5py',        # dataset io
    'pyyaml',      # object library io
    'numba',       # speedup for numpy-quaternion
    'numpy-quaternion',  # numpy integration for quaternions
    'tqdm',         # progress bars
    'opencv-contrib-python',  # image processing
    'open3d==0.12.0',      # point clouds and processing
    'trimesh[easy]',  # this works on windows and linux, as opposed to trimesh[all]
    'pyrender',       # rendering
    'pybullet',       # for the simulation module
    'fpdf',           # creating pdf files
]

extras_require = {
    'docs': [
        'Sphinx',  # tool for creating docs
        'm2r2'    # for automatically parsing the python modules
    ],
    'collision': [
        'python-fcl'   # collision checks with trimesh, on linux only
    ],
    'openexr': [
        'pyexr'     # relies on openexr, which must be manually installed on the system prior to pip install
    ]
}

# also create a full installation with all extras
extras_require['full'] = set(itertools.chain.from_iterable(extras_require.values()))

setuptools.setup(
    name='BURG-toolkit',
    version='0.2.0',
    python_requires=python_versions,
    install_requires=requirements_default,
    extras_require=extras_require,
    packages=setuptools.find_packages(),
    url='https://github.com/mrudorfer/burg-toolkit',
    license='',
    author='Martin Rudorfer',
    author_email='mail@ketza.de',
    description='toolkit for benchmarking and understanding robotic grasping',
    long_description=long_description
)
