import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements_default = [
    'numpy',       # for all datastructures
    'scipy',       # kdtrees (might check out those from open3d instead)
    'matplotlib',  # vis
    'h5py',        # dataset io
    'mat73',       # interface to matlab scene generation tool
    'numba',       # speedup for numpy-quaternion
    'numpy-quaternion',  # numpy integration for quaternions
    'configparser',  # parsing configuration files
    'open3d>=0.12.0',      # point clouds and processing
    'pymeshlab',     # only used for point cloud sampling currently - will be eliminated at some point
    'trimesh[easy]'  # for collision detections, which might get incorporated in newer version of open3d
]

requirements_docs = [
    'Sphinx',      # tool for creating docs
    'm2r2'         # for automatically parsing the python modules
]

# merge requirements and remove duplicates
reqs_all = list(set(requirements_default + requirements_docs))

setuptools.setup(
    name='BURG-toolkit',
    version='0.1',
    packages=setuptools.find_packages(),
    setup_requires=['setuptools', 'wheel'],
    install_requires=reqs_all,
    url='',
    license='',
    author='Martin Rudorfer',
    author_email='m.rudorfer@bham.ac.uk',
    description='toolkit for benchmarking and understanding robotic grasping',
    long_description=long_description,
    python_requires='>=3.6'
)
