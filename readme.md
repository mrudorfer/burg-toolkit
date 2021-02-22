# grasp data toolkit

This becomes a Python toolkit for benchmarking and evaluating methods for robotic grasping. Features are:
- visualize scenes generated with the sceneGeneration_MATLAB project
- visualize dense ground truth grasps from [Eppner et al., 2019](#references)
- algorithms for grasp sampling (future)
- evaluate grasp similarity as well as grasp coverage of sampler w.r.t. to the ground truth set

## project structure

The project contains the following directories:
- **docs** - configuration files to create documentation with sphinx
- **grasp_data_toolkit** - the core Python library, used for io, mesh and point cloud processing, data visualization, etc.
- **scripts** - entry points, scripts for exploring the data, compiling datasets, evaluation
- **config** - configuration files, specifying e.g. important paths, which meshes to use, scale factors, etc.

## installation

In the future we might provide a `setup.py` file. For now, install dependencies by running:

``
pip install -r requirements.txt
``

and use the package like this:

```
import numpy as np
import grasp_data_toolkit as gdt

gs1 = gdt.grasp.GraspSet.from_translations(np.random.random(50, 3))
gs2 = gdt.grasp.GraspSet.from_translations(np.random.random(30, 3))
print('grasp coverage is:', gdt.grasp.coverage(gs1, gs2))
```

see the scripts for more examples on usage and the docs for more detailed specifications.

you should be able to build the documentation on the command line via:

```
cd docs/
make html
```

## plans for the project
### todos
- improve PPF grasp sampler
- implement metrics (precision) for comparison with eppner2019
- implement metrics (force-closure) from fang2020
- have a script to export (segmented, partial) point clouds

### longer-term todos:
- move all point clouds to o3d
    - in object_library we could already keep the o3d point clouds, which should save some processing
    - we could also save the point clouds to files, which saves some waiting time during each run
    - however, they're more flexible as numpy array and if we need to include trimesh we would have a hazzle to
      switch from the different o3d/trimesh object instances
- restructure object library
    - currently, objects have Type and Instance classes, but background obejcts are treated differently, which
      is somewhat inconvenient
    - also, object library index is based on the order of the objects in the array, which is not ideal
- poisson disk sampling:
    - point densities are not uniform, instead it is relative to the size of the object
    - i think uniform density would be better, but we can skip some part of the table
    - o3d has a voxel-based down-sample method - maybe that could be an approach?


## References

- Clemens Eppner, Arsalan Mousavian and Dieter Fox: "A Billion Ways to Grasps - An Evaluation of Grasp Sampling Schemes on a Dense, Physics-based Grasp Data Set", ISRR 2019 - https://sites.google.com/view/abillionwaystograsp
- Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun: "Open3D: A modern library for 3D data processing", 2018 - http://www.open3d.org/
- Alessandro Muntoni and Paolo Cignoni: "PyMeshLab", 2021 - https://doi.org/10.5281/zenodo.4438750
