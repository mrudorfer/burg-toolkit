# grasp data toolkit

This becomes a toolkit for benchmarking and evaluating methods for robotic grasping.


## todos:
- move to gitlab
- paint GT point clouds according to segmentation (what about partial?)
- have a script to export (segmented, partial) point clouds

## longer term todos:
- move all point clouds to o3d
    - in object_library we could already keep the o3d point clouds, which should save some processing
    - we could also save the point clouds to files, which saves some waiting time during each run
- poisson disk sampling:
    - point densities are not uniform, instead it is relative to the size of the object
    - i think uniform density would be better, but we can skip some part of the table
