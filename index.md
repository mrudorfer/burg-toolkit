# BURG Tools
We present an open source toolset for **B**enchmarking and **U**nderstanding **R**obotic **G**rasping developed in the scope of the [BURG project](#references) to support research and development of methods for robotic grasping. 

The main contribution of our tools are:
- easy creation of virtual scenes, generation of training data and performing grasping in simulation
- support for arrangement of objects in the real world to accurately re-create those scenes for real robot experiments
- share scene compositions with other researchers to foster comparability and reproducibility of experimental results

The set of tools include the following components:
- [BURG-Toolkit](#burg-toolkit)
- [SetupTool](#setuptool)
- [SceneVisualizer](#scenevisualizer)

## BURG-Toolkit

The Python [BURG Toolkit](https://github.com/mrudorfer/burg-toolkit) is the core component of the BURG Tools. 
It offers the following features:
- core data structures for object types and instances, scenes, grippers, grasps and grasp sets
- antipodal grasp sampling
- physics simulation with pybullet
- depth/point cloud rendering with pybullet and/or pyrender
- metrics for evaluation of grasps and grasp sets
- visualization of scenes and grasps using open3d
- dataset creation
- printout generation
- generating stable poses of the objects used in virtual scene compositions
- collision and boundary checking of objects in virtual scenes

## SetupTool

The [SetupTool](https://github.com/markus-suchi/burg-setuptool) is a graphical front end of the BURG-Toolkit dedicated to virtual scene creation. It is implemented as a Blender AddOn and handles the following tasks:
- creation of virtual scenes
- loading/saving virtual scenes
- visualization of collision and out of bound checks of individual objects
- saving of printout sheets of virtual scenes

## SceneVisualizer


## Acknowledgments

This work was conducted within the BURG research project for Benchmarking and Understanding Robotic Grasping. 
It is supported by CHIST-ERA and EPSRC grant no. EP/S032487/1, FWF (grant no. I3967-N30).
## References

### research

- BURG research project: https://burg.acin.tuwien.ac.at/

