"""
BURG toolkit

Benchmarking and Understanding Robotic Grasping
"""

# directly import frequently used data structures (shortens names during usage)
from .core import *
from .grasp import *

# import all other modules
from . import constants, gripper, io, mesh_processing, metrics, render, sampling, sim, scene_sim, util, visualization
from . import printout

import logging
# todo: setting debug level for now, will have to change this so applications can override it
# see https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.basicConfig(level=logging.DEBUG)

