"""
BURG toolkit

Benchmarking and Understanding Robotic Grasping
"""
import logging

# directly import frequently used data structures (shortens names during usage)
from .core import StablePoses, ObjectType, ObjectInstance, ObjectLibrary, Scene
from .grasp import Grasp, GraspSet

# import all other modules
from . import constants, gripper, io, mesh_processing, metrics, render, sampling, sim, scene_sim, util, visualization
from . import printout, robots

logging.getLogger(__name__).addHandler(logging.NullHandler())


def log_to_console(level=logging.DEBUG):
    """
    This is for debug purposes, it adds a StreamHandler to the logger to output all logs from this package to console.
    Furthermore, it disables the propagation to the root logger so that messages do not appear twice.

    :param level: int, logging level
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    handler.setLevel(level)

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    logger.info('logging to console for debug purposes. propagation to root disabled.')
