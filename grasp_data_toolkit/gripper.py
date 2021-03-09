import enum

import numpy as np
import open3d as o3d

from . import util

class RefFrame(enum.Enum):
    """
    Provides different modes of defining the gripper reference frame.
    STEM: origin in the center of the stem, finger tips pointing along z-axis, closing along x-axis.
    TCP: origin inbetween the finger tips, closing along x-axis, z-axis pointing towards the stem
    """
    STEM = 0
    TCP = 1

class ParallelJawGripper:
    """
    Represents a parallel jawed gripper.
    Fingers are assumed to be cuboids with same width and height (the `finger_thickness`) and a specified
    `finger_length`.
    The inside of the fingers are at most `opening_width` apart.
    All values are given in meters, defaults are arbitrarily chosen values.
    The reference point is in

    :param finger_length: Length of the fingers.
    :param opening_width: Maximum distance between both fingers.
    :param finger_thickness: Side-lengths of the fingers.
    :param mesh: A mesh representation of the gripper.
    :param ref_frame: An enum from class RefFrame defining the reference frame of the gripper
    """

    def __init__(self, finger_length=0.05, opening_width=0.08, finger_thickness=0.01, mesh=None,
                 ref_frame=RefFrame.STEM):
        self.finger_length = finger_length
        self.opening_width = opening_width
        self.finger_thickness = finger_thickness
        self._mesh = mesh
        self.ref_frame = ref_frame

    def _create_simplified_mesh(self):
        """
        Creates a simple gripper mesh, consisting of the two fingers and a stem or bridge connecting them.

        :return: Nothing, just sets the created mesh internally.
        """
        # boxes spawn with left, front, bottom corner at 0, 0, 0
        if self.ref_frame == RefFrame.STEM:
            finger1 = o3d.geometry.TriangleMesh.create_box(
                self.finger_thickness, self.finger_thickness, self.finger_length)
            finger2 = o3d.geometry.TriangleMesh(finger1)
            finger1.translate(np.array([-self.finger_thickness - self.opening_width/2, -self.finger_thickness/2, 0]))
            finger2.translate(np.array([self.opening_width/2, -self.finger_thickness/2, 0]))

            stem = o3d.geometry.TriangleMesh.create_box(
                self.opening_width + 2*self.finger_thickness, self.finger_thickness, self.finger_thickness)
            stem.translate(np.array([-self.finger_thickness - self.opening_width/2, -self.finger_thickness/2,
                                     -self.finger_thickness]))

        elif self.ref_frame == RefFrame.TCP:
            finger1 = o3d.geometry.TriangleMesh.create_box(
                self.finger_thickness, self.finger_thickness, self.finger_length)
            finger2 = o3d.geometry.TriangleMesh(finger1)
            finger1.translate(np.array([-self.finger_thickness - self.opening_width/2, -self.finger_thickness/2, 0]))
            finger2.translate(np.array([self.opening_width/2, -self.finger_thickness/2, 0]))

            stem = o3d.geometry.TriangleMesh.create_box(
                self.opening_width + 2 * self.finger_thickness, self.finger_thickness, self.finger_thickness)
            stem.translate(np.array([-self.finger_thickness - self.opening_width / 2, -self.finger_thickness / 2,
                                     self.finger_length]))

        self._mesh = util.merge_o3d_triangle_meshes([finger1, finger2, stem])


    @property
    def mesh(self):
        """
        The mesh representation of this gripper. If gripper has none, a simplified mesh will be created
        based on the dimensions of the gripper.
        """
        if self._mesh is None:
            self._create_simplified_mesh()
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        # maybe we should make a copy?
        self._mesh = mesh
