

class ParallelJawGripper:
    """
    Represents a parallel jawed gripper.
    Fingers are assumed to be cuboids with same width and height (the `finger_thickness`) and a specified
    `finger_length`.
    The inside of the fingers are at most `opening_width` apart.
    All values are given in meters, defaults are arbitrarily chosen values.

    :param finger_length: Length of the fingers.
    :param opening_width: Maximum distance between both fingers.
    :param finger_thickness: Side-lengths of the fingers.
    :param mesh: A mesh representation of the gripper.
    """

    def __init__(self, finger_length=0.05, opening_width=0.08, finger_thickness=0.01, mesh=None):
        self.finger_length = finger_length
        self.opening_width = opening_width
        self.finger_thickness = finger_thickness
        self._mesh = mesh

    def _create_simplified_mesh(self):
        print('WARNING: simplified mesh not implemented yet. Mesh is', self._mesh)

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
        # todo:
        #  maybe we should make a deep copy?
        self._mesh = mesh
