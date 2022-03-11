import numpy as np
from .gripper_base import GripperBase


class GripperEZGripper(GripperBase):
    def __init__(self, simulator, gripper_size=1.0):
        r""" Initialization of EZgripper
        specific args for EZgripper:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.1805 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        
        # define force and speed (grasping)
        self._force = 500
        self._grasp_speed = 1.5

        finger1_joint_ids = [2, 3]
        finger2_joint_ids = [4, 5]
        self._finger_joint_ids = [finger1_joint_ids, finger2_joint_ids]
        self._driver_joint_id = 2
        self._follower_joint_ids = [3, 4, 5]
        self._joint_ids = [self._driver_joint_id] + self._follower_joint_ids

        self._joint_lower = 0.8
        self._joint_upper = 1.9

    def load(self, position, orientation, open_scale=1.0):
        if open_scale != 1.0:
            raise NotImplementedError('robotiq 2f gripper can only be loaded in a fully open state (open_scale=1.0)')
        gripper_urdf = self.get_asset_path('ezgripper/model.urdf')
        body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        self._body_id = body_id
        return body_id

    def configure(self):
        # Set friction coefficients for gripper fingers
        for i in range(self._bullet_client.getNumJoints(self.body_id)):
            self._bullet_client.changeDynamics(self.body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                                               rollingFriction=0.0001, frictionAnchor=True)
        self._sim.register_step_func(self.step_constraints)

    def step_constraints(self):
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[1.9-pos, pos, 1.9-pos],
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids)
        )
        return pos

    def open(self, open_scale=1.0):
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'
        # open_scale = np.clip(open_scale, 0.1, 1.0)

        # larger joint position corresponds to smaller open width
        open_pos = open_scale*self._joint_lower + (1-open_scale)*self._joint_upper
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=open_pos,
            force=self._force
        )
        self._sim.step(seconds=2)

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force
        )
        self._sim.step(seconds=2)

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_contact_link_ids(self):
        # todo: this is tricky
        # we have two finger links on each side
        # obviously we want to have some contact from each side
        return self._joint_ids

    def get_vis_pts(self, open_scale):
        width = 0.09 * np.sin(open_scale)
        return self._gripper_size * np.array([
            [-width, 0],
            [width, 0]
        ])
