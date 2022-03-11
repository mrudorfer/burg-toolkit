import numpy as np
from .gripper_base import GripperBase
import time
import threading


class GripperRobotiq2F85(GripperBase):
    def __init__(self, simulator, gripper_size=1.0):
        r""" Initialization of robotiq-2f-85 gripper
        specific args for robotiq-2f-85:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.165 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi / 2])
        
        # define force and speed (grasping)
        self._force = 300
        self._grasp_speed = 1
        self._n_seconds = 4

        # define driver joint; the follower joints need to satisfy constraints when grasping
        self._driver_joint_id = 5
        self._driver_joint_lower = 0
        self._driver_joint_upper = 0.8
        self._follower_joint_ids = [0, 2, 7, 4, 9]
        self._follower_joint_sign = [1, -1, -1, 1, 1]

    def load(self, position, orientation, open_scale=1.0):
        gripper_urdf = self.get_asset_path('robotiq_2f_85/model.urdf')
        body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        if open_scale != 1.0:
            raise NotImplementedError('robotiq 2f gripper can only be loaded in a fully open state (open_scale=1.0)')

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
        targets = pos * np.array(self._follower_joint_sign)
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            [joint_id for joint_id in self._follower_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=targets,
            forces=[self._force] * len(self._follower_joint_ids),
            positionGains=[1] * len(self._follower_joint_ids)
        )
        return pos

    def open(self, open_scale=1.0):
        open_scale = np.clip(open_scale, 0.1, 1.0)
        # recalculate scale because larger joint position corresponds to smaller open width
        target_pos = open_scale*self._driver_joint_lower + (1-open_scale)*self._driver_joint_upper
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_pos,
            force=self._force
        )
        self._sim.step(seconds=2)
        return
        n_steps = int(2 / self._sim.dt)  # 2 seconds
        for i in range(n_steps):
            driver_pos = self.step_constraints()
            if np.abs(driver_pos - target_pos) < 1e-5:
                break
            self._sim.step()

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force
        )
        self._sim.step(seconds=4)
        return
        n_steps = int(4 / self._sim.dt)  # seconds
        for i in range(n_steps):
            driver_pos = self.step_constraints()
            if driver_pos > self._driver_joint_upper:
                break
            self._sim.step()

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_vis_pts(self, open_scale):
        width = 0.05 * np.sin(open_scale)
        return self._gripper_size * np.array([
            [-width, 0],
            [width, 0]
        ])
