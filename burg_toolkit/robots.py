import logging
import abc
import os

import numpy as np
from . import constants

_log = logging.getLogger(__name__)


class RobotBase(abc.ABC):
    """
    abstract base class for robots, allows to use trajectory planners from this module.
    load the robot in simulation in the init method and set the body id
    """
    def __init__(self, simulator):
        self._simulator = simulator
        self._body_id = None

    @property
    def body_id(self):
        return self._body_id

    @property
    @abc.abstractmethod
    def end_effector_link_id(self):
        pass

    @property
    def num_joints(self):
        return self._simulator.bullet_client.getNumJoints(self.body_id)

    def end_effector_pos(self):
        """ x, y, z cartesian position of end effector """
        pos, *_ = self._simulator.bullet_client.getLinkState(
            self.body_id,
            self.end_effector_link_id
        )
        return np.array(pos)

    def joint_pos(self):
        """ current joint positions of this robot """
        joints = list(range(self.num_joints))
        joint_states = self._simulator.bullet_client.getJointStates(self.body_id, joints)
        pos = [joint_states[joint][0] for joint in joints]
        return np.array(pos)

    def set_target_pos_and_vel(self, target_joint_pos, target_joint_vel):
        """ set target joint positions and target velocities for velocity control """
        joints = list(range(self.num_joints))
        self._simulator.bullet_client.setJointMotorControlArray(
            self.body_id,
            joints,
            self._simulator.bullet_client.VELOCITY_CONTROL,
            targetPositions=list(target_joint_pos),
            targetVelocities=list(target_joint_vel),
            forces=[500]*len(joints),
        )

    def execute_joint_trajectory(self, joint_trajectory):
        """
        Executes the commands from a JointTrajectory.

        :param joint_trajectory: JointTrajectory object

        :return: bool, True if all joints arrived at target configuration
        """
        start_time = self._simulator.simulated_seconds
        for time_step, dt, target_pos, target_vel in joint_trajectory:
            # set intermediate target
            self.set_target_pos_and_vel(target_pos, target_vel)

            # simulate until we reach next timestep
            step_end_time = start_time + time_step + dt
            while self._simulator.simulated_seconds < step_end_time:
                self._simulator.step()
            _log.debug(f'expected vs. actual joint pos after time step\n\t{target_pos}\n\t{self.joint_pos()}')

        # should have arrived at the final target now, check if true
        arrived = joint_trajectory.joint_pos[-1] - self.joint_pos() < 0.001
        if np.all(arrived):
            _log.debug('finished trajectory execution. arrived at goal position.')
            return True
        else:
            _log.warning(f'trajectory execution terminated but target configuration not reached:'
                         f'\n\tjoint pos diff is: {joint_trajectory.joint_pos[-1] - self.joint_pos()}'
                         f'\n\tend-effector pos is: {self.end_effector_pos()}')
            return False


class XYZRobot(RobotBase):
    """
    This is a simple robot with three PRISMATIC axes with a range of [-1, 1].
    Use this if you only want translational motion, no change of orientation is possible.
    """
    def __init__(self, simulator, position, orientation):
        super().__init__(simulator)

        robot_urdf = os.path.join(constants.ASSET_PATH, 'dummy_xyz_robot.urdf')
        self._body_id, self.robot_joints = self._simulator.load_robot(robot_urdf, position=position,
                                                                      orientation=orientation, fixed_base=True)

        # control all joints to stay at 0 with high force, in order to stay at the same position
        self._simulator.bullet_client.setJointMotorControlArray(
            self.body_id,
            list(range(self.num_joints)),
            self._simulator.bullet_client.POSITION_CONTROL,
            targetPositions=[0] * self.num_joints,
            forces=[1000] * self.num_joints
        )

    @property
    def end_effector_link_id(self):
        return self.robot_joints['end_effector_link']['id']


class JointTrajectory:
    """
    This represents a robot trajectory in joint space.
    After initialising the trajectory with time steps and joint positions, the dt (distance between time steps) and
    joint velocities are computed.
    You can use `n = len(trajectory)` and `time, dt, pos, vel = trajectory[i]` to access data from individual time
    steps.

    :param time_steps: ndarray (n,) time steps from start time 0 to end time T_e
    :param joint_pos: ndarray (n, n_j) target joint positions that shall be reached with next time step
    """
    def __init__(self, time_steps, joint_pos):
        # pass some checks
        assert len(time_steps) == len(joint_pos), 'time steps and joint pos must have same length'
        for i in range(1, len(time_steps)):
            assert time_steps[i] > time_steps[i-1], 'time steps must be strictly monotonically increasing'

        self._time_steps = time_steps
        self._joint_pos = joint_pos
        self._dt = self._compute_time_step_differences()
        self._joint_vel = self._compute_joint_velocities()

    def __len__(self):
        return len(self.time_steps)

    def __getitem__(self, i):
        return self.time_steps[i], self.dt[i], self.joint_pos[i], self.joint_vel[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _compute_joint_velocities(self):
        joint_vel = np.zeros(shape=self.joint_pos.shape)
        for i in range(len(self.joint_pos)-1):        # last velocity remains zero, approaching target
            distances = self.joint_pos[i+1] - self.joint_pos[i]
            joint_vel[i] = distances / self.dt[i]

        return joint_vel

    def _compute_time_step_differences(self):
        dt = np.zeros(shape=self.time_steps.shape)
        for i in range(len(self.time_steps)-1):
            dt[i] = self.time_steps[i+1] - self.time_steps[i]
        dt[-1] = dt[-2]  # set last dt to be same as previous one, mostly this will be equidistant anyway
        return dt

    @property
    def time_steps(self):
        return self._time_steps

    @property
    def dt(self):
        return self._dt

    @property
    def joint_pos(self):
        return self._joint_pos

    @property
    def joint_vel(self):
        return self._joint_vel

    def max_abs_joint_velocities(self):
        """
        Gives the maximum absolute joint velocities along the trajectory.

        :return: ndarray (n_j,)
        """
        return np.max(np.abs(self.joint_vel), axis=0)


class TrajectoryPlanner:
    """
    Class for producing JointTrajectories for a given robot.
    Currently supports only positional control, does not consider end-effector orientation.

    :param simulator: burg.sim.SimulatorBase
    :param robot: burg.robot.RobotBase
    """
    def __init__(self, simulator, robot):
        self._simulator = simulator
        self._robot = robot

        self.default_cartesian_max_vel = 0.3
        self.default_cartesian_max_acc = 1.0
        self.default_dt = 1. / 20

    def lin(self, target_pos, max_vel=None, max_acc=None, dt=None):
        """
        Plans a linear cartesian path from robot's current end-effector position to the target position by
        inserting interpolated waypoints every `dt` seconds along the way.
        Uses a trapezoidal velocity profile as in "Modern Robotics Mechanics, Planning, and Control" by Lynch, Park,
        2017, pp. 332. It consists of three stages: Acceleration with `max_acc`, plateau with `max_vel`, deceleration
         with `-max_acc`. If this trapezoidal structure cannot be accomplished (when the target is very close), will
        skip the plateau stage and reduce acceleration/deceleration stages accordingly.
        This should give the fastest possible path from a to b, considering the velocity and acceleration constraints.

        :param target_pos: (3,) target position
        :param max_vel: maximum cartesian velocity of end-effector, if None will use default of TrajectoryPlanner
        :param max_acc: maximum cartesian acceleration of end-effector, if None will use default of TrajectoryPlanner
        :param dt: duration between time steps/waypoints in seconds, if None will use default of TrajectoryPlanner

        :return: JointTrajectory
        """
        # use default values if no custom settings provided
        v = max_vel or self.default_cartesian_max_vel
        a = max_acc or self.default_cartesian_max_acc
        dt = dt or self.default_dt

        # use current robot pos as start pos
        start_pos = self._robot.end_effector_pos()
        target_pos = np.array(target_pos)
        cartesian_dist = np.linalg.norm(target_pos - start_pos)
        assert cartesian_dist > 0, 'target and start_pos seem to be the same'

        if cartesian_dist < v**2/a:
            # the distance is too short for the trapezoidal profile when using max vel and acc
            # reduce velocity to ensure the trapezoidal form (at least full acceleration/deceleration phase, no top)
            v = np.sqrt(cartesian_dist*a)

        trajectory_time = (cartesian_dist * a + v ** 2) / (v * a)
        trajectory_steps = int(trajectory_time // dt) + 1
        _log.debug(
            f'planning trajectory:'
            f'\n\tdistance: {cartesian_dist:.4f}m'
            f'\n\texpected duration: {trajectory_time:.4f}s'
            f'\n\t{trajectory_steps} interpolation points'
        )

        time_steps = np.linspace(0, trajectory_time, trajectory_steps)
        direction = (target_pos - start_pos) / np.linalg.norm(target_pos - start_pos)
        cartesian_waypoints = np.zeros(shape=(trajectory_steps, 3))

        # compute cartesian waypoints
        for i, t in enumerate(time_steps):
            if t <= v / a:  # acceleration period
                distance_from_start = 1 / 2 * a * t ** 2
            elif t <= trajectory_time - v / a:  # max velocity period
                distance_from_start = v * t - v ** 2 / (2 * a)
            else:  # deceleration period
                distance_from_start = (2 * a * v * trajectory_time - 2 * v ** 2 - a ** 2 * (
                            t - trajectory_time) ** 2) / (2 * a)
            cartesian_waypoints[i] = start_pos + direction * distance_from_start

        assert np.allclose(cartesian_waypoints[-1], target_pos), f'waypoint interpolation went wrong, target pose is' \
                                                                 f'{target_pos} but last waypoint is ' \
                                                                 f'{cartesian_waypoints[-1]}'

        # compute joint space waypoints
        target_joint_pos = np.zeros(shape=(trajectory_steps, self._robot.num_joints))
        # todo: this does not currently consider the joint limits / range, might output infeasible trajectories
        for i, cartesian_pos in enumerate(cartesian_waypoints):
            target_joint_pos[i] = self._simulator.bullet_client.calculateInverseKinematics(
                bodyIndex=self._robot.body_id,
                endEffectorLinkIndex=self._robot.end_effector_link_id,
                targetPosition=list(cartesian_pos),
            )

        trajectory = JointTrajectory(time_steps, target_joint_pos)
        # todo: should probably check whether the joint velocities are in range of the maximum joint velocities and
        # stretch the trajectory accordingly in the relevant sections
        return trajectory

    def ptp(self):
        raise NotImplementedError('synchronised PTP trajectory planning is not implemented yet')
