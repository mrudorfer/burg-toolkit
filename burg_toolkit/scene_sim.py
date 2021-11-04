import numpy as np

from .sim import SimulatorBase


class SceneSimulator(SimulatorBase):
    """
    A simulator that simulates objects in a scene until they attain a resting pose.
    The resting pose is assessed by the maximum velocity element (linear/angular) of the objects.
    The maximum velocity element needs to be below `eps` for at least `min_secs_below_eps` amount of simulated time to
    be considered at rest.

    :param verbose: set to True if you want to see the GUI
    :param timeout: If no resting pose is attained, simulation stops after this amount of simulated seconds.
    :param eps: threshold for the maximum velocity element
    :param min_secs_below_eps: amount of simulated seconds the maximum velocity needs to be below `eps` to be at rest
    """
    def __init__(self, verbose=False, timeout=10, eps=1e-03, min_secs_below_eps=0.5):
        super().__init__(verbose=verbose)
        self.timeout = timeout
        self.eps = eps
        self.min_secs_below_eps = min_secs_below_eps
        self._reset(plane_and_gravity=True)

    def simulate_object_instance(self, object_instance):
        """
        Simulates the given `object_instance` for at most `timeout` seconds or until it attains a resting pose on an
        xy-plane at z=0. The pose of the instance will be updated after the simulation ends.

        :param object_instance: core.ObjectInstance which shall be simulated

        :return: number of simulated seconds as indicator whether the simulator timed out or not.
        """
        self._reset(plane_and_gravity=True)
        instance = self._add_object(object_instance)

        steps_below_eps = 0
        max_steps = self.min_secs_below_eps / self.dt
        while self._simulated_seconds < self.timeout and steps_below_eps < max_steps:
            self._step()
            vel, angular_vel = self._p.getBaseVelocity(instance)
            velocities = np.asarray([*vel, *angular_vel])
            max_vel = np.abs(velocities).max()
            # print(f'{self._simulated_seconds:.3f}s, max_vel: {max_vel}')
            if max_vel < self.eps:
                steps_below_eps += 1

        object_instance.pose = self._get_body_pose(instance, convert2burg=True)
        return self._simulated_seconds

    def simulate_scene(self, scene):
        """
        Simulates the given `scene` for at most `timeout` seconds or until all its containing object instances
        attain a resting pose. An xy-plane at z=0 will be added and all background objects will be fixed, e.g.
        cannot move in space.
        The poses of the instances will be updated after the simulation ends.

        :param scene: core.Scene which shall be simulated

        :return: number of simulated seconds as indicator whether the simulator timed out or not.
        """
        self._reset(plane_and_gravity=True)
        instance_body_ids = {}
        bg_body_ids = {}
        for instance in scene.objects:
            instance_body_ids[instance] = self._add_object(instance)
        for bg_instance in scene.bg_objects:
            bg_body_ids[bg_instance] = self._add_object(bg_instance, fixed_base=True)

        steps_below_eps = 0
        max_steps = self.min_secs_below_eps / self.dt
        while self._simulated_seconds < self.timeout and steps_below_eps < max_steps:
            self._step()
            # check velocities of all objects
            max_vel = 0
            for body_id in instance_body_ids.values():
                vel, angular_vel = self._p.getBaseVelocity(body_id)
                max_body_vel = np.abs(np.asarray([*vel, *angular_vel])).max()
                max_vel = max(max_vel, max_body_vel)
                if max_vel > self.eps:
                    # don't need to check the other objects if already exceeding threshold
                    break
            if max_vel < self.eps:
                steps_below_eps += 1

        for instance in scene.objects:
            instance.pose = self._get_body_pose(instance_body_ids[instance], convert2burg=True)

        return self._simulated_seconds
