import gym
import numpy
from ray.rllib.env.env_context import EnvContext
from gym.spaces import Discrete, Box
import numpy as np
import random

EPS = 0.05
F1_INIT = 85
F2_INIT = 50
X2_INIT = 75
Y2_INIT = 0
vx2_INIT = 40
vy2_INIT = 65
G = 9.81
M1 = 9
M2 = 17


def kinematics(v0, t, a, x=False):
    """
    Simple kinematics function
        :param v0: initial velocity, either v_x or v_y (m/s)
        :param t: time (s)
        :param a: acceleration
        :param x: if x is false, this will default to the v_f = v_0 + at equation. If x is true, this will default
                  to the x = 0.5*a*t^2 + v0*t equation
        :return: either final velocity or displacement, depending on x parameter (see above)
    """
    if x:
        return v0 * t + 0.5 * a * t ** 2
    else:
        return v0 + a*t


class Path2D(gym.envs):
    """
    This is a 2D Pathing environment meant to simulate the trajectory of two UAVs on a 2D plane.
    UAVs will have properties based on their x and y positions, x and y velocities, and force applied.
    Episode ends when either the UAV or the Scimitar flies off course (indicated by exiting the first quadrant),
    or if the Scimitar makes contact with the UAV (contact depending on epsilon value)

    Discrete action space: [do nothing,
                            speed up (increase F),
                            slow down (decrease F),
                            angle left (-x),
                            angle right (+x)]
    
    """
    def __init__(self, config: EnvContext):
        self.UAV_1 = [0, 0, 0, 0, F1_INIT]  # [x, y, vx, vy, F]
        self.UAV_2 = [X2_INIT, Y2_INIT, vx2_INIT, vy2_INIT, F2_INIT]  # [x, y, v, F]
        self.action_space = Discrete(5)
        self.observation_space = Box(np.array([0]), np.array([numpy.Inf]), dtype=np.float32)
        self.seed(config.worker_index * config.num_workers)

    def next_pos(self, timestep):
        self.UAV_1[0] += kinematics(self.UAV_1[3], timestep, self.UAV_1[5]/M1, True)
        self.UAV_2[0] += kinematics(self.UAV_2[3], timestep, self.UAV_2[5]/M2, True)
        return self.UAV_1[0], self.UAV_2[0]

    def contact(self):
        if abs(self.UAV_2[0] - self.UAV_1[0]) <= EPS and abs(self.UAV_2[1] - self.UAV_1[1]) <= EPS:
            return True
        else:
            return False

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        assert action in [0, 1, 2, 3, 4], action
        return [self.hand1, self.hand2], self.reward, True, {}

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        self.UAV_1 = [0, 0, 0, F1_INIT]
        self.UAV_2 = [X2_INIT, Y2_INIT, vx2_INIT, vy2_INIT, F2_INIT]
        return self.UAV_1, self.UAV_2

    def render(self, mode='human'):
        pass

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return random.seed(seed)
