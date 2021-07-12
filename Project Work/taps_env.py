import gym
from ray.rllib.env.env_context import EnvContext
from gym.spaces import Discrete, Box
import numpy as np
import random


class TapsEnv(gym.envs):
    """
    GAME: 'Taps'
    RULES: Each player starts out with one finger on each hand. A player's turn consists of two possible
           actions: either "tapping" one of the hands of the opponent or "mixing" their own hands. Once a
           hand reaches or exceeds five fingers, that hand can neither be played nor "mixed". The game ends
           when one person does not have any remaining hands, meaning the opponent wins.
    ACTIONS: A "Tap" adds the amount of fingers from the player to the opponent
             A "Mix" combines the amount of fingers in a player's hand and rearranges them among their hands,
             which must be in a different configuration than before. Note that all hands must have at least
             one finger present and cannot exceed four fingers.
    REWARDS: +10 points for each finger added to an opponent's hand
             -10 points for each finger an opponent adds to your hand
             +30 points for eliminating a hand
             -30 points for having your hand eliminated
             +5 points per hand for each finger lost from a mix
             -5 points per hand for each finger gained from a mix
    """
    def __init__(self, config: EnvContext):
        self.hand1 = [1, 1]  # Player hands
        self.hand2 = [1, 1]  # Opponent hands
        self.action_space = Discrete(4)
        self.observation_space = Box(np.array([0]), np.array([inf]), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(config.worker_index * config.num_workers)

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
        reward = 0
        assert action in [0, 1, 2, 3], action
        if action == 0 and self.hand[0] != 0:
            temp = self.hand2[0]
            self.hand2[0] += self.hand1[0]
            if self.hand2[0] >= 5:
                self.hand2[0] = 0
        elif action == 1 and self.hand[0] != 0:
            temp = self.hand2[0]
            self.hand2[0] += self.hand2[1]
            if self.hand2[0] >= 5:
                self.hand2[0] = 0
        elif action == 2 and self.hand[1] != 0:
            temp = self.hand2[1]
            self.hand2[1] += self.hand2[0]
            if self.hand2[1] >= 5:
                self.hand2[1] = 0
        elif action == 3 and self.hand[1] != 0:
            temp = self.hand2[1]
            self.hand2[1] += self.hand2[1]
            if self.hand2[1] >= 5:
                self.hand2[1] = 0

        # Produce a random reward when we reach the goal.
        return [self.hand1, self.hand2], random.random() * 2 if done else -0.1, done, {}

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
        self.hand1 = [1, 1]
        self.hand2 = [1, 1]
        return self.hand1, self.hand2

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

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
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False
