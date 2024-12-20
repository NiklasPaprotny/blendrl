import gymnasium as gym
import cv2

class DQNWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_mode):
        super(DQNWrapper, self).__init__(env) # Ensure initialization of parent class ObservationWrapper from gymnasium/core.py.
        self.obs_mode = obs_mode
        if obs_mode == "dqn":
            # This observation_space for dqn is also defined in ocatari/core.py.
            self.observation_space = gym.spaces.Box(
                low=0, high=255.0, shape=(84, 84)
            )

    # observation() method is being used in the step() methods of gymnasium/core.py. Since they use raw RGB pixel formats, they will need to be adjusted when obs_mode == "dqn".
    def observation(self, obs):
        if self.obs_mode == "dqn":
            new_obs = cv2.resize(obs, (84, 84)) # obs initially has shape (210, 160, 3) due to its RGB format. If obs_mode == "dqn", then it needs to be resized to (84, 84).
            new_obs = cv2.cvtColor(new_obs, cv2.COLOR_RGB2GRAY) # Convert to grayscale, since dqn doesn't use RGB format.
        return new_obs