from gymnasium import Env, ObservationWrapper
import numpy as np

class MazeObservationWrapper(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
    
    def observation(self, observation: dict):
        green_ball_observation = observation["observation"]
        goal_observation = observation["desired_goal"]
        
        transformed_observation = np.concatenate([green_ball_observation,goal_observation])
        
        return transformed_observation
        
