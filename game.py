import gym
# Import your maze environment if it's custom, e.g.,
# from your_maze_module import MazePointEnv
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
def get_action_from_input():
    """
    Map user input (e.g., 'w', 'a', 's', 'd') to an action.
    Modify the mapping based on your environment's action space.
    """
    action = None
    while action is None:
        user_input = input("Enter action (w=up, s=down, a=left, d=right, q=quit): ").lower()
        if user_input == 'w':
            action = [0., 1 ]  # For example, 0 might correspond to 'up'
        elif user_input == 's':
            action = [0., -1.]  # 1 could correspond to 'down'
        elif user_input == 'a':
            action = [-1., 0]  # 2 for 'left'
        elif user_input == 'd':
            action = [1., 0.]  # 3 for 'right'
        elif user_input == 'q':
            print("Exiting...")
            exit(0)
        else:
            print("Invalid input. Please try again.")
    return action

def main():
    # Create your environment.
    # If your env is registered with Gym, you can do something like:
    maze = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]
    env = gym.make('PointMaze_UMazeDense-v3',render_mode="human",maze_map=maze)
    
    # Reset the environment to get the initial observation.
    observation = env.reset(options={"goal_cell": (6,6), "reset_cell": (1,1)})
    terminated= truncated = False
    score = 0 
    steps = 0
    while not (terminated or truncated):
        # Render the current state. This might open a window or print to the console,
        # depending on your environment's implementation.
        env.render()
        
        # Get the action from the user.
        action = get_action_from_input()
        steps += 1
        
        # Take a step in the environment.
        observation, reward, truncated,terminated, info = env.step(action)
        terminated = info['success']
        score += reward
        
    print(f"Score: {score} ,Steps: {steps}")
    
    env.close()

if __name__ == "__main__":
    main()