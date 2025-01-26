import time
import matplotlib.pyplot as plt
import os 
import numpy as np

def plot_results(score_history, hyperparams, save_dir='training_results'):
    # Create directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"{save_dir}/{timestamp}"
    os.makedirs(save_path, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Score per Episode', color='blue')
    plt.title('PPO Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.axhline(y=np.mean(score_history[-5:]), color='red', linestyle='--', 
                label=f'Average Score (last 5 episodes): {np.mean(score_history[-5:]):.1f}')
    
    # Add hyperparameter annotations
    hyperparam_text = '\n'.join([f'{k}: {v}' for k, v in hyperparams.items()])
    plt.text(0.02, 0.98, hyperparam_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(f"{save_path}/training_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save hyperparameters to a text file
    with open(f"{save_path}/hyperparameters.txt", 'w') as f:
        f.write(hyperparam_text)
