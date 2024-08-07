import numpy as np
import tensorflow as tf
from tensorflow import keras
from hospital_env import HospitalEnv

# Create the environment
env = HospitalEnv()

# Load the trained model
model = keras.models.load_model('dqn_model.keras')

# Number of episodes to play
episodes = 5

for e in range(episodes):
    # Reset the environment
    state, _ = env.reset()
    state = np.reshape(state, [1, *env.observation_space.shape])
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        # Predict action
        action = np.argmax(model.predict(state, verbose=0)[0])
        
        # Take action
        next_state, reward, done, _, _ = env.step(action)
        
        # Render the environment (prints the grid)
        env.render()
        
        # Update total reward
        total_reward += reward
        
        # Update state
        state = np.reshape(next_state, [1, *env.observation_space.shape])
        
        step += 1
        print(f"Step: {step}, Action: {action}, Reward: {reward}")
        
    print(f"Episode: {e+1}/{episodes}, Total Steps: {step}, Total Reward: {total_reward}")
    print("-----------------")

print("Simulation complete.")