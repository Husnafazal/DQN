from hospital_env import HospitalEnv

env = HospitalEnv()

for _ in range(5):
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action: {action}, Reward: {reward}")
    
    print(f"Episode finished. Total reward: {total_reward}")
    print("-----------------")