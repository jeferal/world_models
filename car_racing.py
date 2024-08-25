import gymnasium as gym
env = gym.make("CarRacing-v2", domain_randomize=True, render_mode='human')

observation, info = env.reset()
for _ in range(1000):
  action = env.action_space.sample()  # this is where you would insert your policy
  
  # Convert the 
  observation, reward, terminated, truncated, info = env.step(action.tolist())

  breakpoint()

  if terminated or truncated:
    observation, info = env.reset()

  env.render()

env.close()
