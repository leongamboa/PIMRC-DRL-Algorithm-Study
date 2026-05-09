import gymnasium as gym
from stable_baselines3 import PPO 

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs, _ = env.reset()
total_reward = 0

for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward}")
env.close()