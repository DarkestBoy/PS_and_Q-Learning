import gym
env = gym.make('FrozenLake-v1')
obs = env.reset()
print("success!：", obs)
