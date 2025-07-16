import gym
env = gym.make('FrozenLake-v1')
obs = env.reset()
print("环境创建成功，初始观测值：", obs)
