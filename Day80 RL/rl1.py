import gym
env = gym.make("CartPole-v1", render_mode="human")

env.reset()
env.render()

input("Press enter to start the episode...")
env.close()