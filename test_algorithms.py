from IPython import embed
import gym
import argparse

from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, SAC, TD3
from stable_baselines.common.policies import MlpPolicy

NAME_TO_ALGO = {
    "PPO": PPO2,
    "A2C": A2C,
    "ACER": ACER,
    "ACKTR": ACKTR,
}

def test_RL_algorithm(rl_algo):
    print("Testing RL Algorithm: {}".format(rl_algo))
    env = gym.make('CartPole-v1')

    model = NAME_TO_ALGO[rl_algo](MlpPolicy, env)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(200):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

parser = argparse.ArgumentParser(description="Specify RL algorithm to test on CartPole")
parser.add_argument('method', type=str, help="RL method to test")

args = parser.parse_args()
test_RL_algorithm(args.method)


