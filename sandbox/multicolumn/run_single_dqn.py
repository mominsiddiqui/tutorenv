import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

from tutorenvs.utils import MultiDiscreteToDiscreteWrapper

if __name__ == "__main__":

    # multiprocess environment
    env = gym.make('MulticolumnArithSymbolic-v0')
    env = MultiDiscreteToDiscreteWrapper(env)
    model = DQN(MlpPolicy, env, verbose=1,
                learning_rate=0.0025,
                train_freq=1,
                exploration_fraction=0.5,
                exploration_initial_eps=0.45,
                gamma=0.0,
                learning_starts=1,
                policy_kwargs={'net_arch': [65, 65, 65]}, # {'qf': [65], 'pi': [65]}]},
                # tensorboard_log="./tensorboard_dqn_multi/"
                )
            # gamma=0.1,
            # tensorboard_log="./tensorboard/v0/")

    while True:
        # Train
        model.learn(total_timesteps=1000000)

        # Test
        # obs = env.reset()
        # rwd = 0
        # for _ in range(10000):
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones, info = env.step(action)
        #     rwd += np.sum(rewards)
        #     env.render()
        # print(rwd)
