## TutorGym

This library contains headless versions of a number of commonly used tutoring system environments for training simulated students.  There are currently two different tutoring systems that can be loaded: the fraction arithmetic tutor and a multi-column arithmetic tutor.

## Installation

To install the base Gym library, use `pip install tutorgym`.

## Example with PPO

```
import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
import tutorenvs
import numpy as np


if __name__ == "__main__":

    # multiprocess environment
    env = make_vec_env('MultiColumnArith-v2', n_envs=9)
    model = PPO2(CnnPolicy, env, verbose=1,
            gamma=0.5,
            policy_kwargs={'net_arch': [65, 65, {'vf': [65], 'pi': [65]}]},
            tensorboard_log="./tensorboard/v2/")

    while True:
        model.learn(total_timesteps=100)

        # To demonstrate saving and loading
        # model.save("ppo2_multicolumn-v0")
        # del model
        # model = PPO2.load("ppo2_multicolumn-v0")

        # Enjoy trained agent
        obs = env.reset()
        rwd = 0
        for _ in range(10000000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            rwd += np.sum(rewards)
            env.render()
        print(rwd)

```


## Environment Versioning

We recommend using Python 3.8.0 for this project. 

## Citation

A paper from when TutorGym just came out is available https://educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_112.pdf, and can be cited with the following bibtex entry:

```
MacLellan, C.J., Gupta, A. (2021). Learning Expert Models for Educationally Relevant Tasksusing Reinforcement Learning. Proceedings of the Fourteenth International Conference on Educational Data Mining.

```

A video presentation from this citation at the EDM 2021 Conference is available at: https://youtu.be/WD3fujvp86Q