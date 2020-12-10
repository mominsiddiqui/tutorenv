import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import tutorenvs
from tutorenvs.multicolumn import MultiColumnAdditionDigitsEnv
from tutorenvs.multicolumn import MultiColumnAdditionSymbolic
import numpy as np

from sklearn.tree import DecisionTreeClassifier

def train_tree(n=10):
    X = []
    y = []
    tree = DecisionTreeClassifier()
    env = MultiColumnAdditionSymbolic()

    p = 0
    while p < n:
        state = env.get_rl_state()
        env.render()

        try:
            response = decision_tree.predict(state)
        except:
            response = None

        if response is None:
            print('hint')
            sai = env.request_demo()

        else:
            sai = (response['selection'],
                   response['action'],
                   response['inputs'])

        reward = env.apply_sai(sai[0], sai[1], sai[2])
        print('reward', reward)

        if reward < 0:
            print('hint')
            sai = env.request_demo()
            reward = env.apply_sai(sai[0], sai[1], sai[2])

        X.append(state)
        y.append(sai)

        if sai.selection == "done" and reward == 1.0:
            p += 1

    return tree

if __name__ == "__main__":

    # tree = train_tree(10)
    env = MultiColumnAdditionSymbolic()

    while True:
        sai = env.request_demo()
        env.apply_sai(sai[0], sai[1], sai[2])
        env.render()
