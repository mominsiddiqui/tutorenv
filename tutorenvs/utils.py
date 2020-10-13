from pprint import pprint

import gym
from gym import error, spaces, utils
from sklearn.feature_extraction import DictVectorizer
import numpy as np

class BaseOppEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, tutor_class, max_depth=1):
        print('building env')
        self.tutor = tutor_class()

        self.max_depth = max_depth
        self.internal_memory = {}

        self.possible_attr = set(self.tutor.get_possible_args())
        for _ in range(self.max_depth):
            new = set()
            for opp in self.get_rl_operators():
                for a1 in self.possible_attr:
                    for a2 in self.possible_attr:
                        new.add((opp, a1, a2))
            self.possible_attr = self.possible_attr.union(new)
        print('# features = %i' % len(self.possible_attr))

        self.possible_args = list(set([attr[1] if isinstance(attr, tuple) else
            attr for attr in self.possible_attr]))
        print('# args = %i' % len(self.possible_args))
        
        # one additional option to save result internally
        n_selections = len(self.tutor.get_possible_selections()) + 1
        print('getting rl state')
        n_features = len(self.get_rl_state()) 
        print('done getting rl state')
        n_operators = len(self.get_rl_operators())
        n_args = len(self.possible_args)
        self.dv = DictVectorizer()
        self.dv.fit([self.get_rl_state()])

        self.observation_space = spaces.Box(low=0.0,
                high=1.0, shape=(1, n_features), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([n_selections, n_operators,
            n_args, n_args])
        print('done')

    def get_rl_operators(self):
        return [
                'copy',
                'add',
                'multiply',
                'mod10',
                'div10',
                ]

    def get_rl_state(self):
        # self.state = {
        #     'hundreds_carry': '',
        #     'tens_carry': '',
        #     'ones_carry': '',
        #     'upper_hundreds': upper_hundreds,
        #     'upper_tens': upper_tens,
        #     'upper_ones': upper_ones,
        #     'lower_hundreds': lower_hundreds,
        #     'lower_tens': lower_tens,
        #     'lower_ones': lower_ones,
        #     'operator': '+',
        #     'answer_thousands': '',
        #     'answer_hundreds': '',
        #     'answer_tens': '',
        #     'answer_ones': ''
        # }

        state = {}
        for attr in self.tutor.state:

            # TODO need generic way to handle this.
            if attr == "operator":
                continue

            # just whether or not there is a value
            state[attr] = self.tutor.state[attr] != ""

        # if its in internal memory, then return true, else false.
        for possible_attr in self.possible_attr:
            state[possible_attr] = possible_attr in self.internal_memory

        print('done with base attributes in state')
        print('# of base attributes = %i' % len(state))

        # relations (equality, >10)
        new_relations = {}

        for attr in state:
            attr_val = None
            if attr in self.tutor.state:
                attr_val = self.tutor.state[attr]
            elif attr in self.internal_memory:
                attr_val = self.internal_memory[attr]
            else:
                attr_val = ''

            # greater than 9
            try:
                new_relations['greater_than_9(%s)' % str(attr)] = float(attr_val) > 9
            except Exception:
                new_relations['greater_than_9(%s)' % str(attr)] = False

            # # equality
            # for attr2 in state:
            #     if str(attr) >= str(attr2):
            #         continue

            #     attr2_val = None
            #     if attr2 in self.tutor.state:
            #         attr2_val = self.tutor.state[attr2]
            #     elif attr2 in self.internal_memory:
            #         attr2_val = self.internal_memory[attr2]
            #     else:
            #         attr2_val = ''
            #     new_relations['eq(%s,%s)' % (attr, attr2)] = attr_val == attr2_val

        print('done with creating new relations')
        print('# of new relations = %i' % len(new_relations))

        for attr in new_relations:
            state[attr] = new_relations[attr]

        # convert all attributes to strings
        return {str(attr): state[attr] for attr in state}

    def step(self, action):
        try:
            s, a, i = self.decode(action)
            
            if isinstance(s, tuple):
                if s in self.internal_memory or i == '':
                    reward = -1
                else:
                    self.internal_memory[s] = i
                    reward = -0.01
            else:
                reward = self.tutor.apply_sai(s, a, i)
            done = (s == 'done' and reward == 1.0)
        except ValueError:
            reward = -1
            done = False

        # print(s, a, i)
        # print()
        # print(reward)
        
        state = self.get_rl_state()
        # pprint(state)
        obs = self.dv.transform([state])[0].toarray()
        info = {}

        return obs, reward, done, info


    def apply_rl_op(self, op, arg1, arg2):
        a1 = None
        a2 = None

        if arg1 in self.tutor.state:
            a1 = self.tutor.state[arg1] 
        elif arg1 in self.internal_memory:
            a1 = self.internal_memory[arg1]
        else:
            raise ValueError('Element not in memory')

        if arg2 in self.tutor.state:
            a2 = self.tutor.state[arg2] 
        elif arg2 in self.internal_memory:
            a2 = self.internal_memory[arg2]
        else:
            raise ValueError('Element not in memory')

        if op == "copy":
            return a1 
        elif op == "add":
            return str(int(a1) + int(a2))
        elif op == "multiply":
            return str(int(a1) * int(a2))
        elif op == "mod10":
            return str(int(a1) % 10)
        elif op == "div10":
            return str(int(a1) // 10)

    def decode(self, action):
        # print(action)

        op = self.get_rl_operators()[action[1]]
        arg1 = self.possible_args[action[2]]
        arg2 = self.possible_args[action[3]]

        if action[0] == len(self.tutor.get_possible_selections()):
            s = (opp, arg1, arg2)
        else:
            s = self.tutor.get_possible_selections()[action[0]]

        if s == "done":
            a = "ButtonPressed"
        else:
            a = "UpdateField"
        
        if s == "done":
            v = -1
        if s == "check_convert":
            v = "x"
        else:
            v = self.apply_rl_op(op, arg1, arg2)

        i = {'value': str(v)}

        return s, a, i

    def reset(self):
        self.tutor.set_random_problem()
        state = self.get_rl_state()
        self.internal_memory = {}
        obs = self.dv.transform([state])[0].toarray()
        return obs

    def render(self, mode='human', close=False):
        self.tutor.render()
