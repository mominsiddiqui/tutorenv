from random import randint
from random import choice
from pprint import pprint

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.feature_extraction import FeatureHasher
import numpy as np


class FractionArithSymbolic:

    def __init__(self):
        """
        Creates a state and sets a random problem.
        """
        self.set_random_problem()
        # self.reset("", "", "", "", "")

    def reset(self, num1, denom1, operator, num2, denom2):
        """
        Sets the state to a new fraction arithmetic problem as specified by the
        provided arguments.
        """
        self.steps = 0
        self.state = {
            'initial_num_left': num1,
            'initial_denom_left': denom1,
            'initial_operator': operator,
            'initial_num_right': num2,
            'initial_denom_right': denom2,
            'check_convert': '',
            'convert_num_left': '',
            'convert_denom_left': '',
            'convert_operator': operator,
            'convert_num_right': '',
            'convert_denom_right': '',
            'answer_num': '',
            'answer_denom': '',
        }

    def get_possible_selections(self):
        return ['check_convert',
                'convert_num_left',
                'convert_denom_left',
                'convert_num_right',
                'convert_denom_right',
                'answer_num',
                'answer_denom',
                'done']

    def render(self):
        output = "%s\t\t%s\n---\t%s\t---\t=\n%s\t\t%s\n\nConvert? | %s |\n\n%s\t\t%s\t\t%s\n---\t%s\t---\t=\t---\n%s\t\t%s\t\t%s\n" % (self.state['initial_num_left'],
                self.state['initial_num_right'],
                self.state['initial_operator'],
                self.state['initial_denom_left'],
                self.state['initial_denom_right'],
                self.state['check_convert'],
                self.state['convert_num_left'],
                self.state['convert_num_right'],
                self.state['answer_num'],
                self.state['convert_operator'],
                self.state['convert_denom_left'],
                self.state['convert_denom_right'],
                self.state['answer_denom'])

        print("------------------------------------------------------")
        print(output)
        print("------------------------------------------------------")
        print()

    def get_state(self):
        """
        Returns the current state as a dict.
        """
        state_output = {attr:
                        {'id': attr, 'value': self.state[attr],
                         'type': 'TextField',
                         'contentEditable': self.state[attr] == "",
                         'dom_class': 'CTATTable--cell',
                         'above': '',
                         'below': '',
                         'to_left': '',
                         'to_right': ''
                         }
                        for attr in self.state}
        state_output['done'] = {
            'id': 'done',
            'type': 'Component',
            'dom_class': 'CTATDoneButton',
            'above': '',
            'below': '',
            'to_left': '',
            'to_right': ''
        }

        return state_output

    def set_random_problem(self):
        num1 = str(randint(1, 15))
        num2 = str(randint(1, 15))
        denom1 = str(randint(2, 15))
        denom2 = str(randint(2, 15))
        operator = choice(['+', '*'])

        self.reset(num1, denom1, operator, num2, denom2)

    def apply_sai(self, selection, action, inputs):
        """
        Give a SAI, it applies it. This method returns feedback (i.e., -1 or 1).
        """
        self.steps += 1
        reward = self.evaluate_sai(selection, action, inputs)

        if reward == -1.0:
            return reward

        if selection == "done":
            print("DONE! Only took %i steps." % self.steps)
            self.render()
            # pprint(self.state)
            self.set_random_problem()

        else:
            self.state[selection] = inputs['value']

        return reward

    def evaluate_sai(self, selection, action, inputs):
        """
        Given a SAI, returns whether it is correct or incorrect.
        """
        # done step
        if selection == "done":

            if action != "ButtonPressed":
                return -1.0

            if self.state['answer_num'] != "" and self.state['answer_denom'] != "":
                return 1.0
            else:
                return -1.0

        # we can only edit selections that are editable
        if self.state[selection] != "":
            return -1.0

        if (self.state['initial_operator'] == '+' and
                self.state['initial_denom_left'] == self.state['initial_denom_right']):
            # add same denoms
            if (selection == 'answer_num' and (inputs['value'] ==
                                               str(int(self.state['initial_num_left']) +
                                                int(self.state['initial_num_right'])))):
                return 1.0

            if (selection == 'answer_denom' and inputs['value'] ==
                    self.state['initial_denom_left']):
                return 1.0

            return -1.0

        if (self.state['initial_operator'] == "+" and
                self.state['initial_denom_left'] != self.state['initial_denom_right']):
            # add, different denoms
            if selection == "check_convert":
                return 1.0

            if (selection == "convert_num_left" and
                    self.state['convert_denom_left'] != "" and
                    inputs['value'] == str(int(self.state['initial_num_left']) *
                                        int(self.state['initial_denom_right']))):
                return 1.0

            if (selection == "convert_denom_left" and
                    self.state['check_convert'] != "" and
                    inputs['value'] == str(int(self.state['initial_denom_left']) *
                                        int(self.state['initial_denom_right']))):
                return 1.0

            if (selection == "convert_num_right" and
                    self.state['convert_denom_right'] != "" and
                    inputs['value'] == str(int(self.state['initial_num_right']) *
                                        int(self.state['initial_denom_left']))):
                return 1.0

            if (selection == "convert_denom_right" and
                    self.state['convert_denom_left'] != "" and
                    inputs['value'] == str(int(self.state['initial_denom_left']) *
                                        int(self.state['initial_denom_right']))):
                return 1.0

            if (selection == 'answer_num' and
                    self.state['convert_num_left'] != "" and
                    self.state['convert_num_right'] != "" and
                    (inputs['value'] == str(int(self.state['convert_num_left']) +
                                         int(self.state['convert_num_right'])))):
                return 1.0

            if (selection == 'answer_denom' and
                    self.state['convert_num_left'] != "" and
                    self.state['convert_num_right'] != "" and
                    inputs['value'] == self.state['convert_denom_right']):
                return 1.0

            return -1.0

        if (self.state['initial_operator'] == "*"):
            # multiply
            if (selection == 'answer_num' and (inputs['value'] ==
                                               str(int(self.state['initial_num_left']) *
                                                int(self.state['initial_num_right'])))):
                return 1.0

            if (selection == 'answer_denom' and (inputs['value'] ==
                                                 str(int(self.state['initial_denom_left']) *
                                                  int(self.state['initial_denom_right'])))):
                return 1.0

            return -1.0

        raise Exception("evaluate_sai logic missing")

    def request_demo(self):
        """
        Returns a correct next-step SAI
        """
        if (self.state['initial_operator'] == '+' and
                self.state['initial_denom_left'] == self.state['initial_denom_right']):
            if self.state['answer_num'] == "":
                return ('answer_num', "UpdateField",
                        {'value': str(int(self.state['initial_num_left']) +
                                      int(self.state['initial_num_right']))})

            if self.state['answer_denom'] == "":
                return ('answer_denom', "UpdateField",
                        {'value': self.state['initial_denom_left']})

            return ('done', "ButtonPressed", {'value': -1})

        if (self.state['initial_operator'] == "+" and
                self.state['initial_denom_left'] != self.state['initial_denom_right']):
            
            if self.state['check_convert'] == "":
                return ('check_convert', 'UpdateField', {"value": 'x'})

            if self.state['convert_denom_left'] == "":
                return ('convert_denom_left', "UpdateField",
                        {'value': str(int(self.state['initial_denom_left']) *
                                      int(self.state['initial_denom_right']))})

            if self.state['convert_num_left'] == "":
                return ('convert_num_left', "UpdateField",
                        {'value': str(int(self.state['initial_num_left']) *
                                      int(self.state['initial_denom_right']))})

            if self.state['convert_denom_right'] == "":
                return ('convert_denom_right', "UpdateField",
                        {'value': str(int(self.state['initial_denom_left']) *
                                      int(self.state['initial_denom_right']))})

            if self.state['convert_num_right'] == "":
                return ('convert_num_right', "UpdateField",
                        {'value': str(int(self.state['initial_denom_left']) *
                                      int(self.state['initial_num_right']))})

            if self.state['answer_num'] == "":
                return ('answer_num', "UpdateField",
                        {'value': str(int(self.state['convert_num_left']) +
                                      int(self.state['convert_num_right']))})

            if self.state['answer_denom'] == "":
                return ('answer_denom', "UpdateField",
                        {'value': self.state['convert_denom_right']})

            return ('done', "ButtonPressed", {'value': -1})

        if (self.state['initial_operator'] == "*"):
            if self.state['answer_num'] == "":
                return ('answer_num', "UpdateField",
                        {'value': str(int(self.state['initial_num_left']) *
                                      int(self.state['initial_num_right']))})

            if self.state['answer_denom'] == "":
                return ('answer_denom', "UpdateField",
                        {'value': str(int(self.state['initial_denom_left']) *
                                      int(self.state['initial_denom_right']))})

            return ('done', "ButtonPressed", {'value': -1})

        raise Exception("request demo - logic missing")

class FractionArithDigitsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def get_rl_state(self):
        # self.state = {
        #     'initial_num_left': num1,
        #     'initial_denom_left': denom1,
        #     'initial_operator': operator,
        #     'initial_num_right': num2,
        #     'initial_denom_right': denom2,
        #     'check_convert': '',
        #     'convert_num_left': '',
        #     'convert_denom_left': '',
        #     'convert_operator': operator,
        #     'convert_num_right': '',
        #     'convert_denom_right': '',
        #     'answer_num': '',
        #     'answer_denom': '',
        # }

        state = {}
        for attr in self.tutor.state:
            if attr == "initial_operator" or attr == "convert_operator":
                state[attr] = self.tutor.state[attr]
                continue

            state[attr + "[0]"] = ""
            state[attr + "[1]"] = ""
            state[attr + "[2]"] = ""

            if self.tutor.state[attr] != "":
                l = len(self.tutor.state[attr])

                if l > 2:
                    state[attr + "[0]"] = self.tutor.state[attr][-3]
                if l > 1:
                    state[attr + "[1]"] = self.tutor.state[attr][-2]

                state[attr + "[2]"] = self.tutor.state[attr][-1]

            # print(self.tutor.state[attr])
            # pprint(state)

        return state

    def __init__(self):
        self.tutor = FractionArithSymbolic()
        n_selections = len(self.tutor.get_possible_selections())
        n_features = 10000
        self.feature_hasher = FeatureHasher(n_features=n_features)

        self.observation_space = spaces.Box(low=0.0,
                high=1.0, shape=(1, n_features), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([n_selections, 10, 10, 10])

    def step(self, action):
        s, a, i = self.decode(action)
        # print(s, a, i)
        # print()
        reward = self.tutor.apply_sai(s, a, i)
        # print(reward)
        
        state = self.get_rl_state()
        # pprint(state)
        obs = self.feature_hasher.transform([state])[0].toarray()
        done = (s == 'done' and reward == 1.0)
        info = {}

        return obs, reward, done, info

    def decode(self, action):
        # print(action)
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
            v = action[1]
            v += 10 * action[2]
            v += 100 * action[3]
        # if action[4]:
        #     v *= -1
        i = {'value': str(v)}

        return s, a, i

    def reset(self):
        self.tutor.set_random_problem()
        state = self.get_rl_state()
        obs = self.feature_hasher.transform([state])[0].toarray()
        return obs

    def render(self, mode='human', close=False):
        self.tutor.render()


class FractionArithOppEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def get_rl_state(self):
        # self.state = {
        #     'initial_num_left': num1,
        #     'initial_denom_left': denom1,
        #     'initial_operator': operator,
        #     'initial_num_right': num2,
        #     'initial_denom_right': denom2,
        #     'check_convert': '',
        #     'convert_num_left': '',
        #     'convert_denom_left': '',
        #     'convert_operator': operator,
        #     'convert_num_right': '',
        #     'convert_denom_right': '',
        #     'answer_num': '',
        #     'answer_denom': '',
        # }

        state = {}
        for attr in self.tutor.state:
            if attr == "initial_operator" or attr == "convert_operator":
                state[attr] = self.tutor.state[attr]
                continue

            state[attr + "[0]"] = ""
            state[attr + "[1]"] = ""
            state[attr + "[2]"] = ""

            if self.tutor.state[attr] != "":
                l = len(self.tutor.state[attr])

                if l > 2:
                    state[attr + "[0]"] = self.tutor.state[attr][-3]
                if l > 1:
                    state[attr + "[1]"] = self.tutor.state[attr][-2]

                state[attr + "[2]"] = self.tutor.state[attr][-1]

            # print(self.tutor.state[attr])
            # pprint(state)

        return state

    def __init__(self):
        self.tutor = FractionArithSymbolic()
        n_selections = len(self.tutor.get_possible_selections())
        n_features = 10000
        self.feature_hasher = FeatureHasher(n_features=n_features)

        self.observation_space = spaces.Box(low=0.0,
                high=1.0, shape=(1, n_features), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([n_selections, 10, 10, 10])

    def step(self, action):
        s, a, i = self.decode(action)
        # print(s, a, i)
        # print()
        reward = self.tutor.apply_sai(s, a, i)
        # print(reward)
        
        state = self.get_rl_state()
        # pprint(state)
        obs = self.feature_hasher.transform([state])[0].toarray()
        done = (s == 'done' and reward == 1.0)
        info = {}

        return obs, reward, done, info

    def decode(self, action):
        # print(action)
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
            v = action[1]
            v += 10 * action[2]
            v += 100 * action[3]
        # if action[4]:
        #     v *= -1
        i = {'value': str(v)}

        return s, a, i

    def reset(self):
        self.tutor.set_random_problem()
        state = self.get_rl_state()
        obs = self.feature_hasher.transform([state])[0].toarray()
        return obs

    def render(self, mode='human', close=False):
        pass
