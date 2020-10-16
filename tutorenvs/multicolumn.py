from random import randint
from random import choice
from pprint import pprint

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from PIL import Image, ImageDraw

from tutorenvs.utils import BaseOppEnv

def custom_add(a, b):
    if a == '':
        a = '0'
    if b == '':
        b = '0'
    return str(int(a) + int(b))

class MultiColumnAdditionSymbolic:

    def __init__(self):
        """
        Creates a state and sets a random problem.
        """
        self.set_random_problem()
        # self.reset("", "", "", "", "")

    def reset(self, upper, lower):
        """
        Sets the state to a new fraction arithmetic problem as specified by the
        provided arguments.
        """
        correct_answer = str(int(upper) + int(lower))
        self.correct_thousands = ""
        self.correct_hundreds = ""
        self.correct_tens = ""
        self.correct_ones = ""

        if len(correct_answer) == 4:
            self.correct_thousands = correct_answer[0]
            self.correct_hundreds = correct_answer[1]
            self.correct_tens = correct_answer[2]
            self.correct_ones = correct_answer[3]
        elif len(correct_answer) == 3:
            self.correct_hundreds = correct_answer[0]
            self.correct_tens = correct_answer[1]
            self.correct_ones = correct_answer[2]
        elif len(correct_answer) == 2:
            self.correct_tens = correct_answer[0]
            self.correct_ones = correct_answer[1]
        elif len(correct_answer) == 1:
            self.correct_ones = correct_answer[0]
        else:
            raise ValueError("Something is wrong, correct answer should have 1-4 digits")

        upper_hundreds = ''
        upper_tens = ''
        upper_ones = ''

        if len(upper) == 3:
            upper_hundreds = upper[0]
            upper_tens = upper[1]
            upper_ones = upper[2]
        if len(upper) == 2:
            upper_tens = upper[0]
            upper_ones = upper[1]
        if len(upper) == 1:
            upper_ones = upper[0]

        lower_hundreds = ''
        lower_tens = ''
        lower_ones = ''

        if len(lower) == 3:
            lower_hundreds = lower[0]
            lower_tens = lower[1]
            lower_ones = lower[2]
        if len(lower) == 2:
            lower_tens = lower[0]
            lower_ones = lower[1]
        if len(lower) == 1:
            lower_ones = lower[0]

        self.steps = 0
        self.state = {
            'hundreds_carry': '',
            'tens_carry': '',
            'ones_carry': '',
            'upper_hundreds': upper_hundreds,
            'upper_tens': upper_tens,
            'upper_ones': upper_ones,
            'lower_hundreds': lower_hundreds,
            'lower_tens': lower_tens,
            'lower_ones': lower_ones,
            'operator': '+',
            'answer_thousands': '',
            'answer_hundreds': '',
            'answer_tens': '',
            'answer_ones': ''
        }

    def get_possible_selections(self):
        return ['hundreds_carry',
                'tens_carry',
                'ones_carry',
                'answer_thousands',
                'answer_hundreds',
                'answer_tens',
                'answer_ones',
                'done']

    def get_possible_args(self):
        return [
            'hundreds_carry',
            'tens_carry',
            'ones_carry',
            'upper_hundreds',
            'upper_tens',
            'upper_ones',
            'lower_hundreds',
            'lower_tens',
            'lower_ones',
            'answer_thousands',
            'answer_hundreds',
            'answer_tens',
            'answer_ones',
            ]

    def render(self):
        state = {attr: " " if self.state[attr] == '' else self.state[attr] for
                attr in self.state}

        output = " %s%s%s \n  %s%s%s\n+ %s%s%s\n-----\n %s%s%s%s\n" % (
                state["hundreds_carry"],
                state["tens_carry"],
                state["ones_carry"],
                state["upper_hundreds"],
                state["upper_tens"],
                state["upper_ones"],
                state["lower_hundreds"],
                state["lower_tens"],
                state["lower_ones"],
                state["answer_thousands"],
                state["answer_hundreds"],
                state["answer_tens"],
                state["answer_ones"],
                )

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
        upper = str(randint(1,999))
        lower = str(randint(1,999))
        self.reset(upper=upper, lower=lower)

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
            print()
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

            if (self.state['answer_thousands'] == self.correct_thousands and
                    self.state['answer_hundreds'] == self.correct_hundreds and
                    self.state['answer_tens'] == self.correct_tens and
                    self.state['answer_ones'] == self.correct_ones):
                return 1.0
            else:
                return -1.0

        # we can only edit selections that are editable
        if self.state[selection] != "":
            return -1.0

        if (selection == "answer_ones" and
                inputs['value'] == self.correct_ones):
            return 1.0

        if (selection == "ones_carry" and
                len(custom_add(self.state['upper_ones'],
                    self.state['lower_ones'])) == 2 and
                inputs['value'] == custom_add(self.state['upper_ones'],
                    self.state['lower_ones'])[0]):
                return 1.0

        if (selection == "answer_tens" and self.state['answer_ones'] != "" and
                (self.state['ones_carry'] != "" or 
                    len(custom_add(self.state['upper_ones'],
                        self.state['lower_ones'])) == 1) and
                inputs['value'] == self.correct_tens):
                return 1.0

        if (selection == "tens_carry" and
                self.state['answer_ones'] != "" and
                (self.state['ones_carry'] != "" or 
                    len(custom_add(self.state['upper_ones'],
                        self.state['lower_ones'])) == 1)):

            if (self.state['ones_carry'] != ""):
                tens_sum = custom_add(custom_add(self.state['upper_tens'],
                        self.state['lower_tens']), self.state['ones_carry'])
            else:
                tens_sum = custom_add(self.state['upper_tens'],
                        self.state['lower_tens'])

            if len(tens_sum) == 2:
                if inputs['value'] == tens_sum[0]:
                    return 1.0

        if (selection == "answer_hundreds" and
            self.state['answer_tens'] != "" and
            (self.state['tens_carry'] != "" or 
               len(custom_add(self.state['upper_tens'],
                   self.state['lower_tens'])) == 1) and
               inputs['value'] == self.correct_hundreds):
            return 1.0

        if (selection == "hundreds_carry" and
                self.state['answer_tens'] != "" and
                (self.state['tens_carry'] != "" or 
                    len(custom_add(self.state['upper_tens'],
                        self.state['lower_tens'])) == 1)):

            if (self.state['tens_carry'] != ""):
                hundreds_sum = custom_add(custom_add(
                    self.state['upper_hundreds'],
                    self.state['lower_hundreds']),
                    self.state['tens_carry'])
            else:
                hundreds_sum = custom_add(
                        self.state['upper_hundreds'],
                        self.state['lower_hundreds'])

            if len(hundreds_sum) == 2:
                if inputs['value'] == hundreds_sum[0]:
                    return 1.0

        if (selection == "answer_thousands" and
            self.state['answer_hundreds'] != "" and
            self.state['hundreds_carry'] != "" and
            inputs['value'] == self.correct_thousands):
                return 1.0

        return -1.0

    # TODO still need to rewrite for multi column arith
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


class MultiColumnAdditionOppEnv(BaseOppEnv):

    def __init__(self):
        super().__init__(MultiColumnAdditionSymbolic, max_depth=2)

    def get_rl_operators(self):
        return [
                'copy',
                'add',
                'mod10',
                'div10',
                ]

class MultiColumnAdditionDigitsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def get_dv_training(self):
        empty = {attr: '' for attr in self.tutor.state if attr != 'operator'}

        training_data = [empty]

        for i in range(1, 10):
            s = {attr: str(i) for attr in self.tutor.state if attr != 'operator'}
            training_data.append(s)

        return training_data

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
        state = {attr: " " if self.tutor.state[attr] == '' else self.tutor.state[attr] for
                attr in self.tutor.state}

        output = " %s%s%s \n  %s%s%s\n+ %s%s%s\n-----\n %s%s%s%s\n" % (
                state["hundreds_carry"],
                state["tens_carry"],
                state["ones_carry"],
                state["upper_hundreds"],
                state["upper_tens"],
                state["upper_ones"],
                state["lower_hundreds"],
                state["lower_tens"],
                state["lower_ones"],
                state["answer_thousands"],
                state["answer_hundreds"],
                state["answer_tens"],
                state["answer_ones"],
                )

        img = Image.new('RGB', (50, 90), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), output, fill='black')
        img.save('test.png')
        print(np.array(img))

        return self.tutor.state

    def __init__(self):
        self.tutor = MultiColumnAdditionSymbolic()
        n_selections = len(self.tutor.get_possible_selections())
        self.dv = DictVectorizer()
        transformed_training = self.dv.fit_transform(self.get_dv_training())
        n_features = transformed_training.shape[1]

        self.observation_space = spaces.Box(low=0.0,
                high=1.0, shape=(1, n_features), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([n_selections, 10])

    def step(self, action):
        s, a, i = self.decode(action)
        # print(s, a, i)
        # print()
        reward = self.tutor.apply_sai(s, a, i)
        # print(reward)
        
        state = self.get_rl_state()
        # pprint(state)
        obs = self.dv.transform([state])[0].toarray()
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

        i = {'value': str(v)}

        return s, a, i

    def reset(self):
        self.tutor.set_random_problem()
        state = self.get_rl_state()
        obs = self.dv.transform([state])[0].toarray()
        return obs

    def render(self, mode='human', close=False):
        self.tutor.render()

class MultiColumnAdditionPixelEnv(gym.Env):
    metadata = {'render.modes': ['human']}

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
        state = {attr: " " if self.tutor.state[attr] == '' else self.tutor.state[attr] for
                attr in self.tutor.state}

        output = " %s%s%s \n  %s%s%s\n+ %s%s%s\n-----\n %s%s%s%s\n" % (
                state["hundreds_carry"],
                state["tens_carry"],
                state["ones_carry"],
                state["upper_hundreds"],
                state["upper_tens"],
                state["upper_ones"],
                state["lower_hundreds"],
                state["lower_tens"],
                state["lower_ones"],
                state["answer_thousands"],
                state["answer_hundreds"],
                state["answer_tens"],
                state["answer_ones"],
                )

        img = Image.new('RGB', (50, 90), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), output, fill='black')
        img = img.convert('L')
        # img.save('test.png')
        return np.expand_dims(np.array(img)/255, axis=2)

    def __init__(self):
        self.tutor = MultiColumnAdditionSymbolic()
        n_selections = len(self.tutor.get_possible_selections())

        print('shape = ', self.get_rl_state().shape)

        self.observation_space = spaces.Box(low=0.0,
                high=1.0, shape=self.get_rl_state().shape, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([n_selections, 10])

    def step(self, action):
        s, a, i = self.decode(action)
        # print(s, a, i)
        # print()
        reward = self.tutor.apply_sai(s, a, i)
        # print(reward)
        
        obs = self.get_rl_state()
        # pprint(state)
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

        i = {'value': str(v)}

        return s, a, i

    def reset(self):
        self.tutor.set_random_problem()
        obs = self.get_rl_state()
        return obs

    def render(self, mode='human', close=False):
        self.tutor.render()
