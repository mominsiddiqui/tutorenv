from random import randint
from random import choice

import gym
from gym import error, spaces, utils
from gym.utils import seeding


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

    def get_state(self):
        """
        Returns the current state as a dict.
        """
        state_output = {attr:
                        {'id': attr, 'value': self.state[attr], 'type': 'TextField',
                         'contentEditable': self.state[attr] == ""}
                        for attr in self.state}
        state_output['done'] = {
            'id': 'done',
            'type': 'Button'
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
        reward = self.evaluate_sai(selection, action, inputs)

        if reward == -1.0:
            return reward

        if selection == "done":
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

class FractionArithEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass
