from apprentice.agents.tact_agent import TACTAgent
from apprentice.working_memory.representation import Sai

from tutorenvs.fractions import FractionArithSymbolic

from py_rete import Production
from py_rete import Fact
from py_rete import V
from py_rete import Filter


def run_training(agent, n=10):

    env = FractionArithSymbolic()

    p = 0

    while p < n:

        env.render()
        state = env.get_state()
        response = agent.request(state)

        if response == {}:
            print('hint')
            selection, action, inputs = env.request_demo()
            sai = Sai(selection=selection, action=action, inputs=inputs)

        else:
            sai = Sai(selection=response['selection'],
                      action=response['action'],
                      inputs=response['inputs'])

            print(response['inputs'])

        # print(sai)
        reward = env.apply_sai(sai.selection, sai.action, sai.inputs)
        print('reward', reward)

        if reward == -1:
            print(sai)

        next_state = env.get_state()

        # from jsondiff import diff
        # print(diff(state, next_state))

        agent.train(state, sai, reward, next_state=next_state,
                    skill_label="fractions",
                    foci_of_attention=[])

        if sai.selection == "done" and reward == 1.0:
            print('Finished problem {} of {}'.format(p, n))
            p += 1


if __name__ == "__main__":

    @Production(
        Fact(id='answer_num', contentEditable=False) &
        Fact(id='answer_denom', contentEditable=False) &
        Fact(id='done')
    )
    def correct_doneSAI():
        return Sai(selection='done',
                   action='ButtonPressed',
                   inputs={'value': -1})

    def is_number(n):
        try:
            float(n)
            return True
        except Exception:
            return False

    @Production(
        Fact(id=V('id1'), value=V('n1')) &
        Filter(lambda n1: is_number(n1)) &
        Fact(id=V('id2'), value=V('n2')) &
        Filter(lambda n2: is_number(n2)) &
        Filter(lambda id1, id2: id1 <= id2)
    )
    def add(id1, id2, n1, n2):
        n1 = float(n1)
        n2 = float(n2)
        s = n1 + n2
        if int(s) == s:
            s = int(s)

        return [Fact(id="({}+{})".format(id1, id2),
                     value="{}".format(s))]

    @Production(
        Fact(id=V('id1'), value=V('n1')) &
        Filter(lambda n1: is_number(n1)) &
        Fact(id=V('id2'), value=V('n2')) &
        Filter(lambda n2: is_number(n2)) &
        Filter(lambda id1, id2: id1 <= id2)
    )
    def multiply(id1, id2, n1, n2):
        n1 = float(n1)
        n2 = float(n2)
        p = n1 * n2
        if int(p) == p:
            p = int(p)

        return [Fact(id="({}*{})".format(id1, id2),
                     value="{}".format(p))]

    @Production(
        Fact(id=V('selection'), contentEditable=True) &
        Fact(value=V('input_value'))
    )
    def copySAI(input_value, selection):
        return Sai(selection=selection, action="UpdateField",
                   inputs={'value': input_value})

    # @Production(
    #     Fact(id=V('some_field'), value=V('some_value')),
    #     Filter(lambda some_value: len(some_value.split()) > 1)
    # )
    # def unigram(net, some_field, some_value):
    #     words = some_value.split()
    #
    #     for i, word in enumerate(words):
    #         net.add_fact(Fact(id="Unigram-{}-of-{}".format(i, some_field),
    #                           value=word))

    #     new_facts = [Fact(id="Unigram-{}-of-{}".format(i, some_field),
    #                       value=word) for i, word in enumerate(words)]

    #     return new_facts

    @Production(
        Fact(id=V('id1'), value=V('v1')) &
        Filter(lambda v1: is_number(v1)) &
        Fact(id=V('id2'), value=V('v1')) &
        Filter(lambda id1, id2: id1 < id2))
    def equals(id1, id2):
        return [Fact(relation='equality', first=id1, second=id2)]

    agent = TACTAgent(concepts=[equals],
                      skills=[])
                      # skills=[correct_doneSAI, add, multiply, copySAI])

    run_training(agent, n=500)
