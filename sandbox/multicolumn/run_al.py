# from apprentice.agents.ModularAgent import ModularAgent
# from apprentice.agents.pyrete_agent import PyReteAgent
from apprentice.agents.WhereWhenHowNoFoa import WhereWhenHowNoFoa
from apprentice.working_memory.representation import Sai
from py_rete import Production
from py_rete import Fact
from py_rete import V
from py_rete.conditions import Filter

from tutorenvs.multicolumn import MultiColumnAdditionSymbolic


def run_training(agent, n=10):

    env = MultiColumnAdditionSymbolic()

    p = 0

    while p < n:

        state = env.get_state()
        response = agent.request(state)

        if response == {}:
            print('hint')
            selection, action, inputs = env.request_demo()
            sai = Sai(selection=selection, action=action, inputs=inputs)

        elif isinstance(response, Sai):
            sai = response
        else:
            sai = Sai(selection=response['selection'],
                      action=response['action'],
                      inputs=response['inputs'])

        # print('sai', sai.selection, sai.action, sai.inputs)
        reward = env.apply_sai(sai.selection, sai.action, sai.inputs)
        print('reward', reward)

        next_state = env.get_state()

        # env.render()

        agent.train(state, sai, reward, next_state=next_state,
                    skill_label="multicolumn",
                    foci_of_attention=[])

        if sai.selection == "done" and reward == 1.0:
            print('Finished problem {} of {}'.format(p, n))
            p += 1


@Production(
    Fact(id=V('selection'), type="TextField", contentEditable=True, value="")
    & Fact(value=V('value')) & Filter(
        lambda value: value != "" and is_number(value) and float(value) < 10))
def update_field(selection, value):
    return Sai(selection, 'UpdateField', {'value': value})


def is_number(v):
    try:
        float(v)
        return True
    except Exception:
        return False


@Production(
    V('f1') << Fact(id=V('id1'), value=V('v1'))
    & V('f2') << Fact(id=V('id2'), value=V('v2'))
    & Filter(lambda id1, id2, v1, v2: v1 != "" and is_number(v1) and v2 != ""
             and is_number(v2) and id1 < id2))
def add_values(net, f1, f2, id1, id2, v1, v2):
    if 'depth' not in f1:
        depth1 = 0
    else:
        depth1 = f1['depth']
    if depth1 > 1:
        return

    if 'depth' not in f2:
        depth2 = 0
    else:
        depth2 = f2['depth']

    if depth1 + depth2 > 1:
        return

    print("trying to add values")
    v1 = float(v1)
    v2 = float(v2)
    v3 = v1 + v2
    if v3 == round(v3):
        v3 = int(v3)
    f = Fact(id="({}+{})".format(id1, id2),
             value=str(v3),
             depth=max(depth1, depth2) + 1)
    net.add_fact(f)


@Production(
    V('f1') << Fact(id=V('id1'), value=V('v1'))
    & Filter(lambda id1, v1: v1 != "" and is_number(v1) and float(v1) >= 10))
def mod10_value(net, f1, id1, v1):
    if 'depth' not in f1:
        depth1 = 0
    else:
        depth1 = f1['depth']
    if depth1 > 1:
        return

    print("trying to mod10 value")
    v1 = float(v1)
    v2 = v1 % 10
    if v2 == round(v2):
        v2 = int(v2)
    f = Fact(id="({}%10)".format(id1), value=str(v2), depth=depth1 + 1)
    net.add_fact(f)

    print(net)


@Production(
    V('f1') << Fact(id=V('id1'), value=V('v1'))
    & Filter(lambda id1, v1: v1 != "" and is_number(v1) and float(v1) >= 10))
def div10_value(net, f1, id1, v1):
    if 'depth' not in f1:
        depth1 = 0
    else:
        depth1 = f1['depth']
    if depth1 > 1:
        return

    print("trying to div10 value")
    v1 = float(v1)
    v2 = v1 // 10
    if v2 == round(v2):
        v2 = int(v2)
    f = Fact(id="({}//10)".format(id1), value=str(v2), depth=depth1 + 1)
    net.add_fact(f)

    print(net)


if __name__ == "__main__":

    # args = {
    #     "function_set": [
    #         "RipFloatValue",
    #         "Add",
    #         "Add3",
    #         "Div10",
    #         "Mod10",
    #     ],
    #     "feature_set": ["Equals"],
    #     "planner": "numba",
    #     "search_depth": 2,
    #     "when_learner": "decision_tree",
    #     "where_learner": "FastMostSpecific",
    #     "state_variablization": "whereappend",
    #     "strip_attrs": [
    #         "to_left", "to_right", "above", "below", "type", "id",
    #         "offsetParent", "dom_class"
    #     ],
    #     "when_args": {
    #         "cross_rhs_inference": "none"
    #     }
    # }

    # agent = ModularAgent(**args)

    agent = WhereWhenHowNoFoa('multicolumn', 'multicolumn', search_depth=1)

    # agent = PyReteAgent([update_field, add_values, mod10_value])

    run_training(agent, n=5000)
