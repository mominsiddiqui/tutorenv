from apprentice.agents.ModularAgent import ModularAgent
from apprentice.working_memory.representation import Sai

from tutorenvs.fractions import FractionArithSymbolic



def run_training(agent, n=10):

    env = FractionArithSymbolic()

    p = 0

    while p < n:

        state = env.get_state()
        response = agent.request(state)

        if response == {}:
            print('hint')
            selection, action, inputs = env.request_demo()
            sai = Sai(selection=selection,
                           action=action,
                           inputs=inputs)

        else:
            sai = Sai(selection=response['selection'],
                    action=response['action'],
                    inputs=response['inputs'])

        reward = env.apply_sai(sai.selection, sai.action, sai.inputs)
        print('reward', reward)

        agent.train(state, sai, reward)

        if sai.selection == "done" and reward == 1.0:
            p += 1

if __name__ == "__main__":
    args = {"function_set" : ["RipFloatValue","Add",
        'Multiply',
        "Subtract",
        # "Numerator_Multiply", "Cross_Multiply",
        "Divide"],

        "feature_set" : ["Equals"], "planner" : "numba", "search_depth" : 2,
        "when_learner": "trestle", "where_learner": "FastMostSpecific",
        "state_variablization" : "whereappend", "strip_attrs" :
        ["to_left","to_right","above","below","type","id","offsetParent","dom_class"],
        "when_args" : { "cross_rhs_inference" : "none" } }

    agent = ModularAgent(**args)

    run_training(agent, n = 100)
