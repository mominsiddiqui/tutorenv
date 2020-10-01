from gym.envs.registration import register
from tutorenvs.fractions import FractionArithDigitsEnv
from tutorenvs.fractions import FractionArithOppEnv

register(
    id='FractionArith-v0',
    entry_point='tutorenvs:FractionArithDigitsEnv',
)

register(
    id='FractionArith-v1',
    entry_point='tutorenvs:FractionArithOppEnv',
)
