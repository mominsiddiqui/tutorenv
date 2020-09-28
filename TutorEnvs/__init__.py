from gym.envs.registration import register
from tutorenvs.fractions import FractionArithEnv

register(
    id='FractionArith-v0',
    entry_point='tutorenvs:FractionArithEnv',
)
