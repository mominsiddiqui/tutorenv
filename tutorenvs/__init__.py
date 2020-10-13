from gym.envs.registration import register
from tutorenvs.fractions import FractionArithDigitsEnv
from tutorenvs.fractions import FractionArithOppEnv
from tutorenvs.multicolumn import MultiColumnAdditionOppEnv
from tutorenvs.multicolumn import MultiColumnAdditionDigitsEnv

register(
    id='FractionArith-v0',
    entry_point='tutorenvs:FractionArithDigitsEnv',
)

register(
    id='FractionArith-v1',
    entry_point='tutorenvs:FractionArithOppEnv',
)

register(
    id='MultiColumnArith-v0',
    entry_point='tutorenvs:MultiColumnAdditionDigitsEnv',
)

register(
    id='MultiColumnArith-v1',
    entry_point='tutorenvs:MultiColumnAdditionOppEnv',
)
