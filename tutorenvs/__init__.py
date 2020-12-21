from gym.envs.registration import register
from tutorenvs.fractions import FractionArithDigitsEnv
from tutorenvs.fractions import FractionArithOppEnv
from tutorenvs.multicolumn import MultiColumnAdditionDigitsEnv
from tutorenvs.multicolumn import MultiColumnAdditionPixelEnv
from tutorenvs.multicolumn import MultiColumnAdditionPerceptEnv

register(
    id='FractionArith-v0',
    entry_point='tutorenvs:FractionArithOppEnv',
)

register(
    id='FractionArith-v1',
    entry_point='tutorenvs:FractionArithDigitsEnv',
)

# TODO no pixel fractions yet.
# register(
#     id='FractionArith-v2',
#     entry_point='tutorenvs:FractionArithPixelEnv',
# )

register(
    id='MulticolumnArithSymbolic-v0',
    entry_point='tutorenvs:MultiColumnAdditionDigitsEnv',
)

register(
    id='MulticolumnArithPixel-v0',
    entry_point='tutorenvs:MultiColumnAdditionPixelEnv',
)

register(
    id='MulticolumnArithPercept-v0',
    entry_point='tutorenvs:MultiColumnAdditionPerceptEnv',
)
