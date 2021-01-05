from gym.envs.registration import register
from tutorenvs.fractions import FractionArithNumberEnv  # noqa: F401
from tutorenvs.fractions import FractionArithDigitsEnv  # noqa: F401
from tutorenvs.fractions import FractionArithOppEnv  # noqa: F401
from tutorenvs.multicolumn import MultiColumnAdditionDigitsEnv  # noqa: F401
from tutorenvs.multicolumn import MultiColumnAdditionPixelEnv  # noqa: F401
from tutorenvs.multicolumn import MultiColumnAdditionPerceptEnv  # noqa: F401

register(
    id='FractionArith-v0',
    entry_point='tutorenvs:FractionArithNumberEnv',
)

register(
    id='FractionArith-v1',
    entry_point='tutorenvs:FractionArithDigitsEnv',
)

register(
    id='FractionArith-v2',
    entry_point='tutorenvs:FractionArithOppEnv',
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
