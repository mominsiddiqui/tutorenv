from gym.envs.registration import register

register(
    id='FractionArithEnv',
    entry_point='TutorEnvs.envs:FractionArithEnv',
)
