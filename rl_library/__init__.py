from gym.envs.registration import register

register(
    id='EasyBlackJack-v0',
    entry_point='rl_library.rl_envs:EasyBlackJackEnv',
)