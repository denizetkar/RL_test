from gym.envs.toy_text import BlackjackEnv
from gym import spaces


class EasyBlackJackEnv(BlackjackEnv):
    def __init__(self, natural=False):
        super().__init__(natural)
        self.observation_space = spaces.Discrete(704)

    @staticmethod
    def encode(sum_hand, dealer, usable_ace):
        # (32) 11, 2
        i = sum_hand
        i *= 11
        i += dealer
        i *= 2
        i += usable_ace
        return i

    @staticmethod
    def decode(i):
        out = list()
        out.append(i % 2)
        i = i // 2
        out.append(i % 11)
        i = i // 11
        out.append(i)
        assert 0 <= i < 32
        return reversed(out)

    def step(self, action):
        obs, reward, done, _ = super().step(action)
        return EasyBlackJackEnv.encode(*obs), reward, done, _

    def reset(self):
        obs = super().reset()
        return EasyBlackJackEnv.encode(*obs)

    def render(self, mode='human'):
        raise NotImplementedError
